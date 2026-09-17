# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

"""Loading models once, and running every generation on the thread that owns them.

Magnetron binds its context to the first thread that uses it and refuses tensor work from any other,
so a threaded server cannot generate inside its request handlers. The pool therefore owns one
inference thread: it creates the context, loads the models, and runs every generation, while request
handlers only hand it jobs and read chunks back off a queue.

That single thread is also the whole scheduler. Jobs are served in arrival order, each holds the
device for the length of its stream, and a model is only ever swapped between jobs, so nothing is
evicted out from under a running generation. Interleaving would buy nothing anyway: generation
saturates the device, so sharing it would only trade throughput for latency.
"""

from __future__ import annotations

import gc
import queue
import threading
import time

from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from magnetron_models.inference import AUTO_DVC, InferenceConfig, InferenceEngine
from magnetron_models.models import ModelBase

from magnetron_serve import registry
from magnetron_serve.registry import Target

_CHUNK_BUFFER = 256  # Chunks the worker may run ahead of a slow client before it has to wait for one.
_CANCEL_POLL = 0.1  # How long a blocked worker waits before re-checking whether its client is still there.


class PoolBusy(Exception):
    """More work is already queued than the pool is willing to hold."""


@dataclass(frozen=True, slots=True)
class EngineDefaults:
    """Sampling and placement a request inherits when it does not say otherwise."""

    device: str = AUTO_DVC
    dtype: str = 'bfloat16'
    seed: int = 3407
    max_tokens: int = 1024
    temp: float = 0.6
    top_k: int = 200
    system: str = 'You are a helpful assistant.'
    repo_id: str | None = None


@dataclass(slots=True)
class LoadedModel:
    name: str
    engine: InferenceEngine
    snapshot: str
    loaded_at: float = field(default_factory=time.time)
    requests: int = 0


def engine_config(target: Target, snapshot: str, defaults: EngineDefaults) -> InferenceConfig:
    return InferenceConfig(
        device=defaults.device,
        max_tokens=defaults.max_tokens,
        temp=defaults.temp,
        top_k=defaults.top_k,
        seed=defaults.seed,
        model=None,  # The snapshot is on disk by now, and it names its own architecture, config and dtype.
        dtype=defaults.dtype,
        repo_id=defaults.repo_id,
        snapshot=snapshot,
    )


def build_engine(target: Target, defaults: EngineDefaults) -> tuple[InferenceEngine, str]:
    snapshot: str = registry.install(target, defaults.dtype)
    return InferenceEngine(engine_config(target, snapshot, defaults)), snapshot


class Generation:
    def __init__(self, target: Target, build_prompt: Callable[[ModelBase], str] | None, sampling: dict[str, Any]) -> None:
        self.target = target
        self.build_prompt = build_prompt
        self.sampling = sampling
        self.model_name: str = target.name
        self.cancelled = threading.Event()
        self._out: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=_CHUNK_BUFFER)
        self._finished = False

    def emit(self, kind: str, payload: object = None) -> bool:
        while not self.cancelled.is_set():
            try:
                self._out.put((kind, payload), timeout=_CANCEL_POLL)
                return True
            except queue.Full:
                continue
        return False

    def _next(self) -> tuple[str, Any]:
        kind, payload = self._out.get()
        if kind == 'error':
            self._finished = True
            raise payload
        if kind == 'done':
            self._finished = True
        return kind, payload

    def wait_ready(self) -> str:
        kind, payload = self._next()
        if kind != 'ready':
            raise RuntimeError(f'Expected the model to become ready, got {kind}')
        self.model_name = payload
        return payload

    def __iter__(self) -> Iterator[str]:
        while not self._finished:
            kind, payload = self._next()
            if kind == 'chunk':
                yield payload

    def close(self) -> None:
        self.cancelled.set()


class ModelPool:
    def __init__(
        self,
        defaults: EngineDefaults,
        default_model: str | None = None,
        max_loaded: int = 1,
        queue_limit: int = 32,
    ) -> None:
        if max_loaded < 1:
            raise ValueError('max_loaded must be at least 1')
        self.defaults = defaults
        self.default_model = default_model
        self.max_loaded = max_loaded
        self.queue_limit = queue_limit
        self._jobs: queue.Queue[Generation | None] = queue.Queue()
        self._loaded: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, name='magnetron-inference', daemon=True)
        self._worker.start()

    @property
    def loaded(self) -> list[LoadedModel]:
        with self._lock:
            return list(self._loaded.values())

    @property
    def waiting(self) -> int:
        return self._jobs.qsize()

    def resolve(self, name: str | None) -> Target:
        chosen: str | None = name or self.default_model
        if chosen is None:
            raise ValueError('No model requested and the server was started without a default one')
        return registry.resolve(chosen)

    def submit(
        self,
        name: str | None,
        build_prompt: Callable[[ModelBase], str],
        max_tokens: int | None = None,
        temp: float | None = None,
        top_k: int | None = None,
    ) -> Generation:
        if self._jobs.qsize() >= self.queue_limit:
            raise PoolBusy(f'{self._jobs.qsize()} requests already queued')
        return self._enqueue(Generation(self.resolve(name), build_prompt, {'max_tokens': max_tokens, 'temp': temp, 'top_k': top_k}))

    def preload(self, name: str | None = None) -> str:
        job = self._enqueue(Generation(self.resolve(name), None, {}))
        try:
            return job.wait_ready()
        finally:
            job.close()

    def _enqueue(self, job: Generation) -> Generation:
        self._jobs.put(job)
        return job

    def shutdown(self) -> None:
        self._jobs.put(None)
        self._worker.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            self._execute(job)

    def _execute(self, job: Generation) -> None:
        try:
            model = self._load(job.target)
            if not job.emit('ready', model.name):
                return
            if job.build_prompt is None:
                job.emit('done')
                return
            model.requests += 1
            for chunk in model.engine.gen_stream(job.build_prompt(model.engine.model), reset_cache=True, **job.sampling):
                if not job.emit('chunk', chunk):
                    return
            job.emit('done')
        except Exception as e:
            job.emit('error', e)

    def _load(self, target: Target) -> LoadedModel:
        with self._lock:
            held = self._loaded.get(target.name)
            if held is not None:
                self._loaded.move_to_end(target.name)
                return held
            while len(self._loaded) >= self.max_loaded:
                evicted = self._loaded.popitem(last=False)[1]
                del evicted
                gc.collect()
        engine, snapshot = build_engine(target, self.defaults)
        model = LoadedModel(name=target.name, engine=engine, snapshot=snapshot)
        with self._lock:
            self._loaded[target.name] = model
        return model
