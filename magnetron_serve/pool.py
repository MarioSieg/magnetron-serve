# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

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

_CHUNK_BUFFER = 256
_CANCEL_POLL = 0.1


class PoolBusy(Exception):
    pass


@dataclass(frozen=True, slots=True)
class EngineDefaults:
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


_ADMIT_POLL = 0.05
_ADMIT_TIMEOUT = 30.0
_STOP_TIMEOUT = 30.0


class _ModelWorker:
    def __init__(self, target: Target, defaults: EngineDefaults) -> None:
        self.target = target
        self.defaults = defaults
        self.name: str = target.name
        self.model: LoadedModel | None = None
        self.jobs: queue.Queue[Generation | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=f'magnetron-gen-{target.name}', daemon=True)

    @property
    def idle(self) -> bool:
        return self.jobs.unfinished_tasks == 0

    @property
    def waiting(self) -> int:
        return self.jobs.qsize()

    def start(self) -> None:
        self._thread.start()

    def submit(self, job: Generation) -> None:
        self.jobs.put(job)

    def stop(self, timeout: float = _STOP_TIMEOUT) -> None:
        self.jobs.put(None)
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                self._execute(job)
            finally:
                self.jobs.task_done()

    def _execute(self, job: Generation) -> None:
        try:
            model = self._ensure_loaded()
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

    def _ensure_loaded(self) -> LoadedModel:
        if self.model is None:
            engine, snapshot = build_engine(self.target, self.defaults)
            self.model = LoadedModel(name=self.target.name, engine=engine, snapshot=snapshot)
        return self.model


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
        self._workers: OrderedDict[str, _ModelWorker] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def loaded(self) -> list[LoadedModel]:
        with self._lock:
            return [w.model for w in self._workers.values() if w.model is not None]

    @property
    def waiting(self) -> int:
        with self._lock:
            return sum(w.waiting for w in self._workers.values())

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
        queued = self.waiting
        if queued >= self.queue_limit:
            raise PoolBusy(f'{queued} requests already queued')
        return self._enqueue(Generation(self.resolve(name), build_prompt, {'max_tokens': max_tokens, 'temp': temp, 'top_k': top_k}))

    def preload(self, name: str | None = None) -> str:
        job = self._enqueue(Generation(self.resolve(name), None, {}))
        try:
            return job.wait_ready()
        finally:
            job.close()

    def _enqueue(self, job: Generation) -> Generation:
        deadline = time.monotonic() + _ADMIT_TIMEOUT
        while True:
            retired: list[_ModelWorker] = []
            with self._lock:
                worker = self._workers.get(job.target.name)
                if worker is None:
                    while len(self._workers) >= self.max_loaded:
                        victim = next((n for n, w in self._workers.items() if w.idle), None)
                        if victim is None:
                            break
                        retired.append(self._workers.pop(victim))
                    if len(self._workers) < self.max_loaded:
                        worker = _ModelWorker(job.target, self.defaults)
                        self._workers[job.target.name] = worker
                        worker.start()
                if worker is not None:
                    self._workers.move_to_end(job.target.name)
                    worker.submit(job)
            for dead in retired:
                dead.stop()
            if retired:
                gc.collect()
            if worker is not None:
                return job
            if time.monotonic() >= deadline:
                raise PoolBusy(f'all {self.max_loaded} model slots are busy')
            time.sleep(_ADMIT_POLL)

    def shutdown(self) -> None:
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop()
