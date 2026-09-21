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

from magnetron import Tensor
from magnetron_models.diffusion import ImageGenConfig, ImageGenEngine
from magnetron_models.inference import AUTO_DVC, InferenceConfig, InferenceEngine
from magnetron_models.models import ModelBase

from magnetron_serve import registry
from magnetron_serve.registry import CHAT, IMAGE, Target

_CHUNK_BUFFER = 256
_CANCEL_POLL = 0.1

Engine = InferenceEngine | ImageGenEngine


class PoolBusy(Exception):
    pass


class WrongModelKind(ValueError):
    """A chat request reached a text-to-image model, or an image request a chat model."""


@dataclass(frozen=True, slots=True)
class ImageDefaults:
    height: int = 1024
    width: int = 1024
    steps: int | None = None  # None: the checkpoint's own setting
    negative_prompt: str | None = None
    guidance_scale: float = 1.0
    kv_cache: bool = True
    keep_loaded: bool = False  # Hold all three networks instead of loading each just in time


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
    image: ImageDefaults = ImageDefaults()


@dataclass(frozen=True, slots=True)
class ImageRequest:
    """Per-request overrides; None falls back to the engine's ImageDefaults."""

    height: int | None = None
    width: int | None = None
    steps: int | None = None
    seed: int | None = None
    negative_prompt: str | None = None
    guidance_scale: float | None = None
    count: int = 1


@dataclass(slots=True)
class LoadedModel:
    name: str
    kind: str
    engine: Engine
    snapshot: str
    loaded_at: float = field(default_factory=time.time)
    requests: int = 0


def chat_engine_config(snapshot: str, defaults: EngineDefaults) -> InferenceConfig:
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


def image_engine_config(snapshot: str, defaults: EngineDefaults) -> ImageGenConfig:
    image = defaults.image
    return ImageGenConfig(
        device=defaults.device,
        dtype=defaults.dtype,
        model=None,
        snapshot=snapshot,
        seed=defaults.seed,
        height=image.height,
        width=image.width,
        num_inference_steps=image.steps,
        negative_prompt=image.negative_prompt,
        guidance_scale=image.guidance_scale,
        use_kv_cache=image.kv_cache,
        offload=not image.keep_loaded,
    )


def load_model(target: Target, defaults: EngineDefaults) -> LoadedModel:
    snapshot: str = registry.install(target, defaults.dtype)
    kind: str = registry.kind_of(target, snapshot)
    engine: Engine
    if kind == IMAGE:
        engine = ImageGenEngine(image_engine_config(snapshot, defaults))
    else:
        engine = InferenceEngine(chat_engine_config(snapshot, defaults))
    return LoadedModel(name=target.name, kind=kind, engine=engine, snapshot=snapshot)


def require_kind(name: str, actual: str, wanted: str) -> None:
    if actual == wanted:
        return
    if actual == IMAGE:
        raise WrongModelKind(f'{name} is a text-to-image model, it cannot chat; ask it for an image instead')
    raise WrongModelKind(f'{name} is a chat model, it cannot paint; ask it for a completion instead')


class Job:
    """One unit of work for a model worker: a preload, a chat completion or a painting.

    The worker thread runs execute() and pushes events through emit(); the request thread pulls them back out with
    events(), wait_ready() or the subclass' own iterators. close() from either side ends it early.
    """

    def __init__(self, target: Target) -> None:
        self.target = target
        self.model_name: str = target.name
        self.cancelled = threading.Event()
        self._out: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=_CHUNK_BUFFER)
        self._finished = False

    @property
    def kind(self) -> str | None:
        return None

    def execute(self, model: LoadedModel) -> None:
        pass

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

    def events(self) -> Iterator[tuple[str, Any]]:
        while not self._finished:
            kind, payload = self._next()
            if kind != 'done':
                yield kind, payload

    def close(self) -> None:
        self.cancelled.set()


class Preload(Job):
    pass


class TextGeneration(Job):
    def __init__(self, target: Target, build_prompt: Callable[[ModelBase], str], sampling: dict[str, Any]) -> None:
        super().__init__(target)
        self.build_prompt = build_prompt
        self.sampling = sampling

    @property
    def kind(self) -> str:
        return CHAT

    def execute(self, model: LoadedModel) -> None:
        require_kind(model.name, model.kind, CHAT)
        assert isinstance(model.engine, InferenceEngine)
        for chunk in model.engine.gen_stream(self.build_prompt(model.engine.model), reset_cache=True, **self.sampling):
            if not self.emit('chunk', chunk):
                return

    def __iter__(self) -> Iterator[str]:
        for kind, payload in self.events():
            if kind == 'chunk':
                yield payload


class _Aborted(Exception):
    pass


class ImageGeneration(Job):
    def __init__(self, target: Target, prompt: str, request: ImageRequest) -> None:
        super().__init__(target)
        self.prompt = prompt
        self.request = request

    @property
    def kind(self) -> str:
        return IMAGE

    def execute(self, model: LoadedModel) -> None:
        require_kind(model.name, model.kind, IMAGE)
        assert isinstance(model.engine, ImageGenEngine)
        engine, req = model.engine, self.request
        base_seed: int = engine.config.seed if req.seed is None else req.seed
        for index in range(req.count):

            def on_step(done: int, total: int, index: int = index) -> None:
                if not self.emit('progress', (index, done, total)):
                    raise _Aborted()

            try:
                pixels = engine.generate(
                    self.prompt,
                    height=req.height,
                    width=req.width,
                    num_inference_steps=req.steps,
                    seed=base_seed + index,
                    negative_prompt=req.negative_prompt,
                    guidance_scale=req.guidance_scale,
                    on_step=on_step,
                )
            except _Aborted:
                return
            if not self.emit('image', (index, pixels)):
                return

    def images(self) -> Iterator[tuple[int, Tensor]]:
        for kind, payload in self.events():
            if kind == 'image':
                yield payload


_ADMIT_POLL = 0.05
_ADMIT_TIMEOUT = 30.0
_STOP_TIMEOUT = 30.0


class _ModelWorker:
    def __init__(self, target: Target, defaults: EngineDefaults) -> None:
        self.target = target
        self.defaults = defaults
        self.name: str = target.name
        self.model: LoadedModel | None = None
        self.jobs: queue.Queue[Job | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=f'magnetron-gen-{target.name}', daemon=True)

    @property
    def idle(self) -> bool:
        return self.jobs.unfinished_tasks == 0

    @property
    def waiting(self) -> int:
        return self.jobs.qsize()

    def start(self) -> None:
        self._thread.start()

    def submit(self, job: Job) -> None:
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

    def _execute(self, job: Job) -> None:
        try:
            model = self._ensure_loaded()
            if not job.emit('ready', model.name):
                return
            if not isinstance(job, Preload):
                model.requests += 1
            job.execute(model)
            job.emit('done')
        except Exception as e:
            job.emit('error', e)

    def _ensure_loaded(self) -> LoadedModel:
        if self.model is None:
            self.model = load_model(self.target, self.defaults)
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

    def submit(self, job: Job) -> Job:
        """Queue a job on its model's worker, loading the model first if it is not resident.

        A job whose model kind is already known to be wrong is refused here; a bare Hub repo only reveals its kind
        once on disk, so that mismatch surfaces from the worker instead.
        """
        if job.kind is not None and job.target.kind is not None:
            require_kind(job.target.name, job.target.kind, job.kind)
        queued = self.waiting
        if queued >= self.queue_limit:
            raise PoolBusy(f'{queued} requests already queued')
        return self._enqueue(job)

    def preload(self, name: str | None = None) -> str:
        job = self._enqueue(Preload(self.resolve(name)))
        try:
            return job.wait_ready()
        finally:
            job.close()

    def _enqueue(self, job: Job) -> Job:
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
