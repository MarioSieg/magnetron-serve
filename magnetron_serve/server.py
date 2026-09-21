# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import tempfile
import time
import uuid

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from magnetron import Tensor
from magnetron_models.diffusion import ImageGenEngine
from magnetron_models.models import DIFFUSION_MODELS_MAP, MODELS_MAP
from rich.console import Console

from magnetron_serve import registry
from magnetron_serve.pool import ImageGeneration, ImageRequest, Job, ModelPool, PoolBusy, TextGeneration, WrongModelKind

console = Console()

_MAX_BODY = 32 * 1024 * 1024
_MAX_IMAGES = 8
_IMAGE_FORMATS: dict[str, str] = {'png': '.png', 'jpeg': '.jpg', 'jpg': '.jpg'}


class RequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class ClientGone(Exception):
    """The socket died mid-stream. Nothing is wrong; stop generating."""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = '127.0.0.1'
    port: int = 11434


def _as_float(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if payload.get(key) is not None:
            return float(payload[key])
    return None


def _as_int(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if payload.get(key) is not None:
            return int(payload[key])
    return None


def _messages_to_turns(messages: object, default_system: str) -> tuple[str, list[tuple[str, str]]]:
    """Split an OpenAI message list into the system prompt and the turns around it."""
    if not isinstance(messages, list) or not messages:
        raise RequestError(HTTPStatus.BAD_REQUEST, 'messages must be a non-empty list')
    system_parts: list[str] = []
    turns: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
            raise RequestError(HTTPStatus.BAD_REQUEST, 'every message needs a role and a content')
        role, content = str(message['role']), message['content']
        if not isinstance(content, str):  # The parts-array form carries images we have no tower for.
            raise RequestError(HTTPStatus.BAD_REQUEST, 'message content must be a string')
        if role == 'system':
            system_parts.append(content)
        else:
            turns.append((role, content))
    if not turns:
        raise RequestError(HTTPStatus.BAD_REQUEST, 'messages holds no user turn')
    return '\n\n'.join(system_parts) if system_parts else default_system, turns


def _image_request(payload: dict[str, Any]) -> ImageRequest:
    width = _as_int(payload, 'width')
    height = _as_int(payload, 'height')
    size = payload.get('size')
    if size is not None and (width is None or height is None):
        try:
            w, h = (int(part) for part in str(size).lower().split('x', 1))
        except ValueError as e:
            raise RequestError(HTTPStatus.BAD_REQUEST, 'size must look like 1024x1024') from e
        width, height = width if width is not None else w, height if height is not None else h
    count = _as_int(payload, 'n') or 1
    if not 1 <= count <= _MAX_IMAGES:
        raise RequestError(HTTPStatus.BAD_REQUEST, f'n must be between 1 and {_MAX_IMAGES}')
    negative = payload.get('negative_prompt')
    if negative is not None and not isinstance(negative, str):
        raise RequestError(HTTPStatus.BAD_REQUEST, 'negative_prompt must be a string')
    return ImageRequest(
        height=height,
        width=width,
        steps=_as_int(payload, 'steps', 'num_inference_steps'),
        seed=_as_int(payload, 'seed'),
        negative_prompt=negative,
        guidance_scale=_as_float(payload, 'guidance_scale'),
        count=count,
    )


def _image_format(payload: dict[str, Any]) -> str:
    if payload.get('response_format', 'b64_json') != 'b64_json':
        raise RequestError(HTTPStatus.BAD_REQUEST, 'only response_format "b64_json" is supported, this server hosts no files')
    fmt = str(payload.get('output_format', 'png')).lower()
    if fmt not in _IMAGE_FORMATS:
        raise RequestError(HTTPStatus.BAD_REQUEST, f'output_format must be one of {", ".join(sorted(_IMAGE_FORMATS))}')
    return fmt


def encode_image(pixels: Tensor, fmt: str) -> bytes:
    """The image encoder writes files only, so go through one. .jpg drops the alpha the model paints."""
    with tempfile.TemporaryDirectory(prefix='magnetron-serve-') as tmp:
        path = os.path.join(tmp, f'image{_IMAGE_FORMATS[fmt]}')
        ImageGenEngine.save(pixels, path)
        with open(path, 'rb') as f:
            return f.read()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'magnetron-serve'
    sys_version = ''

    pool: ModelPool
    config: ServerConfig

    def setup(self) -> None:
        super().setup()
        self._responded = False

    def log_message(self, fmt: str, *args: object) -> None:
        pass

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0:
            raise RequestError(HTTPStatus.BAD_REQUEST, 'empty request body')
        if length > _MAX_BODY:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, 'request body too large')
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as e:
            raise RequestError(HTTPStatus.BAD_REQUEST, f'malformed JSON: {e}') from e
        if not isinstance(payload, dict):
            raise RequestError(HTTPStatus.BAD_REQUEST, 'request body must be a JSON object')
        return payload

    def _drain_body(self) -> None:
        remaining = min(int(self.headers.get('Content-Length') or 0), _MAX_BODY)
        while remaining > 0:
            remaining -= len(self.rfile.read(min(remaining, 64 * 1024)))

    def _cors(self) -> None:
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode()
        self._responded = True
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        payload = {'error': {'message': message, 'type': status.phrase, 'code': int(status)}}
        if not self._responded:
            self._send_json(payload, status)
            return
        self._sse_write(json.dumps(payload))
        self._sse_write('[DONE]')
        self._sse_close()

    def _sse_open(self) -> None:
        self._responded = True
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'text/event-stream; charset=utf-8')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('X-Accel-Buffering', 'no')
        self.send_header('Transfer-Encoding', 'chunked')
        self._cors()
        self.end_headers()

    def _sse_write(self, data: str) -> None:
        body = f'data: {data}\n\n'.encode()
        try:
            self.wfile.write(b'%X\r\n' % len(body) + body + b'\r\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError) as e:
            raise ClientGone() from e

    def _sse_close(self) -> None:
        try:
            self.wfile.write(b'0\r\n\r\n')
            self.wfile.flush()
        except BrokenPipeError, ConnectionResetError:
            pass

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors()
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        routes = {
            '/health': self._get_health,
            '/api/health': self._get_health,
            '/v1/models': self._get_models,
            '/api/models': self._get_models,
        }
        self._dispatch(routes)

    def do_POST(self) -> None:  # noqa: N802
        routes = {
            '/v1/chat/completions': self._post_chat,
            '/v1/images/generations': self._post_images,
            '/api/generate': self._post_generate,
            '/api/images': self._post_images,
        }
        self._dispatch(routes)

    def _dispatch(self, routes: dict[str, Any]) -> None:
        path = self.path.split('?', 1)[0].rstrip('/') or '/'
        handler = routes.get(path)
        if handler is None:
            self._drain_body()
            self._send_error(HTTPStatus.NOT_FOUND, f'no route for {self.command} {path}')
            return
        try:
            handler()
        except RequestError as e:
            with contextlib.suppress(ClientGone, BrokenPipeError, ConnectionResetError):
                self._send_error(e.status, e.message)
        except WrongModelKind as e:
            with contextlib.suppress(ClientGone, BrokenPipeError, ConnectionResetError):
                self._send_error(HTTPStatus.BAD_REQUEST, str(e))
        except ClientGone:
            pass
        except BrokenPipeError, ConnectionResetError:
            pass
        except Exception as e:
            console.print_exception()
            with contextlib.suppress(ClientGone, BrokenPipeError, ConnectionResetError):
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f'{type(e).__name__}: {e}')

    def _get_health(self) -> None:
        self._send_json(
            {
                'status': 'ok',
                'default_model': self.pool.default_model,
                'queued': self.pool.waiting,
                'loaded': [
                    {
                        'name': m.name,
                        'kind': m.kind,
                        'device': m.engine.device,
                        'snapshot': m.snapshot,
                        'loaded_at': m.loaded_at,
                        'requests': m.requests,
                    }
                    for m in self.pool.loaded
                ],
            }
        )

    def _get_models(self) -> None:
        installed = {s.repo_id for s in registry.cached_snapshots()}
        specs = [(name, registry.CHAT, spec) for name, spec in MODELS_MAP.items()]
        specs += [(name, registry.IMAGE, spec) for name, spec in DIFFUSION_MODELS_MAP.items()]
        data = [
            {
                'id': name,
                'object': 'model',
                'owned_by': 'magnetron',
                'kind': kind,
                'installed': spec.snapshot_repo_id in installed,
                'repo': spec.snapshot_repo_id,
            }
            for name, kind, spec in sorted(specs)
        ]
        self._send_json({'object': 'list', 'data': data})

    def _target(self, model: str | None) -> registry.Target:
        try:
            return self.pool.resolve(model)
        except (KeyError, ValueError, FileNotFoundError) as e:
            raise RequestError(HTTPStatus.NOT_FOUND, registry.message(e)) from e

    def _submit[J: Job](self, job: J) -> J:
        try:
            self.pool.submit(job)
        except PoolBusy as e:
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, f'{e}, try again shortly') from e
        except WrongModelKind as e:
            raise RequestError(HTTPStatus.BAD_REQUEST, str(e)) from e
        return job

    def _post_chat(self) -> None:
        payload = self._read_json()
        system, turns = _messages_to_turns(payload.get('messages'), self.pool.defaults.system)
        model_name = payload.get('model') or None
        stream = bool(payload.get('stream', False))
        max_tokens = _as_int(payload, 'max_completion_tokens', 'max_tokens')
        temp = _as_float(payload, 'temperature')
        top_k = _as_int(payload, 'top_k')
        completion_id = f'chatcmpl-{uuid.uuid4().hex}'
        created = int(time.time())
        started = time.perf_counter()

        sampling = {'max_tokens': max_tokens, 'temp': temp, 'top_k': top_k}
        job = self._submit(TextGeneration(self._target(model_name), lambda model: model.build_prompt(system, turns), sampling))
        try:
            name = job.wait_ready()
            if not stream:
                text = ''.join(job)
                self._send_json(_chat_completion(completion_id, created, name, text))
                self._log('POST /v1/chat/completions', name, len(text), started, streamed=False)
                return
            self._sse_open()
            self._sse_write(json.dumps(_chunk(completion_id, created, name, {'role': 'assistant'})))
            tokens = 0
            for piece in job:
                self._sse_write(json.dumps(_chunk(completion_id, created, name, {'content': piece})))
                tokens += 1
            self._sse_write(json.dumps(_chunk(completion_id, created, name, {}, finish_reason='stop')))
            self._sse_write('[DONE]')
            self._sse_close()
            self._log('POST /v1/chat/completions', name, tokens, started, streamed=True)
        finally:
            job.close()

    def _post_generate(self) -> None:
        payload = self._read_json()
        prompt = payload.get('prompt')
        if not isinstance(prompt, str) or not prompt:
            raise RequestError(HTTPStatus.BAD_REQUEST, 'prompt must be a non-empty string')
        stream = bool(payload.get('stream', True))
        max_tokens = _as_int(payload, 'max_tokens')
        temp = _as_float(payload, 'temperature')
        top_k = _as_int(payload, 'top_k')
        started = time.perf_counter()

        sampling = {'max_tokens': max_tokens, 'temp': temp, 'top_k': top_k}
        job = self._submit(TextGeneration(self._target(payload.get('model') or None), lambda _: prompt, sampling))
        try:
            name = job.wait_ready()
            if not stream:
                text = ''.join(job)
                self._send_json({'model': name, 'response': text, 'done': True})
                self._log('POST /api/generate', name, len(text), started, streamed=False)
                return
            self._sse_open()
            tokens = 0
            for piece in job:
                self._sse_write(json.dumps({'model': name, 'response': piece, 'done': False}))
                tokens += 1
            self._sse_write(json.dumps({'model': name, 'response': '', 'done': True}))
            self._sse_write('[DONE]')
            self._sse_close()
            self._log('POST /api/generate', name, tokens, started, streamed=True)
        finally:
            job.close()

    def _post_images(self) -> None:
        """OpenAI images/generations. Non-streamed: {data: [{b64_json}]}. Streamed: one image_generation.progress
        event per denoising step, then an image_generation.completed event per image carrying its b64_json."""
        payload = self._read_json()
        prompt = payload.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            raise RequestError(HTTPStatus.BAD_REQUEST, 'prompt must be a non-empty string')
        request = _image_request(payload)
        fmt = _image_format(payload)
        stream = bool(payload.get('stream', False))
        created = int(time.time())
        started = time.perf_counter()

        job = self._submit(ImageGeneration(self._target(payload.get('model') or None), prompt, request))
        try:
            name = job.wait_ready()
            if not stream:
                data = [_image_item(index, pixels, fmt) for index, pixels in job.images()]
                self._send_json({'created': created, 'model': name, 'data': data})
                self._log_images(name, data, started)
                return
            self._sse_open()
            data = []
            for kind, event in job.events():
                if kind == 'progress':
                    index, done, total = event
                    self._sse_write(json.dumps({'type': 'image_generation.progress', 'index': index, 'step': done, 'total': total}))
                elif kind == 'image':
                    item = _image_item(*event, fmt)
                    data.append(item)
                    self._sse_write(json.dumps({'type': 'image_generation.completed', 'created_at': created, 'model': name, **item}))
            self._sse_write('[DONE]')
            self._sse_close()
            self._log_images(name, data, started)
        finally:
            job.close()

    def _log_images(self, model: str, data: list[dict[str, Any]], started: float) -> None:
        elapsed = time.perf_counter() - started
        sizes = ', '.join(item['size'] for item in data)
        console.print(f'[dim]POST /v1/images/generations {model} - {len(data)} image(s) [{sizes}] in {elapsed:.1f}s[/dim]')

    def _log(self, what: str, model: str, count: int, started: float, streamed: bool) -> None:
        elapsed = time.perf_counter() - started
        unit = 'tok' if streamed else 'chars'
        rate = f', {count / elapsed:.2f} {unit}/s' if elapsed > 0 and count else ''
        console.print(f'[dim]{what} {model} - {count} {unit} in {elapsed:.3f}s{rate}[/dim]')


def _image_item(index: int, pixels: Tensor, fmt: str) -> dict[str, Any]:
    _, height, width = pixels.shape
    return {
        'index': index,
        'b64_json': base64.b64encode(encode_image(pixels, fmt)).decode('ascii'),
        'output_format': 'jpeg' if fmt == 'jpg' else fmt,
        'size': f'{width}x{height}',
    }


def _chunk(completion_id: str, created: int, model: str, delta: dict[str, str], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        'id': completion_id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': model,
        'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}],
    }


def _chat_completion(completion_id: str, created: int, model: str, text: str) -> dict[str, Any]:
    return {
        'id': completion_id,
        'object': 'chat.completion',
        'created': created,
        'model': model,
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': 'stop'}],
    }


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request: object, client_address: object) -> None:
        if isinstance(sys.exception(), (BrokenPipeError, ConnectionResetError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def serve(pool: ModelPool, config: ServerConfig) -> None:
    handler = type('MagnetronHandler', (_Handler,), {'pool': pool, 'config': config})
    httpd = _Server((config.host, config.port), handler)
    console.print(f'[bold green]Listening[/] on http://{config.host}:{config.port}')
    console.print('[dim]POST /v1/chat/completions | POST /v1/images/generations | POST /api/generate | GET /v1/models | GET /health[/dim]')
    if pool.default_model is not None:
        console.print(f'[dim]Default model: {pool.default_model}[/dim]')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        console.print('\n[dim]Shutting down.[/dim]')
    finally:
        httpd.server_close()
        pool.shutdown()
