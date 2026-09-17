# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import contextlib
import json
import sys
import time
import uuid

from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from magnetron_models.models import MODELS_MAP, ModelBase
from rich.console import Console

from magnetron_serve import registry
from magnetron_serve.pool import Generation, ModelPool, PoolBusy

console = Console()

_MAX_BODY = 32 * 1024 * 1024


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
            '/api/generate': self._post_generate,
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
                    {'name': m.name, 'device': m.engine.device, 'snapshot': m.snapshot, 'loaded_at': m.loaded_at, 'requests': m.requests}
                    for m in self.pool.loaded
                ],
            }
        )

    def _get_models(self) -> None:
        installed = {s.repo_id for s in registry.cached_snapshots()}
        data = [
            {
                'id': name,
                'object': 'model',
                'owned_by': 'magnetron',
                'installed': spec.snapshot_repo_id in installed,
                'repo': spec.snapshot_repo_id,
            }
            for name, spec in sorted(MODELS_MAP.items())
        ]
        self._send_json({'object': 'list', 'data': data})

    def _submit(
        self,
        model: str | None,
        build_prompt: Callable[[ModelBase], str],
        max_tokens: int | None,
        temp: float | None,
        top_k: int | None,
    ) -> Generation:
        try:
            return self.pool.submit(model, build_prompt, max_tokens, temp, top_k)
        except PoolBusy as e:
            raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, f'{e}, try again shortly') from e
        except (KeyError, ValueError, FileNotFoundError) as e:
            raise RequestError(HTTPStatus.NOT_FOUND, registry.message(e)) from e

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

        job = self._submit(model_name, lambda model: model.build_prompt(system, turns), max_tokens, temp, top_k)
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

        job = self._submit(payload.get('model') or None, lambda _: prompt, max_tokens, temp, top_k)
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

    def _log(self, what: str, model: str, count: int, started: float, streamed: bool) -> None:
        elapsed = time.perf_counter() - started
        unit = 'tok' if streamed else 'chars'
        rate = f', {count / elapsed:.2f} {unit}/s' if elapsed > 0 and count else ''
        console.print(f'[dim]{what} {model} - {count} {unit} in {elapsed:.3f}s{rate}[/dim]')


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
    console.print('[dim]POST /v1/chat/completions | POST /api/generate | GET /v1/models | GET /health[/dim]')
    if pool.default_model is not None:
        console.print(f'[dim]Default model: {pool.default_model}[/dim]')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        console.print('\n[dim]Shutting down.[/dim]')
    finally:
        httpd.server_close()
        pool.shutdown()
