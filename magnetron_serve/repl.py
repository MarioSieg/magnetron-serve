# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import time

from pathlib import Path

from magnetron_models.diffusion import ImageGenEngine
from magnetron_models.inference import InferenceEngine
from magnetron_models.models import ModelBase
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.text import Text

console = Console()

_HELP: dict[str, str] = {
    '/exit': 'leave the REPL',
    '/clear': 'start a new conversation',
    '/help': 'show this list',
}


class Conversation:
    def __init__(self, model: ModelBase) -> None:
        self.history: list[tuple[str, str]] = []
        self.model = model

    def push_assistant(self, reply: str) -> None:
        self.history.append(('assistant', reply))

    def push_user(self, reply: str) -> None:
        self.history.append(('user', reply))

    def clear(self) -> None:
        self.history.clear()

    def build_prompt(self, system: str) -> str:
        return self.model.build_prompt(system, self.history)

    def build_initial_prompt(self, system: str) -> str:
        return self.model.build_system(system)

    def build_user_turn(self, user: str) -> str:
        return self.model.build_user_turn(user)

    def build_assistant_end(self) -> str:
        return '<|im_end|>\n'


def repl(engine: InferenceEngine, title: str, system: str) -> None:
    console.print(
        Panel.fit(
            Text(f'Magnetron {title} REPL', style='bold white') + Text('\n' + ', '.join(_HELP), style='dim'),
            border_style='cyan',
        )
    )
    conversation = Conversation(engine.model)
    while True:
        try:
            user = Prompt.ask('[bold cyan]You[/]').strip()
        except EOFError, KeyboardInterrupt:
            console.print()
            break
        if not user:
            continue
        if user == '/exit':
            break
        if user == '/help':
            for cmd, what in _HELP.items():
                console.print(f'[bold]{cmd}[/] - {what}')
            continue
        if user == '/clear':
            conversation.clear()
            console.print('[dim]Conversation cleared.[/dim]')
            continue
        is_first_turn = len(conversation.history) == 0
        conversation.push_user(user)
        console.print(Rule(style='dim'))
        console.print('[bold magenta]Assistant[/]:', end=' ')
        start = time.perf_counter()
        parts: list[str] = []
        count = 0
        try:
            prompt: str = (
                conversation.build_initial_prompt(system) + conversation.build_user_turn(user)
                if is_first_turn
                else conversation.build_assistant_end() + conversation.build_user_turn(user)
            )
            for chunk in engine.gen_stream(prompt, reset_cache=is_first_turn):
                parts.append(chunk)
                console.print(chunk, style='bold white', end='')
                count += 1
            conversation.push_assistant(''.join(parts))
        except KeyboardInterrupt:
            console.print('\n[dim]Interrupted.[/dim]')
            continue
        if count > 0:
            elapsed = time.perf_counter() - start
            console.print(f'\n[dim]Tokens/s: {count / elapsed:.2f}, {count} tokens in {elapsed:.3f}s[/dim]')
        else:
            console.print()


_IMAGE_HELP: dict[str, str] = {
    '/exit': 'leave the REPL',
    '/seed N': 'seed the next images from N, counting up',
    '/size WxH': 'paint at W by H pixels',
    '/steps N': 'denoise in N steps',
    '/help': 'show this list',
}


def _numbered(out: str, index: int) -> str:
    path = Path(out)
    return str(path.with_name(f'{path.stem}-{index}{path.suffix or ".png"}'))


def image_repl(engine: ImageGenEngine, title: str, out: str) -> None:
    console.print(
        Panel.fit(
            Text(f'Magnetron {title} REPL', style='bold white') + Text('\n' + ', '.join(_IMAGE_HELP), style='dim'),
            border_style='cyan',
        )
    )
    seed: int = engine.config.seed
    height: int | None = None
    width: int | None = None
    steps: int | None = None
    index = 1
    while True:
        try:
            user = Prompt.ask('[bold cyan]Prompt[/]').strip()
        except EOFError, KeyboardInterrupt:
            console.print()
            break
        if not user:
            continue
        if user == '/exit':
            break
        if user == '/help':
            for cmd, what in _IMAGE_HELP.items():
                console.print(f'[bold]{cmd}[/] - {what}')
            continue
        if user.startswith('/'):
            command, _, value = user.partition(' ')
            try:
                if command == '/seed':
                    seed = int(value)
                elif command == '/size':
                    w, h = (int(part) for part in value.lower().split('x', 1))
                    width, height = w, h
                elif command == '/steps':
                    steps = int(value)
                else:
                    console.print(f'[yellow]Unknown command {command}, /help lists them.[/]')
                    continue
            except ValueError:
                console.print(f'[yellow]{command} needs a number, like {next(k for k in _IMAGE_HELP if k.startswith(command))}.[/]')
                continue
            console.print(f'[dim]{command[1:]} set.[/dim]')
            continue
        console.print(Rule(style='dim'))
        start = time.perf_counter()
        try:
            image = engine.generate(user, height=height, width=width, num_inference_steps=steps, seed=seed)
        except KeyboardInterrupt:
            console.print('\n[dim]Interrupted.[/dim]')
            continue
        except ValueError as e:
            console.print(f'[red]{e}[/]')
            continue
        path = _numbered(out, index)
        engine.save(image, path)
        _, h, w = image.shape
        console.print(f'[green]Saved[/] {w}x{h} image to [bold]{path}[/bold] [dim](seed {seed}, {time.perf_counter() - start:.1f}s)[/dim]')
        seed += 1
        index += 1
