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
                else conversation.build_user_turn(user)
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
