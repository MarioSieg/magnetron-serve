# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from magnetron_serve import registry
from magnetron_serve.pool import EngineDefaults, ModelPool, build_engine
from magnetron_serve.registry import Target
from magnetron_serve.repl import repl
from magnetron_serve.server import ServerConfig, serve

console = Console()

_DEFAULTS = EngineDefaults()


def _human(size: int) -> str:
    value = float(size)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.1f} {unit}' if unit != 'B' else f'{int(value)} B'
        value /= 1024
    return f'{value:.1f} TiB'


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--device',
        type=str,
        default=_DEFAULTS.device,
        metavar='DEVICE',
        help='auto (the fastest accelerator there is, else cpu), or cpu, cuda, or a specific one like cuda:1',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default=_DEFAULTS.dtype,
        choices=['float16', 'bfloat16', 'float32'],
        help='Picks between several snapshots in one repo; a snapshot always runs in the dtype it was written in',
    )
    parser.add_argument('--max-tokens', type=int, default=_DEFAULTS.max_tokens, help='Maximum number of new tokens to generate')
    parser.add_argument('--temp', type=float, default=_DEFAULTS.temp, help='Sampling temperature; 0 is greedy, above it samples the top-k')
    parser.add_argument('--top-k', type=int, default=_DEFAULTS.top_k, help='Top-k sampling')
    parser.add_argument('--seed', type=int, default=_DEFAULTS.seed, help='Random seed for reproducibility')
    parser.add_argument('--system', type=str, default=_DEFAULTS.system, help='System prompt')
    parser.add_argument('--repo-id', type=str, default=None, help='HF repo to pull the tokenizer from, overrides the one in the snapshot')


def _defaults_from(args: argparse.Namespace) -> EngineDefaults:
    return EngineDefaults(
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_k=args.top_k,
        system=args.system,
        repo_id=args.repo_id,
    )


def _resolve(name: str) -> Target:
    try:
        return registry.resolve(name)
    except (KeyError, FileNotFoundError) as e:
        raise SystemExit(registry.message(e)) from e


def cmd_list(args: argparse.Namespace) -> None:
    table = Table(title='Models', title_style='bold', header_style='bold cyan', box=None, pad_edge=False)
    table.add_column('NAME')
    table.add_column('STATUS')
    table.add_column('SIZE', justify='right')
    table.add_column('SNAPSHOT')
    rows = 0
    for name, spec, local in registry.catalog(args.dtype):
        if local is None and not args.available:
            continue
        status = '[green]installed[/]' if local is not None else '[dim]available[/]'
        table.add_row(name, status, _human(local.size) if local else '-', local.filename if local else spec.snapshot_repo_id)
        rows += 1
    for stray in registry.strays():
        table.add_row(stray.repo_id, '[green]installed[/]', _human(stray.size), stray.filename)
        rows += 1
    if rows == 0:
        console.print('No models installed. [dim]`magnetron-serve list --available` shows what can be installed.[/dim]')
        return
    console.print(table)
    if not args.available:
        console.print('[dim]--available also lists models that are not installed yet.[/dim]')


def cmd_install(args: argparse.Namespace) -> None:
    for name in args.model:
        target = _resolve(name)
        if target.is_local_file:
            console.print(f'{name} is already a local file, nothing to download')
            continue
        local = registry.installed(target, args.dtype)
        if local is not None and not args.force:
            console.print(f'[green]{target.name}[/] already installed [dim]({local.path})[/dim]')
            continue
        path = registry.install(target, args.dtype)
        console.print(f'[green]Installed[/] {target.name} [dim]-> {path}[/dim]')


def cmd_remove(args: argparse.Namespace) -> None:
    for name in args.model:
        target = _resolve(name)
        try:
            freed = registry.uninstall(target)
        except (KeyError, ValueError) as e:
            console.print(f'[yellow]{registry.message(e)}[/]')
            continue
        console.print(f'[green]Removed[/] {target.name} [dim](freed {_human(freed)})[/dim]')


def cmd_run(args: argparse.Namespace) -> None:
    target = _resolve(args.model)
    if registry.installed(target, args.dtype) is None:
        console.print(f'[dim]{target.name} is not installed, pulling it now.[/dim]')
    engine, _ = build_engine(target, _defaults_from(args))
    if args.prompt:
        reply = engine.gen_one_shot(engine.model.build_prompt(args.system, [('user', args.prompt)]), reset_cache=True)
        console.print(f'\n\nAnswer: {reply}', style='bold green')
    else:
        repl(engine, target.name, args.system)


def cmd_serve(args: argparse.Namespace) -> None:
    default_model: str | None = args.model
    if default_model is not None:
        _resolve(default_model)
    pool = ModelPool(
        _defaults_from(args),
        default_model=default_model,
        max_loaded=args.max_loaded,
        queue_limit=args.queue_limit,
    )
    if default_model is not None and not args.lazy:
        pool.preload()
    serve(pool, ServerConfig(host=args.host, port=args.port))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='magnetron-serve', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='Run a model in an interactive chat REPL')
    run.add_argument('model', type=str, help='Registry name, Hub repo id, or path to a .mag snapshot')
    run.add_argument('--prompt', type=str, default=None, help='Answer this one prompt and exit instead of opening the REPL')
    _add_engine_args(run)
    run.set_defaults(func=cmd_run)

    serve_cmd = sub.add_parser('serve', help='Start the SSE inference server')
    serve_cmd.add_argument('model', type=str, nargs='?', default=None, help='Model to serve by default when a request names none')
    serve_cmd.add_argument('--host', type=str, default='127.0.0.1', help='Address to bind')
    serve_cmd.add_argument('--port', type=int, default=11434, help='Port to bind')
    serve_cmd.add_argument('--max-loaded', type=int, default=1, help='How many models may sit in memory at once')
    serve_cmd.add_argument('--queue-limit', type=int, default=32, help='Requests allowed to wait before new ones get a 503')
    serve_cmd.add_argument('--lazy', action='store_true', help='Load the default model on first request instead of at startup')
    _add_engine_args(serve_cmd)
    serve_cmd.set_defaults(func=cmd_serve)

    listing = sub.add_parser('list', help='List installed models')
    listing.add_argument('--available', action='store_true', help='Also list models that are available but not installed')
    listing.add_argument('--dtype', type=str, default=_DEFAULTS.dtype, choices=['float16', 'bfloat16', 'float32'])
    listing.set_defaults(func=cmd_list)

    install = sub.add_parser('install', help='Download a model snapshot from the Hub')
    install.add_argument('model', type=str, nargs='+', help='Registry name or Hub repo id')
    install.add_argument('--dtype', type=str, default=_DEFAULTS.dtype, choices=['float16', 'bfloat16', 'float32'])
    install.add_argument('--force', action='store_true', help='Re-check the Hub even when the snapshot is already cached')
    install.set_defaults(func=cmd_install)

    remove = sub.add_parser('rm', help='Delete a downloaded model snapshot')
    remove.add_argument('model', type=str, nargs='+', help='Registry name or Hub repo id')
    remove.set_defaults(func=cmd_remove)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print('\n[dim]Interrupted.[/dim]')
        sys.exit(130)
    except (RuntimeError, ValueError, OSError) as e:
        console.print(f'[red]{registry.message(e)}[/]')
        sys.exit(1)


if __name__ == '__main__':
    main()
