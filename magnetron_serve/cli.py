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
import time

from magnetron_models.diffusion import ImageGenEngine
from magnetron_models.inference import InferenceEngine
from rich.console import Console
from rich.table import Table

from magnetron_serve import registry
from magnetron_serve.pool import EngineDefaults, ImageDefaults, ModelPool, load_model
from magnetron_serve.registry import CHAT, IMAGE, Target
from magnetron_serve.repl import image_repl, repl
from magnetron_serve.server import ServerConfig, serve

console = Console()

_DEFAULTS = EngineDefaults()
_IMAGE_DEFAULTS = ImageDefaults()
_DTYPES = ['float16', 'bfloat16', 'float32']


def _human(size: int) -> str:
    value = float(size)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.1f} {unit}' if unit != 'B' else f'{int(value)} B'
        value /= 1024
    return f'{value:.1f} TiB'


def _add_device_args(parser: argparse.ArgumentParser) -> None:
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
        choices=_DTYPES,
        help='Picks between several snapshots in one repo; a snapshot always runs in the dtype it was written in',
    )
    parser.add_argument('--seed', type=int, default=_DEFAULTS.seed, help='Random seed for reproducibility')


def _add_image_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group('text-to-image')
    group.add_argument('--width', type=int, default=_IMAGE_DEFAULTS.width, help='Image width in pixels, rounded down to a multiple of 32')
    group.add_argument('--height', type=int, default=_IMAGE_DEFAULTS.height, help='Image height in pixels, rounded down to a multiple of 32')
    group.add_argument('--steps', type=int, default=_IMAGE_DEFAULTS.steps, help='Denoising steps, defaults to the checkpoint setting (40)')
    group.add_argument(
        '--negative-prompt',
        type=str,
        default=_IMAGE_DEFAULTS.negative_prompt,
        help='Enables classifier-free guidance together with --guidance-scale > 1',
    )
    group.add_argument(
        '--guidance-scale',
        type=float,
        default=_IMAGE_DEFAULTS.guidance_scale,
        help='Classifier-free guidance scale, 1.0 disables it like the reference',
    )
    group.add_argument('--no-kv-cache', action='store_true', help='Recompute the text tokens at every denoising step instead of caching their K/V')
    group.add_argument('--keep-loaded', action='store_true', help='Keep all three networks in memory instead of loading each just in time')


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    _add_device_args(parser)
    parser.add_argument('--max-tokens', type=int, default=_DEFAULTS.max_tokens, help='Maximum number of new tokens to generate')
    parser.add_argument('--temp', type=float, default=_DEFAULTS.temp, help='Sampling temperature; 0 is greedy, above it samples the top-k')
    parser.add_argument('--top-k', type=int, default=_DEFAULTS.top_k, help='Top-k sampling')
    parser.add_argument('--system', type=str, default=_DEFAULTS.system, help='System prompt')
    parser.add_argument('--repo-id', type=str, default=None, help='HF repo to pull the tokenizer from, overrides the one in the snapshot')


def _defaults_from(args: argparse.Namespace) -> EngineDefaults:
    """Fold whichever engine flags the subcommand exposed over the defaults; the rest keep theirs."""
    image = ImageDefaults(
        height=getattr(args, 'height', _IMAGE_DEFAULTS.height),
        width=getattr(args, 'width', _IMAGE_DEFAULTS.width),
        steps=getattr(args, 'steps', _IMAGE_DEFAULTS.steps),
        negative_prompt=getattr(args, 'negative_prompt', _IMAGE_DEFAULTS.negative_prompt),
        guidance_scale=getattr(args, 'guidance_scale', _IMAGE_DEFAULTS.guidance_scale),
        kv_cache=not getattr(args, 'no_kv_cache', not _IMAGE_DEFAULTS.kv_cache),
        keep_loaded=getattr(args, 'keep_loaded', _IMAGE_DEFAULTS.keep_loaded),
    )
    return EngineDefaults(
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        max_tokens=getattr(args, 'max_tokens', _DEFAULTS.max_tokens),
        temp=getattr(args, 'temp', _DEFAULTS.temp),
        top_k=getattr(args, 'top_k', _DEFAULTS.top_k),
        system=getattr(args, 'system', _DEFAULTS.system),
        repo_id=getattr(args, 'repo_id', _DEFAULTS.repo_id),
        image=image,
    )


def _resolve(name: str) -> Target:
    try:
        return registry.resolve(name)
    except (KeyError, FileNotFoundError) as e:
        raise SystemExit(registry.message(e)) from e


def cmd_list(args: argparse.Namespace) -> None:
    table = Table(title='Models', title_style='bold', header_style='bold cyan', box=None, pad_edge=False)
    table.add_column('NAME')
    table.add_column('TYPE')
    table.add_column('STATUS')
    table.add_column('SIZE', justify='right')
    table.add_column('SNAPSHOT')
    rows = 0
    for target, local in registry.catalog(args.dtype):
        if local is None and not args.available:
            continue
        status = '[green]installed[/]' if local is not None else '[dim]available[/]'
        table.add_row(target.name, target.kind or '?', status, _human(local.size) if local else '-', local.filename if local else target.repo_id)
        rows += 1
    for stray in registry.strays():
        table.add_row(stray.repo_id, registry.snapshot_kind(str(stray.path)), '[green]installed[/]', _human(stray.size), stray.filename)
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


def _load(target: Target, args: argparse.Namespace, kind: str) -> InferenceEngine | ImageGenEngine:
    if target.kind is not None and target.kind != kind:
        other = 'run' if target.kind == CHAT else 'image'
        raise SystemExit(f'{target.name} is a {"chat" if target.kind == CHAT else "text-to-image"} model, use `magnetron-serve {other} {args.model}`')
    if registry.installed(target, args.dtype) is None:
        console.print(f'[dim]{target.name} is not installed, pulling it now.[/dim]')
    model = load_model(target, _defaults_from(args))
    if model.kind != kind:  # A bare Hub repo only shows its kind once its snapshot is on disk.
        other = 'run' if model.kind == CHAT else 'image'
        raise SystemExit(
            f'{target.name} turned out to be a {"chat" if model.kind == CHAT else "text-to-image"} model, use `magnetron-serve {other} {args.model}`'
        )
    return model.engine


def cmd_run(args: argparse.Namespace) -> None:
    target = _resolve(args.model)
    engine = _load(target, args, CHAT)
    assert isinstance(engine, InferenceEngine)
    if args.prompt:
        reply = engine.gen_one_shot(engine.model.build_prompt(args.system, [('user', args.prompt)]), reset_cache=True)
        console.print(f'\n\nAnswer: {reply}', style='bold green')
    else:
        repl(engine, target.name, args.system)


def cmd_image(args: argparse.Namespace) -> None:
    target = _resolve(args.model)
    engine = _load(target, args, IMAGE)
    assert isinstance(engine, ImageGenEngine)
    if args.prompt is None:
        image_repl(engine, target.name, args.out)
        return
    console.print(f'[bold]Prompt:[/bold] {args.prompt}')
    start = time.perf_counter()
    image = engine.generate(args.prompt)
    engine.save(image, args.out)
    _, height, width = image.shape
    console.print(f'Saved {width}x{height} image to [bold]{args.out}[/bold] in {time.perf_counter() - start:.1f}s', style='green')


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

    run = sub.add_parser('run', help='Run a chat model in an interactive REPL')
    run.add_argument('model', type=str, help='Registry name, Hub repo id, or path to a .mag snapshot')
    run.add_argument('--prompt', type=str, default=None, help='Answer this one prompt and exit instead of opening the REPL')
    _add_engine_args(run)
    run.set_defaults(func=cmd_run)

    image = sub.add_parser('image', help='Paint an image with a text-to-image model')
    image.add_argument('model', type=str, help='Registry name (qwen-image-2.1), Hub repo id, or path to a pipeline .mag snapshot')
    image.add_argument('prompt', type=str, nargs='?', default=None, help='What to paint; leave it out for a REPL that paints one image per line')
    image.add_argument('-o', '--out', type=str, default='image.png', help='Output file, .png keeps the alpha channel, .jpg composites over white')
    _add_device_args(image)
    _add_image_args(image)
    image.set_defaults(func=cmd_image)

    serve_cmd = sub.add_parser('serve', help='Start the SSE inference server')
    serve_cmd.add_argument('model', type=str, nargs='?', default=None, help='Model to serve by default when a request names none')
    serve_cmd.add_argument('--host', type=str, default='127.0.0.1', help='Address to bind')
    serve_cmd.add_argument('--port', type=int, default=11434, help='Port to bind')
    serve_cmd.add_argument('--max-loaded', type=int, default=1, help='How many models may sit in memory at once')
    serve_cmd.add_argument('--queue-limit', type=int, default=32, help='Requests allowed to wait before new ones get a 503')
    serve_cmd.add_argument('--lazy', action='store_true', help='Load the default model on first request instead of at startup')
    _add_engine_args(serve_cmd)
    _add_image_args(serve_cmd)
    serve_cmd.set_defaults(func=cmd_serve)

    listing = sub.add_parser('list', help='List installed models')
    listing.add_argument('--available', action='store_true', help='Also list models that are available but not installed')
    listing.add_argument('--dtype', type=str, default=_DEFAULTS.dtype, choices=_DTYPES)
    listing.set_defaults(func=cmd_list)

    install = sub.add_parser('install', help='Download a model snapshot from the Hub')
    install.add_argument('model', type=str, nargs='+', help='Registry name or Hub repo id')
    install.add_argument('--dtype', type=str, default=_DEFAULTS.dtype, choices=_DTYPES)
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
