# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from magnetron_models.inference import DTYPES
from magnetron_models.models import MODELS_MAP, ModelSpec
from magnetron_models.utils import download_or_ensure_resource, find_snapshot_file


@dataclass(frozen=True, slots=True)
class CachedSnapshot:
    repo_id: str
    filename: str
    path: Path
    size: int


@dataclass(frozen=True, slots=True)
class Target:
    name: str
    spec: ModelSpec | None = None
    repo_id: str | None = None
    snapshot: str | None = None

    @property
    def is_local_file(self) -> bool:
        return self.spec is None and self.repo_id is None


def message(e: Exception) -> str:
    return str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)


def dtype_suffix(dtype: str) -> str:
    return DTYPES[dtype].short_name


def resolve(name: str) -> Target:
    if name.endswith('.mag'):
        path = Path(name).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'No such snapshot: {path}')
        return Target(name=path.stem, snapshot=str(path))
    if name in MODELS_MAP:
        spec = MODELS_MAP[name]
        return Target(name=name, spec=spec, repo_id=spec.snapshot_repo_id)
    if '/' in name:
        return Target(name=name, repo_id=name)
    raise KeyError(f'Unknown model {name!r}. Known names: {", ".join(sorted(MODELS_MAP))}. Or pass a Hub repo id or a .mag path.')


def cached_snapshots(repo_id: str | None = None) -> list[CachedSnapshot]:
    from huggingface_hub import scan_cache_dir

    out: list[CachedSnapshot] = []
    try:
        cache = scan_cache_dir()
    except Exception:
        return out
    for repo in cache.repos:
        if repo.repo_type != 'model' or (repo_id is not None and repo.repo_id != repo_id):
            continue
        for revision in repo.revisions:
            for file in revision.files:
                if file.file_name.endswith('.mag'):
                    out.append(CachedSnapshot(repo.repo_id, file.file_name, Path(file.file_path), file.size_on_disk))
    out.sort(key=lambda s: (s.repo_id, s.filename))
    return out


def _pick(candidates: list[CachedSnapshot], dtype: str) -> CachedSnapshot | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    matching = [c for c in candidates if c.filename.endswith(f'-{dtype_suffix(dtype)}.mag')]
    return matching[0] if len(matching) == 1 else None


def installed(target: Target, dtype: str = 'bfloat16') -> CachedSnapshot | None:
    if target.snapshot is not None:
        path = Path(target.snapshot)
        return CachedSnapshot(target.repo_id or '', path.name, path, path.stat().st_size)
    if target.repo_id is None:
        return None
    candidates = cached_snapshots(target.repo_id)
    if target.spec is not None and target.spec.snapshot_file is not None:
        candidates = [c for c in candidates if c.filename == target.spec.snapshot_file]
    return _pick(candidates, dtype)


def install(target: Target, dtype: str = 'bfloat16') -> str:
    if target.snapshot is not None:
        return target.snapshot
    if target.spec is not None:
        return target.spec.download_snapshot(dtype_suffix(dtype))
    assert target.repo_id is not None
    local = installed(target, dtype)
    filename: str = local.filename if local is not None else find_snapshot_file(target.repo_id, dtype_suffix(dtype))
    return download_or_ensure_resource(repo_id=target.repo_id, filename=filename)


def uninstall(target: Target) -> int:
    from huggingface_hub import scan_cache_dir

    if target.repo_id is None:
        raise ValueError(f'{target.name} is a local file, remove it yourself')
    cache = scan_cache_dir()
    hashes = [rev.commit_hash for repo in cache.repos if repo.repo_id == target.repo_id for rev in repo.revisions]
    if not hashes:
        raise KeyError(f'{target.repo_id} is not installed')
    strategy = cache.delete_revisions(*hashes)
    freed: int = strategy.expected_freed_size
    strategy.execute()
    return freed


def catalog(dtype: str = 'bfloat16') -> list[tuple[str, ModelSpec, CachedSnapshot | None]]:
    return [(name, spec, installed(resolve(name), dtype)) for name, spec in sorted(MODELS_MAP.items())]


def strays() -> list[CachedSnapshot]:
    known = {spec.snapshot_repo_id for spec in MODELS_MAP.values()}
    return [s for s in cached_snapshots() if s.repo_id not in known]
