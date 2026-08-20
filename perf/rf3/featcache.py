"""featurize(), memoised to disk, because it is host work that does not belong inside benchlock.

At 1024 aa the vendored AtomWorks pipeline once sat at 100% of one core for 48 minutes and
died on its own timeout. Whatever that is, it is not device work and it must not be holding
the box lock while it happens: featurise once outside the lock, then let the timed arms load
the tensors back in a second.
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_DIR = Path("/home/ttuser/rf3_perf_work/featcache")


def featurized(inp: str, *, n_recycles: int, diffusion_batch_size: int, seed: int,
               cache_dir: str | Path | None = DEFAULT_DIR, verbose: bool = True):
    """The first element of featurize(...), cached on (input, n_recycles, batch, seed)."""
    import torch
    from tt_bio.rf3.featurize import featurize

    key = f"{Path(inp).stem}_r{n_recycles}_d{diffusion_batch_size}_s{seed}.pt"
    path = Path(cache_dir) / key if cache_dir else None
    if path is not None and path.exists():
        if verbose:
            print(f"[featurize] cache hit {path}", flush=True)
        return torch.load(path, weights_only=False)
    fo = featurize(inp, n_recycles=n_recycles,
                   diffusion_batch_size=diffusion_batch_size, seed=seed)[0]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".pt.tmp")
        torch.save(fo, tmp)
        tmp.rename(path)
        if verbose:
            print(f"[featurize] cached {path}", flush=True)
    return fo
