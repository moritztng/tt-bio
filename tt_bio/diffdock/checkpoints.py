"""Finding the two checkpoints a docking run loads.

Upstream names a checkpoint with a directory plus a filename, twice (`--model_dir --ckpt`,
`--confidence_model_dir --confidence_ckpt`). tt-bio names a checkpoint with one path, so
`--checkpoint` and `--confidence_checkpoint` each take either, and a directory resolves the
upstream layout: an untouched `workdir/v1.1/score_model` works (D-dfd-10, conflict 2).

Neither flag is needed in the normal case. The weights row `diffdock` in
``tt_bio.weights`` fetches the v1.1 release asset, sha256-verified, and extracts both
checkpoints under `<cache>/diffdock/models`; this module points at the two directories
inside it. Upstream's `releases/latest/download/diffdock_models.zip` 404s and its fetch
then silently tries a second URL, so nothing here falls back: a mismatch fails loudly.
"""

from __future__ import annotations

from pathlib import Path

from tt_bio import weights

#: Upstream's layout inside the release zip, and the filename each directory holds.
SCORE_DIR = "score_model"
CONFIDENCE_DIR = "confidence_model"
SCORE_CKPT = "best_ema_inference_epoch_model.pt"
CONFIDENCE_CKPT = "best_model_epoch75.pt"
PARAMETERS = "model_parameters.yml"


class CheckpointMissing(RuntimeError):
    """Said once, with the command that fixes it."""


def models_root(cache=None, *, download: bool = False) -> Path:
    """`<cache>/diffdock/models`, fetching the pinned v1.1 asset when asked to."""
    if download:
        return Path(weights.fetch("diffdock", root=cache))
    path = weights.cached_path("diffdock", cache)
    if path is None:
        raise CheckpointMissing(
            "DiffDock-L weights are not on this host. Run `tt-bio weights --download "
            "diffdock` (124 MiB, both checkpoints), or pass --checkpoint.")
    return Path(path)


def resolve(kind: str, given, cache=None, *, download: bool = False) -> tuple[Path, Path]:
    """Return (checkpoint file, model_parameters.yml) for ``kind`` in {score, confidence}.

    ``given`` is what the user passed: a .pt file, a directory in upstream's layout, or
    None for the cached release.
    """
    subdir, filename = {"score": (SCORE_DIR, SCORE_CKPT),
                        "confidence": (CONFIDENCE_DIR, CONFIDENCE_CKPT)}[kind]
    if given is None:
        base = models_root(cache, download=download) / subdir
    else:
        given = Path(given).expanduser()
        if given.is_file():
            base, filename = given.parent, given.name
        else:
            base = given
    ckpt, params = base / filename, base / PARAMETERS
    if not ckpt.exists():
        candidates = sorted(p.name for p in base.glob("*.pt")) if base.is_dir() else []
        hint = f" The directory holds {', '.join(candidates)}." if candidates else ""
        raise CheckpointMissing(f"no {kind} checkpoint at {ckpt}.{hint}")
    if not params.exists():
        raise CheckpointMissing(
            f"{base} has {ckpt.name} but no {PARAMETERS}; a DiffDock checkpoint directory "
            f"carries both, and the hyperparameters are not in the .pt file.")
    return ckpt, params
