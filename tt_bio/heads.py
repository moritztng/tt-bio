"""Custom output heads: your own Python, run on a fold's trunk outputs, written next to it.

``tt-bio predict --head FILE.py:NAME`` (or ``package.module:NAME``) loads NAME from your file
and calls it once per prediction with a :class:`Fold`. It returns a dict of arrays, which is
written to ``<record>_<NAME>.npz`` beside the structure. NAME may be a function or a class; a
class is instantiated once per worker with no arguments, so it can load its own weights in
``__init__``.

A head reads what the fold already computed and nothing reads what it returns, so the
structure and every confidence value are the ones the same fold writes without it. With no
``--head`` nothing in this module runs.

Boltz-2 and ESMFold-2 (and ESMFold-2-Fast): the models whose folds hand their trunk
representations back to the host. See docs/extending.md.
"""

from __future__ import annotations

import importlib
import importlib.util
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


@dataclass(frozen=True)
class Fold:
    """What a head is called with. Token and atom axes hold real entries only.

    s: [N, C_s] Boltz-2: the single representation from the last trunk pass, C_s 384.
       ESMFold-2 has no single track in its trunk; s is its input embedding, C_s 451.
    z: [N, N, C_z] pair representation from the last trunk pass: C_z 128 on Boltz-2, 256 on
       ESMFold-2.
    coords: [samples, atoms, 3] every diffusion sample, in Angstrom.
    plddt: [samples, N] or None. pae, pde: [samples, N, N] or None.
    pred, feats: the raw prediction dict and input batch, padded, for anything not above.
    """

    s: torch.Tensor
    z: torch.Tensor
    coords: torch.Tensor
    plddt: torch.Tensor | None
    pae: torch.Tensor | None
    pde: torch.Tensor | None
    pred: dict
    feats: dict


def resolve(spec: str) -> Path | str:
    """Make a FILE.py:NAME spec absolute, so a worker in another directory loads the same file."""
    src, sep, name = spec.rpartition(":")
    if not sep or not src or not name.isidentifier():
        raise ValueError(f"--head {spec!r}: expected FILE.py:NAME or package.module:NAME")
    if src.endswith(".py"):
        path = Path(src).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--head {spec!r}: {path} does not exist")
        return f"{path}:{name}"
    return spec


def load(spec: str) -> Callable[[Fold], dict]:
    """Import NAME from a resolved spec; instantiate it once if it is a class."""
    src, _, name = spec.rpartition(":")
    if src.endswith(".py"):
        mod_spec = importlib.util.spec_from_file_location(f"tt_bio_head_{Path(src).stem}", src)
        module = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(src)
    obj = getattr(module, name, None)
    if obj is None:
        raise AttributeError(f"--head {spec!r}: {src} defines no {name!r}")
    head = obj() if isinstance(obj, type) else obj
    if not callable(head):
        raise TypeError(f"--head {spec!r}: {name!r} is not callable")
    return head


def name(spec: str) -> str:
    return spec.rpartition(":")[2]


def fold_view(pred: dict, batch: dict) -> Fold:
    """Strip the padding a bucketed batch adds, so a head's token axis is the structure's."""
    tok = pred["token_masks"][0].bool()
    atom = pred["masks"].reshape(-1).bool()

    def pair(x):
        return None if x is None else x[:, tok][:, :, tok]

    return Fold(
        s=pred["s"][0][tok], z=pred["z"][0][tok][:, tok], coords=pred["coords"][:, atom],
        plddt=None if pred.get("plddt") is None else pred["plddt"][:, tok],
        pae=pair(pred.get("pae")), pde=pair(pred.get("pde")), pred=pred, feats=batch)


@contextmanager
def esmfold2_capture(model):
    """Record what one ESMFold-2 forward computed; read it with :func:`esmfold2_view`.

    The hooks only keep references to tensors the forward already made, and they are gone
    when the block exits.
    """
    seen: dict = {}
    hooks = [
        model.inputs_embedder.register_forward_hook(lambda m, a, out: seen.update(s=out)),
        model.parcae_coda.register_forward_hook(lambda m, a, out: seen.update(z=out)),
        model.register_forward_hook(lambda m, a, kw, out: seen.update(feats=kw, pred=out),
                                    with_kwargs=True),
    ]
    try:
        yield seen
    finally:
        for h in hooks:
            h.remove()


def esmfold2_view(seen: dict) -> Fold:
    """The :class:`Fold` of an ESMFold-2 forward, padding stripped like :func:`fold_view`."""
    pred, feats = seen["pred"], seen["feats"]
    tok = feats["token_attention_mask"][0].bool()
    atom = feats["atom_attention_mask"][0].bool()

    def pair(x):
        return None if x is None else x[:, tok][:, :, tok]

    return Fold(
        s=seen["s"][0][tok].float(), z=seen["z"][0][tok][:, tok].float(),
        coords=pred["sample_atom_coords"][:, atom], plddt=pred["plddt"][:, tok],
        pae=pair(pred.get("pae")), pde=pair(pred.get("pde")), pred=pred, feats=feats)


def run(heads: dict[str, Callable], fold: Fold, out_dir: Path, record_id: str) -> None:
    """Call each head once and write what it returns. Outputs are detached float32/ints."""
    for head_name, head in heads.items():
        with torch.no_grad():
            arrays = head(fold)
        if not isinstance(arrays, dict) or not arrays:
            raise TypeError(f"head {head_name!r} must return a non-empty dict of arrays, "
                            f"got {type(arrays).__name__}")
        np.savez_compressed(Path(out_dir) / f"{record_id}_{head_name}.npz",
                            **{k: _array(v) for k, v in arrays.items()})


def _array(v: Any) -> np.ndarray:
    if torch.is_tensor(v):
        v = v.detach().cpu()
        return (v.float() if v.is_floating_point() else v).numpy()
    return np.asarray(v)
