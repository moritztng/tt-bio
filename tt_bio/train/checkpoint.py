"""Checkpoints: the adapter round-trip, and the ``Checkpointer`` that owns the cadence.

Tier 2. ``save_adapter``/``load_adapter`` are the raw round-trip; ``Checkpointer`` is the
thing a loop holds, because "every N steps, keep the best K, write provenance beside it" is
bookkeeping every training loop reinvents and gets subtly wrong.

The MASTER is what gets saved, never the bf16 weight the device holds. The master is the
authoritative value and the device copy is a rounded view of it, so saving the view loses
exactly the low bits the master exists to keep and a save/reload cycle quietly decays the
adapter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from .optim import AdamW
from .tensors import to_device

if TYPE_CHECKING:
    from .. import autograd as ag

__all__ = ["Checkpointer", "save_adapter", "load_adapter"]


_SPLIT = "|"


def save_adapter(path, opt: AdamW, *, meta: Optional[dict] = None) -> None:
    """Write the fp32 masters plus the Adam moments to one safetensors file.

    The MASTER is what gets saved, not the bf16 weight the device holds: the master is the
    authoritative value and the device copy is a rounded view of it. Saving the rounded
    view would lose exactly the low bits the master exists to keep, so a save/reload cycle
    would quietly decay the adapter.
    """
    from safetensors.numpy import save_file
    tensors = {}
    for name, arr in opt.master.items():
        tensors[f"master{_SPLIT}{name}"] = arr
        tensors[f"exp_avg{_SPLIT}{name}"] = opt.exp_avg[name]
        tensors[f"exp_avg_sq{_SPLIT}{name}"] = opt.exp_avg_sq[name]
    header = {"optimizer": json.dumps(opt.state_dict()),
              "tt_bio_adapter": "1"}
    if meta:
        header["meta"] = json.dumps(meta)
    save_file(tensors, str(path), metadata=header)


def load_adapter(path, params: Dict[str, ag.Tensor], device, *, opt: Optional[AdamW] = None):
    """Load an adapter into ``params`` (and optionally an optimizer's moments).

    Returns the saved metadata dict. Every name in ``params`` must be present in the file
    and no extras are tolerated: a silently partial load is how a reload reproduces the
    BASE model's loss and gets read as "the fine-tune did nothing".
    """
    from safetensors.numpy import load_file
    import numpy as np
    blob = load_file(str(path))
    have = {k.split(_SPLIT, 1)[1] for k in blob if k.startswith(f"master{_SPLIT}")}
    want = set(params)
    if have != want:
        raise ValueError(f"adapter/parameter mismatch: missing {sorted(want - have)}, "
                         f"unexpected {sorted(have - want)}")
    for name, t in params.items():
        arr = blob[f"master{_SPLIT}{name}"].astype(np.float32)
        t.value = to_device(arr, device, dtype=t.value.dtype)
        if opt is not None:
            opt.master[name] = arr.copy()
            opt.exp_avg[name] = blob[f"exp_avg{_SPLIT}{name}"].astype(np.float32).copy()
            opt.exp_avg_sq[name] = blob[f"exp_avg_sq{_SPLIT}{name}"].astype(np.float32).copy()
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        header = json.loads(fh.read(n))
    md = header.get("__metadata__", {})
    if opt is not None and "optimizer" in md:
        opt.load_state_dict(json.loads(md["optimizer"]))
    return json.loads(md["meta"]) if "meta" in md else {}


# --------------------------------------------------------------------- the cadence

@dataclass
class Checkpointer:
    """Write an adapter every ``every`` steps, keep the best ``keep`` by a metric.

    ``lower_is_better`` is required to be stated rather than guessed from the metric's name.
    A checkpointer that infers direction from whether the key contains "loss" keeps the worst
    K checkpoints the first time someone passes an accuracy, and the run that discovers it is
    the one that needed the checkpoint.

    Provenance is written beside every checkpoint, not optionally. ``save()`` takes the record
    :func:`tt_bio.train.provenance.record` produces and stores it in the safetensors header,
    so a checkpoint found on disk a month later still names the clock, the seed and the sha it
    came from.
    """

    dir: Path
    every: int = 100
    keep: int = 3
    metric: str = "loss"
    lower_is_better: bool = True
    prefix: str = "adapter"
    written: list = field(default_factory=list)

    def __post_init__(self):
        self.dir = Path(self.dir)
        if self.every < 1:
            raise ValueError(f"every must be at least 1, got {self.every}")
        if self.keep < 1:
            raise ValueError(f"keep must be at least 1, got {self.keep}")

    def due(self, step: int) -> bool:
        return step > 0 and step % self.every == 0

    def save(self, step: int, opt: AdamW, *, metrics: Optional[dict] = None,
             provenance: Optional[dict] = None, force: bool = False) -> Optional[Path]:
        """Write if due. Returns the path written, or ``None`` when it was not due.

        The metric is read out of ``metrics`` and stored in the header rather than encoded in
        the filename. A filename is the wrong place for a float: it gets truncated, it gets
        sorted as a string, and it cannot carry which metric it was.
        """
        if not (force or self.due(step)):
            return None
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{self.prefix}-{step:08d}.safetensors"
        score = None if metrics is None else metrics.get(self.metric)
        meta = {"step": step, "metric": self.metric, "score": score,
                "metrics": metrics or {}, "provenance": provenance or {}}
        save_adapter(path, opt, meta=meta)
        self.written.append({"path": path, "step": step, "score": score})
        self._prune()
        return path

    def best(self) -> Optional[dict]:
        scored = [w for w in self.written if w["score"] is not None]
        if not scored:
            return None
        return min(scored, key=lambda w: w["score"] if self.lower_is_better else -w["score"])

    def _prune(self) -> None:
        """Delete everything outside the best ``keep``, and never the newest.

        The newest is retained whatever it scores, because a run that is interrupted resumes
        from the last checkpoint and not from the best one. Keeping only the best K loses the
        ability to resume from where the run actually is.
        """
        if len(self.written) <= self.keep:
            return
        newest = self.written[-1]
        scored = [w for w in self.written if w["score"] is not None and w is not newest]
        unscored = [w for w in self.written if w["score"] is None and w is not newest]
        scored.sort(key=lambda w: w["score"] if self.lower_is_better else -w["score"])
        survivors = [newest] + scored[:max(self.keep - 1, 0)]
        # An unscored checkpoint cannot be compared, so it is kept only if there is room after
        # the scored ones. Dropping it silently would make `keep` mean something different on
        # a run that passes no metrics.
        room = self.keep - len(survivors)
        survivors += unscored[-room:] if room > 0 else []
        keepset = {id(w) for w in survivors}
        for w in list(self.written):
            if id(w) not in keepset:
                Path(w["path"]).unlink(missing_ok=True)
                self.written.remove(w)
