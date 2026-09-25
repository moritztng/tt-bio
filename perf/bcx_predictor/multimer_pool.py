#!/usr/bin/env python3
"""The five `multimer_v3` trunks BindCraft 2 designs on, as one object the splice can hold.

`examples/pdl1.json` resolves five design models and samples one per gradient step
(`bindcraft/af2.py:261`), so a device trunk pinned to one checkpoint cannot run the shipped
configuration. Every arm this campaign has run pinned one, which is why nobody has measured the
shipped configuration's accepted-binder count.

`MultimerPool` is a drop-in for `afgrad.Dev`. `splice.EvoformerOnDevice` only ever calls `up`,
`down`, `sync`, `stack` and reads `.device`, so routing those to whichever trunk is currently
selected needs no change in `splice.py` at all -- the pool quacks like one `Dev` and swaps what
is behind it. `ttbio_predictor.TTBioAlphaFoldDesignModel` resolves BindCraft 2's per-step model
name and calls `use()` before handing the call to BindCraft 2.

Weights are loaded lazily and kept, so the first gradient step that reaches a given model pays
for it once. Five trunks is 5 x 91,177,010 parameters; `--pool-resident` caps how many stay on
card if that does not fit, evicting least-recently-used.
"""
from __future__ import annotations

import os
import pathlib
import time

POOL = tuple(f"model_{i}_multimer_v3" for i in range(1, 6))


class MultimerPool:
    """Several `afgrad.Dev`s behind one `Dev`-shaped surface, selected by model name."""

    def __init__(self, params_dir: str, models=POOL, k_evo: int = 48,
                 resident: int | None = None, verbose: bool = True):
        self.dir = pathlib.Path(os.path.expanduser(params_dir))
        self.models = tuple(models)
        self.k_evo = k_evo
        self.resident = resident or len(self.models)
        self.verbose = verbose
        self._devs: dict[str, object] = {}
        self._order: list[str] = []
        self._current: str | None = None
        self.load_seconds: dict[str, float] = {}
        self.selections: dict[str, int] = {}
        missing = [m for m in self.models if not (self.dir / f"params_{m}.npz").exists()]
        if missing:
            raise FileNotFoundError(f"{self.dir} has no {', '.join(missing)}")

    # ------------------------------------------------------------------ selection

    def use(self, name: str) -> None:
        """Make `name` the trunk the splice runs. Unknown names are an error, not a default:
        silently folding a design on the wrong checkpoint is the failure this exists to stop."""
        if name not in self.models:
            raise KeyError(f"{name!r} is not in the pool {self.models}")
        self._current = name
        self.selections[name] = self.selections.get(name, 0) + 1
        self._ensure(name)

    def _ensure(self, name: str):
        dev = self._devs.get(name)
        if dev is None:
            import afgrad as _A
            t0 = time.time()
            dm, _ = _A.load_models(str(self.dir / f"params_{name}.npz"),
                                   multimer=True, refs=False)
            dev = _A.Dev(dm.to_device())
            self.load_seconds[name] = round(time.time() - t0, 1)
            if self.verbose:
                print(f"[pool] {name} on card in {self.load_seconds[name]}s", flush=True)
            self._devs[name] = dev
        if name in self._order:
            self._order.remove(name)
        self._order.append(name)
        while len(self._order) > self.resident:
            self._devs.pop(self._order.pop(0), None)
        return dev

    @property
    def dev(self):
        if self._current is None:
            raise RuntimeError("no model selected; MultimerPool.use(name) first")
        return self._ensure(self._current)

    # ------------------------------------------------------------------ the Dev surface

    @property
    def device(self):
        return self.dev.device

    def up(self, t):
        return self.dev.up(t)

    def down(self, t, shape):
        return self.dev.down(t, shape)

    def leaf(self, t):
        return self.dev.leaf(t)

    def sync(self):
        return self.dev.sync()

    def extra(self, i, z):
        return self.dev.extra(i, z)

    def evo(self, i, m, z, msa_mask=None, pair_masks=(None, None)):
        return self.dev.evo(i, m, z, msa_mask, pair_masks)

    def stack(self, *args, **kwargs):
        return self.dev.stack(*args, **kwargs)

    def stamp(self) -> dict:
        return {"models": list(self.models), "resident": self.resident,
                "selections": dict(self.selections), "load_seconds": dict(self.load_seconds),
                "on_card": list(self._order)}
