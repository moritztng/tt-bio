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
for it once. Five trunks is 5 x 91,177,010 parameters and all five fit on one p150a, measured:
8.9, 5.4, 4.6, 6.1, 6.2 s to load, 31.2 s in total, and a trajectory then runs against them.

`resident` caps how many are held and evicts least-recently-used, and it earns its place: with
all five resident a levers-on trajectory at n=288 died at gradient step 8 on
`bank_manager.cpp:439`, the allocator refusing an 84,934,656 B DRAM buffer in the backward. The
identical arm at `resident=1` ran 12 steps with no such refusal, four past where the other died,
at the same rate -- 66-89 s per step against 63-89 s -- because the reload it costs is smaller
than a step. Five trunks is about 910 MB of weights and that is enough to bring a known defect
forward.

So `--pool-resident 1` is the setting for a long pool run until `bcx-dram`'s allocator fix lands.
What still is NOT measured is the eviction itself: it drops the only Python reference and relies
on ttnn freeing each weight tensor's buffer when it is collected, since tt_bio has no model-level
deallocate. The survival above is consistent with the buffers being freed and is not a direct
reading of the allocator across an eviction.
"""
from __future__ import annotations

import json
import os
import pathlib
import time

POOL = tuple(f"model_{i}_multimer_v3" for i in range(1, 6))


class MultimerPool:
    """Several `afgrad.Dev`s behind one `Dev`-shaped surface, selected by model name."""

    def __init__(self, params_dir: str, models=POOL, k_evo: int = 48,
                 resident: int | None = None, verbose: bool = True,
                 log_path: str | None = None):
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
        #: One line per selection, appended as it happens. `stamp()` only lands when the process
        #: exits cleanly, and a design campaign is exactly the kind of run that does not: the
        #: seq-288 arms all died mid-trajectory and took their selection counts with them. A
        #: timestamped line per step is also the only way to get a per-model step RATE out of a
        #: run, which is the number the campaign wants and the counts alone cannot give.
        self.log_path = log_path
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
        if self.log_path:
            try:
                with open(self.log_path, "a") as fh:
                    fh.write(json.dumps({"t": round(time.time(), 3), "model": name,
                                         "n": self.selections[name]}) + "\n")
            except OSError:
                # A full or read-only disk must not end a design campaign for the sake of a log.
                self.log_path = None

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

    def __getattr__(self, name):
        """Everything else is the selected trunk's.

        This started as a hand-written forwarder for the five members `splice.py` appeared to
        use, and it got `tt` wrong -- the taped path reaches for `dev.tt` and the campaign died
        on it 24 minutes in, after the pool had already done its job and switched to model_3.
        Enumerating another object's surface by reading its callers is a list that is wrong as
        soon as a caller changes, so the pool delegates instead. `__getattr__` runs only for
        names this class does not define, so `use`, `models`, `dev` and `stamp` still win.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.dev, name)

    def stamp(self) -> dict:
        return {"models": list(self.models), "resident": self.resident,
                "log_path": self.log_path,
                "selections": dict(self.selections), "load_seconds": dict(self.load_seconds),
                "on_card": list(self._order)}
