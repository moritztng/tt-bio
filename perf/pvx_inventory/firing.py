#!/usr/bin/env python3
"""Calls per fold, per lever, per model -- counted at run time, never read off a gate.

`scripts/lever_census.py` answers this for the INSTALLED package by spawning `tt-bio predict`
and collecting a per-process dump. That is the right shape for a release artifact and the
wrong shape for a cross-model comparison: `predict` fans into workers, each fold pays a model
load, and the cold fold's counts include every kernel compile. Here the question is narrower --
two models, one size, one process each -- so the fold runs IN PROCESS through
`tt_baseline.build_fold`, the same entry every perf harness in this tree uses, and the counters
are read directly out of the modules that own them.

The count reported is the WARM fold's. Every counter is zeroed between the cold fold and the
timed one, so a number here is calls per fold and not calls per process. The cold-fold count is
kept beside it: the two differing is how a once-per-process hoist (B2_BIAS_SLICE_HOIST,
B2_ADALN_S_MEMO) tells itself apart from a per-call lever.

AICLK is sampled DURING the warm fold in a background thread, because the governor alone swings
this fold 1.27-1.41x and a fold time without a clock is not a measurement. The counts do not
depend on the clock; the fold seconds recorded next to them do.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

import lever_census as LC                                             # noqa: E402


def _counter_objects():
    """Every mutable counter the census reads, so a reset cannot miss one it reports."""
    seen, out = set(), []
    paths = [c.partition(":")[0] for _f, _m, _a, c, _h in LC.LEVERS if c]
    paths += list(LC.REJECTS_ATTR.values())
    for attr in paths:
        if attr in seen:
            continue
        seen.add(attr)
        mod, _, name = attr.rpartition(".")
        m = sys.modules.get(mod)
        o = getattr(m, name, None) if m is not None else None
        if o is not None:
            out.append(o)
    return out


def reset_counters():
    for o in _counter_objects():
        if isinstance(o, list):
            for i in range(len(o)):
                o[i] = 0
        elif isinstance(o, set):
            pass               # `_SDPA_Q_CHUNK_OVER_L1` is a device fact, not a call count
        elif isinstance(o, dict):
            for k in list(o):
                if isinstance(o[k], int):
                    o[k] = 0
                elif isinstance(o[k], dict):
                    for kk in list(o[k]):
                        if isinstance(o[k][kk], int):
                            o[k][kk] = 0
    for v in LC.WRAP_COUNTS.values():
        v[0] = v[1] = 0
    LC.WRAP_REJECTS.clear()


def aiclk_now(card: int):
    try:
        r = subprocess.run([os.path.expanduser("~/.local/bin/tt-smi"), "-s"],
                           capture_output=True, text=True, timeout=30)
        d = json.loads(r.stdout)
        dev = d["device_info"][card]
        # `telemetry.aiclk`, not `chip_telemetry`: tt-smi 3.x renamed the block and the old key
        # raises KeyError, which the except below turns into a string. A clock that reads
        # "unreadable" is not a clock, and a fold time without one is not a measurement.
        return int(str(dev["telemetry"]["aiclk"]).strip())
    except Exception as exc:                                          # noqa: BLE001
        return "unreadable:%s" % type(exc).__name__


class ClockWatch(threading.Thread):
    """Samples AICLK while the fold runs. `tt-smi -s` is a separate process reading telemetry
    over ARC, so it does not take the device and cannot perturb the fold it watches."""

    def __init__(self, card):
        super().__init__(daemon=True)
        self.card, self.samples, self.stop = card, [], threading.Event()

    def run(self):
        while not self.stop.wait(4.0):
            self.samples.append(aiclk_now(self.card))


# LEDGER N1 step 1, which this row owns: does Protenix's Transition reach the Blackhole row
# raise (`tenstorrent.py:8626`, `_TRANSITION_L1_ROWS and _c <= _BH_TRANSITION_L1_ROWS_MAX_C`)
# with `_c == 256`? The raise's own gate is a local branch with no counter, so the census reads
# its INPUT instead -- the pair channel each Transition call presents -- and the eligibility
# follows from the constant. Counting the input is stronger than counting the branch: it says
# what the channel distribution IS, so a bound that moves later can be priced against it without
# another fold.
TRANSITION_CH = {}


def census_transition(T):
    """Wrap `Transition.__call__` to record (channel, padded width) per call. No product edit --
    `tenstorrent.py` belongs to `pvx-eligibility` and a census must not touch a shared gate."""
    cls = T.Transition
    if getattr(cls, "_pvx_wrapped", False):
        return
    inner = cls.__call__

    def wrapped(self, x, *a, **k):
        try:
            c = int(x.shape[-1])
            w = int(x.shape[-2])
            key = "c=%d,W=%d,raise_eligible=%s" % (
                c, w, c <= T._BH_TRANSITION_L1_ROWS_MAX_C)
            TRANSITION_CH[key] = TRANSITION_CH.get(key, 0) + 1
        except Exception:                                             # noqa: BLE001
            pass
        return inner(self, x, *a, **k)

    cls.__call__ = wrapped
    cls._pvx_wrapped = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", default="512")
    ap.add_argument("--card", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    fix = ROOT / "perf" / "size512" / "fixtures"
    target, a3m = fix / ("cdk2x2_%s.yaml" % a.size), fix / ("cdk2x2_%s.a3m" % a.size)
    assert target.is_file() and a3m.is_file(), "no fixture for %s aa" % a.size

    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
    # Each model folds at ITS OWN shipped protocol. Folding one model at another's recycling
    # count would change every trunk count on this page by that ratio and nothing would say so.
    B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)

    if a.model == "boltz2":
        # `build_fold`'s cfg carries no Boltz-2 hyperparameters, so `load_model` raises
        # KeyError for conf_kwargs. Reuse the injection `perf/other512/fold_ab_multi.py`
        # already maintains rather than keeping a second copy of boltz-2's config in sync.
        sys.path.insert(0, str(ROOT / "perf" / "other512"))
        from fold_ab_multi import patch_boltz2_cfg
        patch_boltz2_cfg()

    work = ROOT / "perf" / "pvx_inventory" / ("work_%s_%s" % (a.model, a.size))
    one_fold, meta, _state = B.build_fold(a.model, work / "msa", target, a3m)

    import tt_bio.tenstorrent as T
    census_transition(T)
    LC._install_wraps()
    cold_s, _cold_m = one_fold()
    cold = LC._snapshot_process()

    reset_counters()
    TRANSITION_CH.clear()
    LC._install_wraps()
    watch = ClockWatch(a.card)
    clk_before = aiclk_now(a.card)
    watch.start()
    warm_s, warm_m = one_fold()
    watch.stop.set()
    watch.join(timeout=40)
    warm = LC._snapshot_process()

    rows = []
    for flag, mod, _attr, _counter, how in LC.LEVERS:
        w, c = warm.get(flag), cold.get(flag)
        if w is None and c is None:
            rows.append({"flag": flag, "module": mod, "how": how, "state": "not-imported"})
            continue
        w, c = w or {}, c or {}
        rows.append({"flag": flag, "module": mod, "how": how,
                     "resolved": w.get("resolved"),
                     "served": w.get("served"), "declined": w.get("declined"),
                     "cold_served": c.get("served"), "cold_declined": c.get("declined"),
                     "rejects": w.get("rejects"), "gauges": w.get("gauges")})

    out = {"model": a.model, "size": int(a.size), "host": os.uname().nodename,
           "card": a.card, "grid": LC._compute_grid(),
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "n_tokens": warm_m.get("n_tokens"), "plddt": warm_m.get("plddt"),
           "cold_s": round(cold_s, 3), "warm_s": round(warm_s, 3),
           "aiclk_before": clk_before, "aiclk_during": watch.samples,
           "hardware": meta.get("hardware"), "load_s": meta.get("load_s"),
           "n_msa": meta.get("n_msa"),
           "transition_channels": dict(sorted(TRANSITION_CH.items())),
           "bh_transition_rows_max_c": T._BH_TRANSITION_L1_ROWS_MAX_C,
           "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    print("Transition channels this fold: %s (BH raise bound c <= %d)"
          % (dict(sorted(TRANSITION_CH.items())), T._BH_TRANSITION_L1_ROWS_MAX_C))
    print("%s %s aa: cold %.3f s, warm %.3f s, AICLK during %s, plDDT %s"
          % (a.model, a.size, cold_s, warm_s, watch.samples, warm_m.get("plddt")))
    for r in rows:
        if r.get("state") == "not-imported":
            continue
        print("  %-26s %-13s served=%s declined=%s"
              % (r["flag"], str(r.get("resolved"))[:12], r.get("served"), r.get("declined")))


if __name__ == "__main__":
    main()
