#!/usr/bin/env python3
"""Which `_MM_BLOCK` keys a Boltz-2 512 aa fold actually asks for.

WHY THIS EXISTS. `main_drift_screen.py` refuses a session whose tree is behind `origin/main`
unless the delta is table entries. Main moved 46 commits under session 3 and the `tt_bio/` half
of that delta is five new `_MM_BLOCK` rows -- (8,32), (8,33), (2,6), (2,8), (2,9) -- added for
protenix-v2, plus a pxdesign Wormhole CEILINGS row. `_MM_BLOCK` is a lookup table, so a new row
changes behaviour only for a call whose key IS that row. Whether a Boltz-2 512 aa fold ever asks
for one of them is an EXECUTED fact, not something to read off a c_z constant
(`eligibility-firing-condition-is-not-a-code-fact`), which is why this takes a chip.

It is a COUNTER, not a wall-clock read. The coupled resource on a p300c board pair is the board
power budget, which moves the AICLK and therefore the seconds; it does not move which key a
matmul asks for. So this is admissible on a contended chip and on a loud host, and it claims no
seconds.

`_mm_block_for` calls itself "the single reader of the (kt, nt) key" (tenstorrent.py:7295), so
wrapping it sees every request through that path. The two other readers of `_MM_BLOCK` are
`swiglu_fused.py` and `trimul_tail.py`, and both gate on a literal key set -- `{(4,16)}` and
`{(8,8)}` -- which this script asserts still excludes every watched key, so they need no fold.

Usage (contended chip is fine):
  TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:c14-land-tail \
    python3 perf/c14_land/mmkey_census.py --out perf/c14_land/mmkey_512.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
FIX = REPO / "perf" / "size512" / "fixtures"

# The five rows origin/main added under session 3. Read off `git diff` and asserted against the
# live table below, so a typo here fails loudly instead of clearing the screen by accident.
WATCHED = [(8, 32), (8, 33), (2, 6), (2, 8), (2, 9)]

OUT: dict = {}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1, default=str))


def _load_helpers():
    p = REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py"
    spec = importlib.util.spec_from_file_location("_b2x_flaglev", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", default="512")
    ap.add_argument("--steps", default="6,14",
                    help="two step counts. The key SET is a function of the weight shapes the "
                         "fold reaches, so it must not move with sampling steps; the second "
                         "count is the control that a short fold did not simply miss a site.")
    ap.add_argument("--recycles", type=int, default=3)
    args = ap.parse_args()
    OUT_PATH = args.out

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    from tt_bio import swiglu_fused, trimul_tail
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree "
        "(memory parity-gate-scores-installed-package-not-checkout)")

    # The watched keys must BE in the live table, or the census would report "never asked for"
    # about rows that are not there -- a pass for the wrong reason.
    missing = [k for k in WATCHED if k not in TT._MM_BLOCK]
    assert not missing, f"watched keys absent from _MM_BLOCK on this tree: {missing}"

    # The two gated readers, checked from their literal key sets rather than by folding.
    gated = {
        "SWIGLU_BLOCK_KEYS": sorted(map(list, swiglu_fused.SWIGLU_BLOCK_KEYS)),
        "F1_BLOCK_KEYS": sorted(map(list, trimul_tail.F1_BLOCK_KEYS)),
    }
    gated_overlap = sorted(
        {tuple(k) for k in gated["SWIGLU_BLOCK_KEYS"] + gated["F1_BLOCK_KEYS"]}
        & set(WATCHED))

    REQ: dict = {}
    _orig = TT._mm_block_for

    def _counted(w):
        key = ((int(w.shape[-2]) + 31) // 32, (int(w.shape[-1]) + 31) // 32)
        hit = _orig(w)
        r = REQ.setdefault(str(list(key)), [0, 0])
        r[0 if hit is not None else 1] += 1
        return hit

    TT._mm_block_for = _counted

    dev = get_device()
    try:
        g = dev.compute_with_storage_grid_size()
        grid = [g.x, g.y]
    except Exception:
        grid = None

    OUT["env"] = {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": grid,
        "tt_bio_file": _TB.__file__,
        "git_head": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "size_aa": args.size, "recycles": args.recycles,
        "note": "KEY census. No wall-clock claim, so a contended board-pair sibling and a loud "
                "host do not invalidate it.",
    }
    OUT["watched_keys"] = [list(k) for k in WATCHED]
    OUT["gated_readers"] = gated
    OUT["gated_readers_overlap_watched"] = [list(k) for k in gated_overlap]
    dump()

    H = _load_helpers()
    work = Path(tempfile.mkdtemp(prefix="c14-mmkey-", dir=str(REPO / "perf" / "c14_land")))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    H._seed_msa(FIX / f"cdk2x2_{args.size}.yaml",
                (FIX / f"cdk2x2_{args.size}.a3m").read_text(), msa_dir)

    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    cfg = H.build_cfg(msa_dir, struct_dir)
    cfg["recycling_steps"] = args.recycles
    cfg["sampling_steps"] = steps[0]
    cfg["conf_kwargs"]["predict_args"]["recycling_steps"] = args.recycles
    cfg["conf_kwargs"]["predict_args"]["sampling_steps"] = steps[0]
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("c14-land-tail-mmkey", cfg)

    runs = []
    for n in steps:
        cfg["sampling_steps"] = n
        cfg["conf_kwargs"]["predict_args"]["sampling_steps"] = n
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        REQ.clear()
        state.pfn = None
        ttnn.synchronize_device(dev)
        err = None
        try:
            state.predict_one(FIX / f"cdk2x2_{args.size}.yaml", cfg)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        ttnn.synchronize_device(dev)
        runs.append({
            "sampling_steps": n, "error": err,
            "keys_requested": dict(sorted(REQ.items())),
            "watched_requested": {str(list(k)): REQ.get(str(list(k))) for k in WATCHED
                                  if str(list(k)) in REQ},
        })
        OUT["runs"] = runs
        dump()

    key_sets = [set(r["keys_requested"]) for r in runs]
    OUT["key_set_stable_across_steps"] = all(s == key_sets[0] for s in key_sets)
    OUT["key_set"] = sorted(key_sets[0]) if key_sets else []
    watched_hit = sorted({k for r in runs for k in r["watched_requested"]})
    OUT["watched_requested_any"] = watched_hit

    # NEGATIVE CONTROL. The claim is "the checker would have seen a request". Point the same
    # checker at a key the fold DOES ask for and require it to fire; without this, an instrument
    # that recorded nothing at all would also report the watched keys as never asked for.
    control_key = None
    for k, v in sorted(runs[0]["keys_requested"].items()):
        if v[0] > 0:                      # a key that HIT the table, i.e. a real swept entry
            control_key = k
            break
    OUT["negative_control"] = {
        "key": control_key,
        "counts": runs[0]["keys_requested"].get(control_key),
        "checker_fires_on_it": control_key is not None
        and control_key in runs[0]["keys_requested"],
        "why": "a served key from this same fold, run through the same lookup the watched keys "
               "are checked with. If this does not fire the instrument is blind and the watched "
               "result means nothing.",
    }

    OUT["verdict"] = (
        "CLEAR: none of origin/main's five added _MM_BLOCK keys is requested by this fold"
        if not watched_hit and OUT["key_set_stable_across_steps"]
        and not gated_overlap and OUT["negative_control"]["checker_fires_on_it"]
        else "NOT CLEAR")
    dump()
    print(json.dumps({k: OUT[k] for k in (
        "key_set", "key_set_stable_across_steps", "watched_requested_any",
        "gated_readers_overlap_watched", "negative_control", "verdict")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
