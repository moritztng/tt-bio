#!/usr/bin/env python3
"""The per-arch readback dispatch, scored inside a real fold rather than on a synthetic tensor.

`download_shape.py` timed the two candidate shapes on a tensor of the head's shape and found the
ranking inverts between the parts. This runs the same comparison where it ships: the two arms are
`ConfidenceHeadsDevice.__call__`'s own branch, selected by monkeypatching the module's
`is_wormhole` so one process folds both, interleaved, against one device open and one warm cache.

  narrow   to_layout(ROW_MAJOR) + slice on the card, then download 4 channels.   (Wormhole ships)
  tiled    download the tile whole, slice the 4 channels on the host.            (Blackhole ships)

Three readings per fold. The head's own stage wall gives the download stage in ms, which is what
the dispatch is supposed to move. The head's return dict gives pae/pde/tm, and the written CIF plus
its pLDDT give the fold the user actually receives; neither must move at all. The equality is the
load-bearing one -- a readback shape that changed a number would be a bug, not a lever -- so both
are asserted on every fold and not just the first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics as st
import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(HERE))
FIX = REPO / "perf" / "size512" / "fixtures"

WALL = re.compile(r"^\[confheads\] (\S+(?: \S+)*?)\s+([0-9.]+) ms$")
ARMS = ("narrow", "tiled")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=4)
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import tt_baseline as B
    import tt_bio as _TB
    import tt_bio.boltz2 as boltz2
    import tt_bio.tenstorrent as T
    from tt_bio.main import _resolve_recycling_steps
    from score_scalars import capture, diff

    assert Path(_TB.__file__).resolve().is_relative_to(REPO), \
        f"imported tt_bio from {_TB.__file__}, not this worktree"
    for f in ("TT_BIO_DEVICE_CONF_HEADS", "TT_BIO_DEVICE_CONFIDENCE"):
        assert f not in os.environ, f"{f} is set in the environment; this script owns it"
        os.environ[f] = "1"
    os.environ["TT_BIO_DEVICE_CONFIDENCE_PROFILE"] = "1"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    B.SAMPLING_STEPS = 200
    snap = list(sys.path)
    sys.path.insert(0, str(REPO / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()

    grabbed: list = []
    orig = boltz2.ConfidenceHeads.forward

    def wrapped(self, *args, **kw):
        out = orig(self, *args, **kw)
        grabbed.append(capture(out))
        return out

    boltz2.ConfidenceHeads.forward = wrapped

    one_fold, meta, _state = B.build_fold("boltz2", HERE / f".msa_{a.size}",
                                          FIX / f"cdk2x2_{a.size}.yaml",
                                          FIX / f"cdk2x2_{a.size}.a3m")
    struct = Path(meta["struct_dir"])

    def fold(arm: str):
        """One fold on `arm`: its confheads stage wall, the head's output, and the written CIF."""
        T.is_wormhole = lambda: arm == "narrow"
        buf = StringIO()
        t0 = time.perf_counter()
        with redirect_stdout(buf):
            _t, metrics = one_fold()
        wall_s = time.perf_counter() - t0
        h = hashlib.sha256()
        for f in sorted(struct.rglob("*")):
            if f.is_file():
                h.update(f.name.encode())
                h.update(f.read_bytes())
        cif = {"sha256_16": h.hexdigest()[:16], "plddt": round(float(metrics["plddt"]), 6)}
        stages: dict = {}
        for line in buf.getvalue().splitlines():
            m = WALL.match(line.strip())
            if m:
                stages.setdefault(m.group(1), []).append(float(m.group(2)))
        return wall_s, stages, grabbed[-1], cif

    print(f"=== cold folds (discarded), one per arm === struct_dir {struct}", flush=True)
    for arm in ARMS:
        fold(arm)
    grabbed.clear()

    out = {"doc": __doc__,
           "env": {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "host": os.uname().nodename, "size": a.size, "reps": a.reps,
                   "commit": os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
                   "tt_bio_file": _TB.__file__, "arch": T.arch_name(),
                   "loadavg_start": os.getloadavg(),
                   "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS},
           "folds": []}
    dl: dict = {k: [] for k in ARMS}
    ref = ref_cif = None
    for rep in range(a.reps):
        # reverse the order inside the rep so a monotonic drift in the box cancels
        order = ARMS if rep % 2 == 0 else ARMS[::-1]
        for arm in order:
            wall_s, stages, rec, cif = fold(arm)
            if ref is None:
                ref, ref_cif = rec, cif
            assert cif == ref_cif, f"{arm} rep {rep} moved the written fold: {cif} vs {ref_cif}"
            d = diff(ref, rec)
            # _mean_ref is the reference's own mean, not a delta -- only the d_
            # scalars and the _max_abs / _mean_abs norms say whether anything moved.
            bad = {k: v for k, v in d.items() if v != 0.0 and not k.endswith("_mean_ref")}
            assert not bad, f"{arm} rep {rep} moved the head's output: {bad}"
            step = {k: round(st.median(v), 3) for k, v in stages.items()}
            dl[arm].append(step["download"])
            out["folds"].append({"rep": rep, "arm": arm, "wall_s": round(wall_s, 4),
                                 "stages_median_ms": step, "cif": cif,
                                 "loadavg": os.getloadavg()[0]})
            print(f"  rep {rep} {arm:7s} fold {wall_s:7.3f} s  download {step['download']:7.3f} ms"
                  f"  untilize {step.get('untilize', 0.0):6.3f} ms", flush=True)

    out["download_ms"] = {k: {"median": round(st.median(v), 3), "min": round(min(v), 3),
                              "max": round(max(v), 3), "all": v} for k, v in dl.items()}
    out["delta_ms"] = round(st.median(dl["narrow"]) - st.median(dl["tiled"]), 3)
    out["head_output_identical"] = True
    out["cif_identical"] = ref_cif
    print(f"  download median: narrow {out['download_ms']['narrow']['median']:.3f} ms  "
          f"tiled {out['download_ms']['tiled']['median']:.3f} ms  "
          f"delta {out['delta_ms']:.3f} ms/fold", flush=True)
    print(f"  head output and written CIF identical on every fold: asserted, {ref_cif}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
