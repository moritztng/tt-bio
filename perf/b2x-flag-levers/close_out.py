#!/usr/bin/env python3
"""The two things the timed A/B could not answer, on the one device open that is left.

1. `TT_BIO_ATOM_AXIS_BUCKET` is bit-exact at 298 aa and not at 512 aa, so the monomeric control
   scores 0.000 A and does NOT exercise the 512 aa reassociation. The prescribed diagnostic for
   exactly that case (memory `cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`) is to
   keep the 512 aa CIFs and superpose the two pseudo-domains SEPARATELY: the fixture-artifact
   signature is each domain under 0.5 A with the whole difference in the hinge rotation. So fold
   all four arms at the published protocol and keep the CIFs.

2. The flag changes the tensor extent and nothing else -- same op graph, same dtypes, same program
   count -- so it is the fold's cleanest bytes-versus-transactions separator, and that only reads
   if the byte side is measured. Capture the atom layer both ways through
   `perf/b2x_difflayer/real_traffic.py`, the corrected counter, rather than adding a fourth one.

Both in one process: the p300c wedges inside `ttnn.open_device` on its fourth open since that
chip's last reset, and this is the third.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "b2x_difflayer"))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    ap.add_argument("--cap-steps", type=int, default=6)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--skip-capture", action="store_true")
    a = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as TT
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *x, **k: None)
    import ab_flag_levers as AB
    import capture_difftx as CD
    import real_traffic as RT

    work = Path(tempfile.mkdtemp(prefix="b2x-flaglev-close-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    fix = REPO / "perf" / "size512" / "fixtures"
    target = fix / f"cdk2x2_{a.size}.yaml"
    AB._seed_msa(target, (fix / f"cdk2x2_{a.size}.a3m").read_text(), msa_dir)

    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = 200, 3
    cfg_full = AB.build_cfg(msa_dir, struct_dir)
    AB.SAMPLING_STEPS, AB.RECYCLING_STEPS = a.cap_steps, 1
    cfg_short = AB.build_cfg(msa_dir, struct_dir)
    _ensure_local_artifacts(cfg_full)

    dev = get_device()
    CD.DEV["d"] = dev
    state = _WorkerState("tenstorrent")
    state.load_model(cfg_full)
    state.bind_run("b2x-flag-levers-close", cfg_full)
    state.model.progress_fn = lambda *x, **k: None
    state.pfn = state.model.progress_fn
    score = state.model.structure_module.score_model

    out = {"size": a.size, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "folds": [], "capture": {}}

    def fold(arm, cfg, keep=None):
        TT._ATOM_AXIS_BUCKET, TT._B2_TOKEN_DIT_SDPA = AB.ARMS[arm]
        try:
            score.reset_static_cache()
        except Exception:                                                     # noqa: BLE001
            pass
        for p in struct_dir.glob("*"):
            if p.is_file():
                p.unlink()
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        if keep:
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cifs[0], keep / cifs[0].name)
        return {"arm": arm, "fold_s": round(wall, 3),
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest()}

    print("[keep] four arms at the published protocol, CIFs kept for the domain split", flush=True)
    for arm in ("base", "A", "B", "AB"):
        r = fold(arm, cfg_full, keep=a.cifdir / f"{a.size}_{arm}_0")
        out["folds"].append(r)
        print(f"  {arm:4s} {r['fold_s']:7.3f}s plddt {r['plddt']} cif {r['cif_sha256'][:16]}",
              flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1))

    if not a.skip_capture:
        print("\n[bytes] atom layer, both arms, real_traffic.py", flush=True)
        CD.wrap(TT.DiffusionTransformerLayer, "difftx", 3)
        for arm in ("base", "B"):
            CD.SEEN.clear()
            fold(arm, cfg_short)
            print(f"  arm {arm}: sigs {sorted(CD.DUMP)}", flush=True)
        calls = {}
        for s, call in CD.DUMP.items():
            c = RT.counts(call)
            c["per_op_top"] = c.pop("per_op")[:12]
            c["inputs"] = call["inputs"]
            calls[s] = c
            print(f"  {s:26s} real {c['real_MB']:8.3f} MB  once {c['once_MB']:8.3f}  "
                  f"floor {c['floor_MB']:8.3f}  ops {c['n_ops']}", flush=True)
        out["capture"]["calls"] = calls
        atom = {s: v for s, v in calls.items() if "x32x128" in s}
        if len(atom) == 2:
            hi = max(atom, key=lambda s: int(s.split("|")[1].split("x")[1]))
            lo = min(atom, key=lambda s: int(s.split("|")[1].split("x")[1]))
            nh, nl = (int(x.split("|")[1].split("x")[1]) for x in (hi, lo))
            out["capture"]["atom_delta"] = {
                "off_sig": hi, "on_sig": lo, "windows_off": nh, "windows_on": nl,
                "window_ratio": round(nl / nh, 5),
                "real_MB_off": round(atom[hi]["real_MB"], 3),
                "real_MB_on": round(atom[lo]["real_MB"], 3),
                "byte_ratio": round(atom[lo]["real_MB"] / atom[hi]["real_MB"], 5),
                "deleted_MB_per_call": round(atom[hi]["real_MB"] - atom[lo]["real_MB"], 3),
                "n_ops_off": atom[hi]["n_ops"], "n_ops_on": atom[lo]["n_ops"],
                "modelled_byte_ratio": 0.6258,
            }
            print("\natom delta:", json.dumps(out["capture"]["atom_delta"], indent=1), flush=True)
        a.out.write_text(json.dumps(out, indent=1))

    out["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    a.out.write_text(json.dumps(out, indent=1))
    print("\nwrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
