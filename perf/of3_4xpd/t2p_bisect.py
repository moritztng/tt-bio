#!/usr/bin/env python3
"""Bisect the two levers that broke the fold together, in one process, one card lease.

T2 (hoist MSA block 0's outer-product mean out of the recycle loop) and P (pad the atom
dim with ttnn.pad instead of a host round trip) were built as one arm and folded to
plDDT 0.330907 / CIF 95912b35a50ab9a1 against the anchor 0.554642 / 283be8b29b15adc0.
One fold per gate combination says which one is false; the plDDT and the digest decide
it, so the fold does not need to be warm.

The last leg profiles `build_openfold3_features` (lever H) with cProfile in the same
process, which costs one more fold and no extra lease.
"""
import argparse, cProfile, hashlib, io, json, os, pstats, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

ANCHOR_PLDDT = 0.554642
ANCHOR_CIF = "283be8b29b15adc0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--profile-host", action="store_true")
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_bio.openfold3_trunk as TR
    import tt_baseline as B
    from tt_bio.main import (_resolve_recycling_steps, _resolve_sampling_steps,
                             _detect_p300_devices, _find_ttnn_mesh_graph_descriptor)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}, set PYTHONPATH"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "openfold3")
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, "openfold3")

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta = B.build_fold("openfold3", ROOT / f".msa_of3deep_{a.size}", tgt, a3m)[:2]
    struct_dir = Path(meta["struct_dir"])

    res = {"anchor": {"plddt": ANCHOR_PLDDT, "cif": ANCHOR_CIF},
           "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "legs": []}

    def leg(name, t2, p):
        TR.HOIST_MSA_OPM0 = t2
        T.DEVICE_ATOM_PAD = p
        fold_s, m = one_fold()
        cif = {q.name: hashlib.sha256(q.read_bytes()).hexdigest()[:16]
               for q in sorted(struct_dir.glob("*")) if q.is_file()}
        d = list(cif.values())
        ok = abs(float(m.get("plddt")) - ANCHOR_PLDDT) < 5e-6 and d and d[0] == ANCHOR_CIF
        rec = {"leg": name, "T2_hoist_opm0": t2, "P_device_pad": p,
               "fold_s": round(fold_s, 3), "plddt": m.get("plddt"), "cif": cif,
               "matches_anchor": bool(ok)}
        res["legs"].append(rec)
        print(f"[bisect] {name:14s} T2={t2!s:5s} P={p!s:5s}  {fold_s:7.3f}s  "
              f"plddt={m.get('plddt')}  cif={d}  anchor={'MATCH' if ok else 'BREAK'}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        return rec

    leg("cold/both-off", False, False)   # cold, discarded for timing, checks the anchor
    leg("both-off", False, False)
    leg("T2-only", True, False)
    leg("P-only", False, True)
    leg("both-on", True, True)

    if a.profile_host:
        import tt_bio.openfold3_data as OD
        prof = cProfile.Profile()
        g = OD.build_openfold3_features
        OD.build_openfold3_features = lambda *x, **k: prof.runcall(g, *x, **k)
        TR.HOIST_MSA_OPM0 = False
        T.DEVICE_ATOM_PAD = False
        one_fold()
        OD.build_openfold3_features = g
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(28)
        res["host_build_features_profile"] = s.getvalue()
        print("\n=== lever H: cProfile of build_openfold3_features ===", flush=True)
        print(s.getvalue(), flush=True)
        a.out.write_text(json.dumps(res, indent=1))

    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
