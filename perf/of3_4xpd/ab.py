#!/usr/bin/env python3
"""One arm of the openfold3 512 aa fold A/B: cold fold discarded, then N warm walls.

No per-region timers, so the number is the uninstrumented fold wall the page cell is made
of. Reports the CIF digest and plDDT per fold (bit-exactness is asserted within one host
and wheel, never across -- qb1/0.67.4 and qb2/0.68.0 fold to different digests), and dumps
`FP32_SOFTMAX_STATS` so a run says which softmax path it took instead of assuming.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
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

    import importlib.metadata as im
    res = {"label": a.label, "ttnn": im.version("ttnn"), "host": os.uname().nodename,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "size": a.size,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "loadavg": open("/proc/loadavg").read().split()[:3], "folds": []}

    cold_s, cold_m = one_fold()
    res["cold_s"] = round(cold_s, 3)
    print(f"[{a.label}] cold {cold_s:.2f}s plddt={cold_m.get('plddt')}", flush=True)

    for i in range(a.repeat):
        fold_s, m = one_fold()
        rec = {"run": i, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
               "n_tokens": m.get("n_tokens"), "n_atoms": m.get("n_atoms"),
               "cif_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                              for p in sorted(struct_dir.glob("*")) if p.is_file()},
               "loadavg": open("/proc/loadavg").read().split()[:3]}
        res["folds"].append(rec)
        print(f"[{a.label}] warm {i}: {fold_s:.3f}s plddt={m.get('plddt')} "
              f"cif={list(rec['cif_sha256'].values())}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))

    walls = sorted(f["fold_s"] for f in res["folds"])
    res["median_s"] = walls[len(walls) // 2] if len(walls) % 2 else round(
        (walls[len(walls) // 2 - 1] + walls[len(walls) // 2]) / 2, 3)
    res["spread_s"] = round(walls[-1] - walls[0], 3)
    res["fp32_softmax_stats"] = dict(T.FP32_SOFTMAX_STATS)
    res["fp32_softmax_l1_refused_keys"] = sorted(str(k) for k in T._FP32_SOFTMAX_L1_REFUSED)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"[{a.label}] median {res['median_s']:.3f}s spread {res['spread_s']:.3f}s", flush=True)
    print(f"[{a.label}] FP32_SOFTMAX_STATS {res['fp32_softmax_stats']}", flush=True)
    print(f"[{a.label}] l1_refused_keys {res['fp32_softmax_l1_refused_keys']}", flush=True)
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
