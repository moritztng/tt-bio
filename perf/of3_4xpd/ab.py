#!/usr/bin/env python3
"""One arm of the openfold3 512 aa fold A/B: cold fold discarded, then N warm walls.

No per-region timers, so the number is the uninstrumented fold wall the page cell is made
of. Reports the CIF digest and plDDT per fold (bit-exactness is asserted within one host
and wheel, never across -- qb1/0.67.4 and qb2/0.68.0 fold to different digests), and dumps
`FP32_SOFTMAX_STATS` so a run says which softmax path it took instead of assuming.

``--altflag NAME`` flips a `tt_bio.tenstorrent` module global between folds and alternates
the arms in ONE process (cold fold per arm first, then on/off/on/off), which is the
cheapest honest A/B for a lever that is already behind a runtime gate.
"""
import argparse, hashlib, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--altflag", default=None,
                    help="tt_bio.tenstorrent global to alternate per fold (True/False arms)")
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
           "altflag": a.altflag,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "loadavg": open("/proc/loadavg").read().split()[:3], "folds": []}

    def set_arm(v):
        if a.altflag:
            assert hasattr(T, a.altflag), f"no tt_bio.tenstorrent.{a.altflag}"
            setattr(T, a.altflag, v)

    def one(tag, arm):
        set_arm(arm)
        fold_s, m = one_fold()
        return {"tag": tag, "arm": arm, "fold_s": round(fold_s, 3), "plddt": m.get("plddt"),
                "n_tokens": m.get("n_tokens"), "n_atoms": m.get("n_atoms"),
                "cif_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                               for p in sorted(struct_dir.glob("*")) if p.is_file()},
                "loadavg": open("/proc/loadavg").read().split()[:3]}

    arms = [True, False] if a.altflag else [None]
    for arm in arms:                                  # one cold fold per arm, discarded
        c = one("cold", arm)
        print(f"[{a.label}] cold arm={arm} {c['fold_s']:.2f}s plddt={c['plddt']}", flush=True)
        res.setdefault("cold", []).append(c)

    seq = [arm for _ in range(a.repeat) for arm in arms]
    for i, arm in enumerate(seq):
        rec = one(f"warm{i}", arm)
        res["folds"].append(rec)
        print(f"[{a.label}] warm {i} arm={arm}: {rec['fold_s']:.3f}s plddt={rec['plddt']} "
              f"cif={list(rec['cif_sha256'].values())}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))

    res["by_arm"] = {}
    for arm in arms:
        w = sorted(f["fold_s"] for f in res["folds"] if f["arm"] == arm)
        res["by_arm"][str(arm)] = {"walls": w, "median": round(statistics.median(w), 3),
                                   "spread": round(w[-1] - w[0], 3)}
        print(f"[{a.label}] arm={arm} median {res['by_arm'][str(arm)]['median']:.3f}s "
              f"spread {res['by_arm'][str(arm)]['spread']:.3f}s", flush=True)
    res["fp32_softmax_stats"] = dict(T.FP32_SOFTMAX_STATS)
    res["fp32_softmax_l1_row_caps"] = {str(k): v for k, v in T._FP32_SOFTMAX_L1_ROW_CAP.items()}
    res["median_s"] = res["by_arm"][str(arms[0])]["median"]
    res["spread_s"] = res["by_arm"][str(arms[0])]["spread"]
    a.out.write_text(json.dumps(res, indent=1))
    print(f"[{a.label}] FP32_SOFTMAX_STATS {res['fp32_softmax_stats']}", flush=True)
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
