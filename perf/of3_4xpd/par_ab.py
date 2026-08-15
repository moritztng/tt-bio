#!/usr/bin/env python3
"""Conformer-pool A/B: the same 512 embeddings, computed concurrently.

Alternates `CONFORMER_THREADS` between folds in one process (off/on/off/on after one cold fold
per arm), keeps every arm's CIF so the accuracy cost can be measured, and times
`build_openfold3_features` per fold so the host half is priced directly rather than
inferred from the fold wall.

The memo is NOT bit-exact by construction: `_compute_conformer` draws its ETKDG seed from
python's `random` per call, so today each of the 512 residues carries its own randomly
seeded conformer. This harness measures what that is worth and what it costs.
"""
import argparse, hashlib, json, os, shutil, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

ANCHOR_PLDDT = float(os.environ.get("ANCHOR_PLDDT", "0.547851"))
ANCHOR_CIF = os.environ.get("ANCHOR_CIF", "da9b4ed68f8c0405")
THREADS = int(os.environ.get("CONFORMER_THREADS", "24"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_bio.openfold3_data as OD
    import tt_baseline as B
    from tt_bio._vendor.openfold3.core.data.primitives.structure import query as Q
    from tt_bio.main import (_resolve_recycling_steps, _resolve_sampling_steps,
                             _detect_p300_devices, _find_ttnn_mesh_graph_descriptor)
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    assert Path(T.__file__).resolve().is_relative_to(ROOT), f"tt_bio from {T.__file__}, set PYTHONPATH"

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "openfold3")
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, "openfold3")

    feat_s = []
    g = OD.build_openfold3_features

    def timed_features(*x, **k):
        t0 = time.perf_counter()
        out = g(*x, **k)
        feat_s.append(time.perf_counter() - t0)
        return out
    OD.build_openfold3_features = timed_features

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    one_fold, meta = B.build_fold("openfold3", ROOT / f".msa_of3deep_{a.size}", tgt, a3m)[:2]
    struct_dir = Path(meta["struct_dir"])
    a.cifdir.mkdir(parents=True, exist_ok=True)

    res = {"anchor": {"plddt": ANCHOR_PLDDT, "cif": ANCHOR_CIF},
           "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "folds": []}

    def leg(tag, memo):
        Q.CONFORMER_THREADS = THREADS if memo else 0
        feat_s.clear()
        fold_s, m = one_fold()
        cifs = [q for q in sorted(struct_dir.glob("*")) if q.is_file()]
        d = [hashlib.sha256(q.read_bytes()).hexdigest()[:16] for q in cifs]
        for q in cifs:
            shutil.copy(q, a.cifdir / f"{tag}_{'pool' if memo else 'plain'}_{q.name}")
        rec = {"tag": tag, "pool": memo, "threads": THREADS if memo else 0, "fold_s": round(fold_s, 3),
               "build_features_s": round(feat_s[0], 3) if feat_s else None,
               "plddt": m.get("plddt"), "cif": d,
               "matches_anchor": bool(d and d[0] == ANCHOR_CIF
                                      and abs(float(m.get("plddt")) - ANCHOR_PLDDT) < 5e-6)}
        res["folds"].append(rec)
        print(f"[P] {tag:8s} pool={memo!s:5s} fold {fold_s:7.3f}s  feat "
              f"{rec['build_features_s']}s  plddt {m.get('plddt')}  cif {d}  "
              f"{'MATCH' if rec['matches_anchor'] else 'DIFFERS'}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))

    leg("cold0", False)
    leg("cold1", True)
    for i in range(a.repeat):
        leg(f"w{i}a", False)
        leg(f"w{i}b", True)

    for memo in (False, True):
        w = sorted(f["fold_s"] for f in res["folds"] if f["pool"] is memo and f["tag"][0] == "w")
        h = sorted(f["build_features_s"] for f in res["folds"]
                   if f["pool"] is memo and f["tag"][0] == "w")
        res[f"pool_{memo}"] = {"fold_median": round(statistics.median(w), 3),
                               "fold_spread": round(w[-1] - w[0], 3),
                               "feat_median": round(statistics.median(h), 3)}
        print(f"[P] pool={memo}: fold median {res[f'pool_{memo}']['fold_median']:.3f}s "
              f"spread {res[f'pool_{memo}']['fold_spread']:.3f}s  "
              f"build_features median {res[f'pool_{memo}']['feat_median']:.3f}s", flush=True)
    res["fold_delta_s"] = round(res["pool_False"]["fold_median"] - res["pool_True"]["fold_median"], 3)
    res["feat_delta_s"] = round(res["pool_False"]["feat_median"] - res["pool_True"]["feat_median"], 3)
    res["ratio"] = round(res["pool_False"]["fold_median"] / res["pool_True"]["fold_median"], 5)
    print(f"[P] fold -{res['fold_delta_s']}s ({res['ratio']}x), "
          f"build_features -{res['feat_delta_s']}s", flush=True)
    a.out.write_text(json.dumps(res, indent=1))
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
