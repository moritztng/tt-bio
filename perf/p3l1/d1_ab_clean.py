#!/usr/bin/env python3
"""E1 clean fold A/B — one arm per process. Full predict_one (featurize + fold + CIF write),
so the fold wall is the real tt_baseline.py wall and a CIF sha256 is captured per fold for
determinism verification. Arms alternate via the shell loop (base/changed, separate processes
because the D1 edit is a code change imported at process start).

For a non-bit-exact change the CIF shas DIFFER between arms (reduction order changes through
200 sampling steps); the sha is to confirm each arm is deterministic across its reps (the
within-arm sha is stable), not to match across arms. plddt match across arms is the smoke
parity signal; the full integration envelope needs cached CPU refs (BLOCKED-REF-REGEN).
"""
import argparse, hashlib, json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def cif_sha256(struct_dir: Path) -> str | None:
    cifs = sorted(struct_dir.glob("*.cif"))
    if not cifs:
        return None
    h = hashlib.sha256()
    with open(cifs[0], "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16], cifs[0].name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--repeat", type=int, default=1, help="warm folds after the cold one")
    ap.add_argument("--target", type=Path, default=ROOT / "examples" / "prot300.yaml")
    ap.add_argument("--a3m", type=Path, default=ROOT / "scripts/gpu_vs_tt/fixtures/prot300.a3m")
    ap.add_argument("--msa-dir", type=Path, default=ROOT / ".msa_d1ab")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_baseline as B
    one_fold, meta, state = B.build_fold("protenix-v2", a.msa_dir, a.target, a.a3m)
    struct_dir = Path(meta["struct_dir"])

    cold_s, cold_m = one_fold()
    assert cold_m.get("msa"), "fold ran without an MSA"
    cold_sha = cif_sha256(struct_dir)

    folds = []
    for i in range(a.repeat):
        t, m = one_fold()
        sha = cif_sha256(struct_dir)
        folds.append({"wall_s": round(t, 3), "plddt": m.get("plddt"),
                       "cif_sha16": sha[0] if sha else None, "cif_name": sha[1] if sha else None})

    res = {"arm": a.arm, "cold_s": round(cold_s, 3), "cold_plddt": cold_m.get("plddt"),
           "cold_cif_sha16": cold_sha[0] if cold_sha else None,
           "fold_walls_s": [f["wall_s"] for f in folds],
           "median_wall_s": round(st.median([f["wall_s"] for f in folds]), 3) if folds else None,
           "min_wall_s": round(min(f["wall_s"] for f in folds), 3) if folds else None,
           "folds": folds,
           "n_tokens": cold_m.get("n_tokens"),
           "tt_bio_git": B._git_sha() if hasattr(B, "_git_sha") else None}
    a.out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "folds"}, indent=2), flush=True)
    state.reset()
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
