"""Is the MSA depth ladder free? One process, one card, arms interleaved, structures compared.

Arm A (incumbent) pads the alignment's rows to the single 1024 bucket. Arm B
(``TT_BIO_MSA_DEPTH_LADDER=1``) pads to the smallest ladder rung that holds them. The claim under
test is BIT-EXACT, so the acceptance test is the structure digest, not a time: 989 rows of zeros
behind a row mask must not reach the answer. Wall clock is recorded because the box is shared and
an absolute from it is not a perf claim -- the ratio and its A/A floor are.
"""
import argparse
import hashlib
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "other512"))

import torch                                                          # noqa: E402
torch.set_grad_enabled(False)

FLAG = "TT_BIO_MSA_DEPTH_LADDER"


def sha_dir(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=None)
    ap.add_argument("--a3m", default=None,
                    help="override the fixture's alignment, to measure the win AGAINST depth")
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from fold_ab_multi import patch_boltz2_cfg

    if a.sampling_steps:
        B.SAMPLING_STEPS = a.sampling_steps
    patch_boltz2_cfg()
    target = ROOT / "perf/size512/fixtures" / f"cdk2x2_{a.tokens}.yaml"
    a3m = Path(a.a3m) if a.a3m else target.with_suffix(".a3m")
    n_msa_true = a3m.read_text().count(">")
    one_fold, meta, state = B.build_fold(
        "boltz2", ROOT / f".msa_b2zwr_{a3m.stem}", target, a3m)
    struct_dir = Path(meta["struct_dir"])

    from tt_bio.token_axis import msa_depth_bucket

    def fold(arm: str) -> dict:
        os.environ[FLAG] = "1" if arm == "ladder" else "0"
        for f in struct_dir.glob("*"):
            f.unlink()
        t0 = time.perf_counter()
        secs, m = one_fold()
        return {"arm": arm, "wall_s": round(secs, 4),
                "elapsed_s": round(time.perf_counter() - t0, 4),
                "plddt": m.get("plddt"), "n_msa": m.get("n_msa"),
                "sha": sha_dir(struct_dir)}

    os.environ[FLAG] = "0"
    cold = fold("cold")                                   # discarded: warms every kernel cache
    rows = [cold]
    for _ in range(a.pairs):
        rows.append(fold("base"))
        rows.append(fold("ladder"))
        rows.append(fold("base"))                         # A/A floor rides in the same run
        print(json.dumps(rows[-3]), flush=True)
        print(json.dumps(rows[-2]), flush=True)
        print(json.dumps(rows[-1]), flush=True)

    base = [r["wall_s"] for r in rows[1:] if r["arm"] == "base"]
    lad = [r["wall_s"] for r in rows[1:] if r["arm"] == "ladder"]
    shas_base = {json.dumps(r["sha"], sort_keys=True) for r in rows[1:] if r["arm"] == "base"}
    shas_lad = {json.dumps(r["sha"], sort_keys=True) for r in rows[1:] if r["arm"] == "ladder"}
    aa = [abs(base[i] - base[i + 1]) / base[i + 1] for i in range(len(base) - 1)]

    import importlib.metadata as im
    import socket
    n_msa = rows[1].get("n_msa") or n_msa_true
    out = {
        "host": socket.gethostname(), "arch": T.arch_name(),
        "chip": os.environ.get("TT_VISIBLE_DEVICES", "?"), "ttnn": im.version("ttnn"),
        "tokens": a.tokens, "sampling_steps": B.SAMPLING_STEPS,
        "a3m": str(a3m), "n_msa_true": n_msa_true,
        "recycling_steps": meta.get("recycling_steps"),
        "n_msa": n_msa,
        "padded_depth_base": ((n_msa_true + 1023) // 1024) * 1024,
        "padded_depth_ladder": None,
        "median_base_s": round(st.median(base), 4) if base else None,
        "median_ladder_s": round(st.median(lad), 4) if lad else None,
        "ratio": round(st.median(base) / st.median(lad), 4) if base and lad else None,
        "aa_floor_pct": round(100 * st.median(aa), 3) if aa else None,
        "bit_exact": len(shas_base) == 1 and shas_base == shas_lad,
        "sha_base": sorted(shas_base), "sha_ladder": sorted(shas_lad),
        "rows": rows,
    }
    os.environ[FLAG] = "1"
    if n_msa:
        out["padded_depth_ladder"] = msa_depth_bucket(int(n_msa))
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("median_base_s", "median_ladder_s", "ratio",
                                          "aa_floor_pct", "bit_exact", "n_msa",
                                          "padded_depth_ladder")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
