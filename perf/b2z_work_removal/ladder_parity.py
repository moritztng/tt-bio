"""Does the MSA row mask hold, and how far does the ladder move the answer?

The ladder is NOT bit-identical to the single 1024 bucket (measured: base ab20569e, ladder
e9c020f8, each reproducible). Two causes have opposite verdicts:

  * bf16 reassociation of the real rows over a shorter axis -- benign, the same thing the token
    bucket does on openfold3, and then the only question left is how many Angstrom it costs;
  * a leaking row mask -- a bug the ladder exposes rather than creates, and a blocker.

`TT_BIO_MSA_PAD_POISON` separates them. Fill the padded rows with a non-zero value: a correctly
masked axis cannot let a padded row reach a real one, so the structure must be byte-identical at
any poison. This runs that 2x2 (bucket x poison) in ONE process on ONE card and keeps every CIF,
so the arm-vs-arm RMSD can be scored afterwards with the shared control_rmsd.py.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "other512"))

import torch                                                          # noqa: E402
torch.set_grad_enabled(False)

LADDER = "TT_BIO_MSA_DEPTH_LADDER"
POISON = "TT_BIO_MSA_PAD_POISON"


def sha_dir(d: Path) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cifdir", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--a3m", default=None, help="override the fixture's a3m (depth sweep)")
    ap.add_argument("--sampling-steps", type=int, default=None)
    ap.add_argument("--poison", default="1000.0")
    ap.add_argument("--repeat-base", action="store_true",
                    help="re-fold the first arm at the end as an A/A determinism control")
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_baseline as B
    from fold_ab_multi import patch_boltz2_cfg

    if a.sampling_steps:
        B.SAMPLING_STEPS = a.sampling_steps
    patch_boltz2_cfg()
    target = ROOT / "perf/size512/fixtures" / f"cdk2x2_{a.tokens}.yaml"
    a3m = Path(a.a3m) if a.a3m else target.with_suffix(".a3m")
    tag = a3m.stem
    one_fold, meta, state = B.build_fold(
        "boltz2", ROOT / f".msa_b2zwr_{tag}", target, a3m)
    struct_dir = Path(meta["struct_dir"])
    cifdir = Path(a.cifdir)
    cifdir.mkdir(parents=True, exist_ok=True)

    def fold(ladder: str, poison: str, label: str) -> dict:
        os.environ[LADDER] = ladder
        os.environ[POISON] = poison
        for f in struct_dir.glob("*"):
            f.unlink()
        secs, m = one_fold()
        sha = sha_dir(struct_dir)
        for f in sorted(struct_dir.glob("*")):
            shutil.copy2(f, cifdir / f"{label}__{f.name}")
        row = {"label": label, "ladder": ladder, "poison": poison,
               "wall_s": round(secs, 4), "plddt": m.get("plddt"), "sha": sha}
        print(json.dumps(row), flush=True)
        return row

    arms = [
        ("0", "0",       "bucket1024_poison0"),
        ("0", a.poison,  "bucket1024_poisonX"),
        ("1", "0",       "ladder_poison0"),
        ("1", a.poison,  "ladder_poisonX"),
    ]
    os.environ[LADDER], os.environ[POISON] = "0", "0"
    for f in struct_dir.glob("*"):
        f.unlink()
    one_fold()                                            # cold fold, discarded
    rows = [fold(*x) for x in arms]
    if a.repeat_base:
        rows.append(fold("0", "0", "bucket1024_poison0_repeat"))

    by = {r["label"]: json.dumps(r["sha"], sort_keys=True) for r in rows}
    verdict = {
        "bucket1024_mask_holds": by["bucket1024_poison0"] == by["bucket1024_poisonX"],
        "ladder_mask_holds": by["ladder_poison0"] == by["ladder_poisonX"],
        "ladder_equals_bucket": by["bucket1024_poison0"] == by["ladder_poison0"],
    }
    if a.repeat_base:
        verdict["aa_deterministic"] = (
            by["bucket1024_poison0"] == by["bucket1024_poison0_repeat"])

    import importlib.metadata as im
    import socket
    out = {"host": socket.gethostname(), "arch": T.arch_name(),
           "chip": os.environ.get("TT_VISIBLE_DEVICES", "?"), "ttnn": im.version("ttnn"),
           "tokens": a.tokens, "a3m": str(a3m), "sampling_steps": B.SAMPLING_STEPS,
           "recycling_steps": meta.get("recycling_steps"), "poison_value": a.poison,
           "verdict": verdict, "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(verdict), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
