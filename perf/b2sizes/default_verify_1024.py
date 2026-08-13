#!/usr/bin/env python3
"""Ship-gate check for the triatt q-split default: one boltz2 1024 aa fold with NO arm override
and TT_BIO_TRIATT_MASK_Q_SPLIT unset, so the process runs exactly what main would ship.

Proves three things on the branch tip (= current main + the ship commit):
  1. the default is ON and gated at <=1024 padded tokens (asserted, not assumed);
  2. the gate ADMITS the 1024 aa input (n_tokens <= 1024 and the split serves, seen via
     PM.STATS / _PM_OVER_L1 -- the q512 throw then q256 service, same as the stress arms);
  3. the CIF is digest-identical to the verified reference d3a02f9315060b3d.

Two folds back to back (cold + warm): the warm fold re-proves the digest after every kernel is
cached. Evidence JSON goes to perf/b2sizes/default_verify_1024_qb1.json.
"""
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "other512"))

import fold_ab_multi as H  # reuse patch_boltz2_cfg + sha_dir; no arm machinery is touched
import tt_baseline as B
from tt_bio import triatt_sdpa as PM
from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps

assert "TT_BIO_TRIATT_MASK_Q_SPLIT" not in os.environ, "env override must be unset for this run"
assert PM._Q_SPLIT is True, f"shipped default is not ON: {PM._Q_SPLIT}"
assert PM._Q_SPLIT_MAX_S == 1024, f"gate moved: {PM._Q_SPLIT_MAX_S}"

B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
B.SAMPLING_STEPS = _resolve_sampling_steps(None, "boltz2")
H.patch_boltz2_cfg()

fixdir = ROOT / "perf" / "size512" / "fixtures"
one_fold, meta, state = B.build_fold("boltz2", ROOT / ".msa_om512_1024",
                                     fixdir / "cdk2x2_1024.yaml", fixdir / "cdk2x2_1024.a3m")
struct_dir = Path(meta["struct_dir"])

res = {"ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
       "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "model": "boltz2", "size": 1024, "arm": "shipped-default",
       "q_split_default": PM._Q_SPLIT, "q_split_max_s": PM._Q_SPLIT_MAX_S,
       "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS, "runs": []}

for label in ("cold", "warm"):
    t0 = time.perf_counter()
    fold_s, m = one_fold()
    wall = time.perf_counter() - t0
    rec = {"label": label, "fold_s": round(fold_s, 3), "wall_s": round(wall, 3),
           "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt"),
           "cif_sha256": H.sha_dir(struct_dir),
           "persistent_mask": {"q_split": PM._Q_SPLIT, "served": PM.STATS[0],
                               "declined": PM.STATS[1],
                               "pm_over_l1": sorted(str(k) for k in PM._PM_OVER_L1)},
           "maxrss_mb": round(int(next(l for l in open("/proc/self/status")
                                       if l.startswith("VmHWM")).split()[1]) / 1024, 1)}
    res["runs"].append(rec)
    print(f"  {label}: {fold_s:.2f}s n_tokens={rec['n_tokens']} "
          f"sha={rec['cif_sha256']} served={PM.STATS[0]} declined={PM.STATS[1]} "
          f"VmHWM={rec['maxrss_mb']} MB", flush=True)

out = ROOT / "perf" / "b2sizes" / "default_verify_1024_qb1.json"
out.write_text(json.dumps(res, indent=1))
print(f"wrote {out}", flush=True)
