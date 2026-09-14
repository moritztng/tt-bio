"""One fold with NOTHING set: does the shipped default reach the device as rung 64?

The A/B set TT_BIO_MSA_LADDER explicitly on both arms, so it priced the flag, not the default.
This runs the tree as a user gets it and reads the depth off the MSA module's own call.
"""
import os, sys, time, json
from pathlib import Path
assert "TT_BIO_MSA_LADDER" not in os.environ, "the point is that nothing is set"
ROOT = Path(__file__).resolve().parents[1] if False else Path("/home/ttuser/.coworker/wt/roof-msa-ladder-bh-ship")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
sys.path.insert(0, str(ROOT / "perf" / "other512"))
import torch; torch.set_grad_enabled(False)
import ttnn, tt_bio.tenstorrent as T, tt_baseline as B
from fold_ab_multi import patch_boltz2_cfg
B.RECYCLING_STEPS = B._resolve_recycling_steps(None, "boltz2"); patch_boltz2_cfg(); T.get_device()
H = ROOT / "perf" / "roof_msa_ladder"
fix = ROOT / "perf" / "size512" / "fixtures"
one_fold, meta, _ = B.build_fold("boltz2", H / ".msa_512", fix / "cdk2x2_512.yaml", fix / "cdk2x2_512.a3m")
depths = []
o = T.MSA.__dict__["__call__"]
T.MSA.__call__ = lambda s, *a, **k: (depths.append(int(a[1].shape[1])), o(s, *a, **k))[1]
t, _m = one_fold()
T.MSA.__call__ = o
out = {"env_TT_BIO_MSA_LADDER": os.environ.get("TT_BIO_MSA_LADDER"), "n_msa": meta["n_msa"],
       "padded_depth_seen": sorted(set(depths)), "calls": len(depths), "fold_s": round(t, 3),
       "card_type": meta.get("card_type")}
print(json.dumps(out))
(H / "deployed_default.json").write_text(json.dumps(out, indent=2) + "\n")
T.cleanup()
