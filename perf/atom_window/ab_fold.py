#!/usr/bin/env python3
"""Fold-level A/B for the batch-parallel atom-window matmul, one process, one model load.

Arm A = AtomTransformer._bmm as committed (batch on the core grid).
Arm B = the same call sites forced back to the plain ttnn.matmul.
Arms are interleaved so host drift cannot favour either. Coordinates are compared with
torch.equal: the two arms must produce bit-identical structures.

Also counts _bmm calls per fold, which is the 1200 calls/fold figure W10 reported.
"""
import argparse, json, statistics as st, tempfile, time
from pathlib import Path

import torch
import ttnn

torch.set_grad_enabled(False)
REPO = Path(__file__).resolve().parents[2]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="protenix-v2", choices=["protenix-v2", "opendde"])
ap.add_argument("--repeat", type=int, default=2)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from tt_bio.tenstorrent import get_device, arch_name
from tt_bio.worker import _WorkerState, _ensure_local_artifacts
from tt_bio import esmfold2 as _E
from tt_bio.protenix import AtomTransformer
import tt_bio.main as _main
from tt_bio.main import _read_bio_chains, _read_bio_constraints, _resolve_a3m_text
from tt_bio.protenix_data import build_complex_features
import sys
sys.path.insert(0, str(REPO / "scripts" / "gpu_vs_tt"))
from tt_baseline import seed_msa_cache  # noqa: E402

_noop = lambda *x, **k: None
_E.set_progress(_noop)
dev = get_device()

target = REPO / "examples" / "prot300.yaml"
a3m = REPO / "scripts" / "gpu_vs_tt" / "fixtures" / "prot300.a3m"
work = Path(tempfile.mkdtemp(prefix="awab-"))
msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
struct_dir = work / "out"; struct_dir.mkdir(parents=True)
cfg = dict(model=a.model, fast=False, output_format="cif", recycling_steps=10,
           sampling_steps=200, diffusion_samples=1, seed=0, trace=False,
           msa_dir=str(msa_dir), struct_dir=str(struct_dir), use_msa_server=True,
           msa_db_path=None, use_envdb=False, msa_endpoint=None, single_sequence=False,
           msa_server_url="https://api.colabfold.com", msa_pairing_strategy="greedy",
           msa_server_username=None, msa_server_password=None, api_key_value=None,
           max_msa_seqs=8192, write_pae=False, write_pde=False, write_embeddings=False,
           method=None)
_ensure_local_artifacts(cfg)
seed_msa_cache(target, a3m, msa_dir)
state = _WorkerState("tenstorrent")
state.load_model(cfg)
state.bind_run("awab", cfg)
state.pfn = _noop

chains = _read_bio_chains(target)
bonds = _read_bio_constraints(target)
specs = [(cseq, _resolve_a3m_text(spec, cseq, msa_dir) if mt == "protein" else None, mt)
         for _cid, cseq, spec, mt in chains]
feats = build_complex_features(specs, mol_dir=cfg.get("mol_dir"),
                               chain_ids=[c for c, _s, _sp, _m in chains], bonds=bonds)
print(f"tokens={int(feats['restype'].shape[0])} arch={arch_name()}", flush=True)

FAST = AtomTransformer._bmm
CALLS = {"n": 0, "shapes": {}}


def counted(self, x, y, in0_block_w=1):
    CALLS["n"] += 1
    k = f"{tuple(x.shape)}x{tuple(y.shape)}"
    CALLS["shapes"][k] = CALLS["shapes"].get(k, 0) + 1
    return FAST(self, x, y, in0_block_w)


def plain(self, x, y, in0_block_w=1):
    CALLS["n"] += 1
    return ttnn.matmul(x, y, compute_kernel_config=self.compute_kernel_config)


def fold():
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    coords, _ = state.model.fold(dict(feats), n_step=200, n_sample=1, seed=0,
                                 progress_fn=_noop, return_confidence=True, n_cycles=10,
                                 max_parallel_samples=None, trace=False)
    ttnn.synchronize_device(dev)
    return time.perf_counter() - t0, coords


AtomTransformer._bmm = counted
t, _ = fold()                      # cold: kernel compile, never timed
print(f"cold {t:.3f} s   _bmm calls/fold = {CALLS['n']}", flush=True)
per_fold_calls = CALLS["n"]
shapes = dict(CALLS["shapes"])
AtomTransformer._bmm = plain
tb, _ = fold()                     # arm B's own kernel compile, never timed either
print(f"cold-B {tb:.3f} s", flush=True)

res = {"model": a.model, "card": "qb2-card1", "ttnn": getattr(ttnn, "__version__", "?"),
       "bmm_calls_per_fold": per_fold_calls, "bmm_shapes": shapes, "A": [], "B": []}
ref = {}
for i in range(a.repeat):
    for arm, fn in (("A", counted), ("B", plain)):
        AtomTransformer._bmm = fn
        s, c = fold()
        res[arm].append(round(s, 4))
        cc = c[0] if isinstance(c, (list, tuple)) else c
        cc = cc.detach().cpu() if hasattr(cc, "detach") else torch.as_tensor(cc)
        ref.setdefault(arm, cc)
        print(f"  {arm} fold {i}: {s:.3f} s", flush=True)

eq = bool(torch.equal(ref["A"], ref["B"]))
d = (ref["A"].float() - ref["B"].float()).abs().max().item()
ma, mb = st.median(res["A"]), st.median(res["B"])
res.update({"median_A_s": round(ma, 4), "median_B_s": round(mb, 4),
            "saved_ms_per_fold": round((mb - ma) * 1e3, 1), "speedup": round(mb / ma, 4),
            "coords_bit_exact": eq, "coords_max_abs": d})
print(f"\n{a.model}: A(batch-parallel) {ma:.3f} s   B(plain matmul) {mb:.3f} s   "
      f"saved {(mb-ma)*1e3:.1f} ms/fold ({mb/ma:.4f}x)   coords torch.equal={eq} maxabs={d:.3e}",
      flush=True)
json.dump(res, open(a.out, "w"), indent=2)
print("wrote", a.out, flush=True)
