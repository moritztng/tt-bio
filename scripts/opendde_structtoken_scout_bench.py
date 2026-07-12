"""Profile StructuralTokenExpander standalone (the one novel OpenDDE compute block).

Models the timing rigor of scripts/kernel_scout_next_bench.py: realistic config + realistic
structural-token counts (measured from real targets), warm program cache, device-synchronized
timings, and a per-phase / per-device-op breakdown that answers the three scout questions:

  (a) are the 49 role-pair projections dispatched as ~49 separate small matmuls
      (dispatch-bound) or batched?
  (b) does pair_chunk_size=128 chunking cost avoidable DRAM round-trips vs a single chunk?
  (c) how much time does the bias-add epilogue take vs the projection matmul (fusion lever)?

Random weights at the real opendde_v1 config (same shapes/dtypes -> identical device op timing
to real weights; the expander block alone is parity-verified vs the real golden separately in
opendde_structtoken_parity.py, this harness is timing-only).

Run: PYTHONPATH=<worktree> TT_VISIBLE_DEVICES=0 \
    /home/moritz/tt-bio/env/bin/python3 scripts/opendde_structtoken_scout_bench.py
"""
import argparse
import json
import time
from collections import defaultdict

import torch
import ttnn

from tt_bio.tenstorrent import get_device
from tt_bio.opendde import StructuralTokenExpander, OPENDDE_CONFIG
from tt_bio.protenix_data import build_complex_features
from tt_bio.opendde_data import build_structural_token_features

torch.set_grad_enabled(False)

# Realistic structural-token counts, measured from real targets via
# build_structural_token_features (2026-07-12): 7ROA 117res -> Ns=229, hemoglobin 141res -> 275.
# 512/1024 exercise the chunking scaling regime for larger complexes.
DEFAULT_SIZES = [229, 275, 512, 1024]


def _make_inputs(n_res, c_s, c_z, c_s_inputs, seed=20260712):
    g = torch.Generator().manual_seed(seed)
    s_inputs_res = torch.randn(n_res, c_s_inputs, generator=g)
    s_res = torch.randn(n_res, c_s, generator=g)
    z_res = torch.randn(n_res, n_res, c_z, generator=g)
    # Synthesize a protein-only feature dict of the right length, then derive the real
    # structural-token ifd from it so role/parent/adjacency maps are realistic (not all-one-role).
    seq = "AGWKSG" * (n_res // 6 + 1)
    seq = seq[:n_res]
    feats = build_complex_features([(seq, None, "protein")])
    # force the residue count to exactly n_res (build_complex_features may pad)
    for k in ("asym_id", "residue_index", "entity_id", "sym_id"):
        if feats[k].shape[0] > n_res:
            feats[k] = feats[k][:n_res]
    feats["restype"] = feats["restype"][:n_res]
    ifd = build_structural_token_features(feats)
    # ifd's Ns may differ slightly from the requested size; trim/replicate parent/role to match.
    return s_inputs_res, s_res, z_res, ifd


def _pcc(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


class TimedExpander(StructuralTokenExpander):
    """Wrap device ops + phase methods with synchronized timing + call counts."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.dev = get_device()
        self.totals = defaultdict(float)
        self.calls = defaultdict(int)
        self._sync = True

    def _sync_now(self):
        if self._sync:
            ttnn.synchronize_device(self.dev)

    # device-op wrappers (synced)
    def _up(self, t):
        self._sync_now()
        s = time.perf_counter()
        r = super()._up(t)
        self._sync_now()
        self.totals["up"] += time.perf_counter() - s
        self.calls["up"] += 1
        return r

    def _lin(self, x, wkey, bkey=None, activation=None):
        self._sync_now()
        s = time.perf_counter()
        r = super()._lin(x, wkey, bkey, activation)
        self._sync_now()
        self.totals["linear"] += time.perf_counter() - s
        self.calls["linear"] += 1
        return r

    def _ln(self, x, wkey, bkey=None):
        self._sync_now()
        s = time.perf_counter()
        r = super()._ln(x, wkey, bkey)
        self._sync_now()
        self.totals["layernorm"] += time.perf_counter() - s
        self.calls["layernorm"] += 1
        return r

    # phase wrappers (host+device, synced at boundaries)
    def _pair_project_full(self, z_chunk_h, role, row_index):
        self._sync_now()
        s = time.perf_counter()
        r = super()._pair_project_full(z_chunk_h, role, row_index)
        self._sync_now()
        self.totals["pair_project_full"] += time.perf_counter() - s
        self.calls["pair_project_full"] += 1
        return r

    def _pair_init_bias(self, pf):
        self._sync_now()
        s = time.perf_counter()
        r = super()._pair_init_bias(pf)
        self._sync_now()
        self.totals["pair_init_bias"] += time.perf_counter() - s
        self.calls["pair_init_bias"] += 1
        return r

    def _attn_bias(self, pf):
        self._sync_now()
        s = time.perf_counter()
        r = super()._attn_bias(pf)
        self._sync_now()
        self.totals["attn_bias"] += time.perf_counter() - s
        self.calls["attn_bias"] += 1
        return r


def _run(mod, ifd, s_inputs_res, s_res, z_res):
    ttnn.synchronize_device(mod.dev)
    s = time.perf_counter()
    out = mod(ifd, s_inputs_res, s_res, z_res)
    ttnn.synchronize_device(mod.dev)
    return time.perf_counter() - s, out


def _rand_state_dict(c_s, c_z, c_s_inputs, n_roles=7, seed=20260712):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    sd["single_input_role_embedding.weight"] = torch.randn(n_roles, c_s_inputs, generator=g)
    sd["single_role_embedding.weight"] = torch.randn(n_roles, c_s, generator=g)
    sd["single_split_mlp.0.weight"] = torch.randn(c_s, generator=g)
    sd["single_split_mlp.0.bias"] = torch.randn(c_s, generator=g)
    sd["single_split_mlp.1.weight"] = torch.randn(2 * c_s, c_s, generator=g)
    sd["single_split_mlp.3.weight"] = torch.randn(c_s, 2 * c_s, generator=g)
    for i in range(n_roles * n_roles):
        sd["pair_block_proj.%d.weight" % i] = torch.randn(c_z, c_z, generator=g) * 0.1
    for nm in ["same_parent_embedding", "same_residue_twin_embedding", "prev_bb_chain_embedding",
               "next_bb_chain_embedding"]:
        sd[nm + ".weight"] = torch.randn(2, c_z, generator=g) * 0.1
    # role_pair_type values are 0..7 (7 named role-pair combos + default); embeddings indexed by it.
    sd["role_pair_type_embedding.weight"] = torch.randn(8, c_z, generator=g) * 0.1
    sd["attn_bias_role_pair_type"] = torch.randn(8, generator=g)
    sd["attn_bias_same_parent"] = torch.randn(1, generator=g)
    sd["attn_bias_same_residue_twin"] = torch.randn(1, generator=g)
    sd["attn_bias_prev_bb_chain"] = torch.randn(1, generator=g)
    sd["attn_bias_next_bb_chain"] = torch.randn(1, generator=g)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--chunk-sweep", action="store_true",
                    help="also time pair_chunk_size=Ns (single chunk) vs 128")
    ap.add_argument("--real-weights", action="store_true",
                    help="use the real opendde.pt expander subtree + real 7ROA ifd "
                         "(breakdown traceable to real weights; one size only)")
    args = ap.parse_args()
    C = OPENDDE_CONFIG
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)

    real_sd = None
    real_ifd = None
    real_n_res = None
    if args.real_weights:
        from tt_bio.opendde import load_opendde_checkpoint, route_opendde_weights
        from tt_bio.protenix_data import build_complex_features
        SEQ_7ROA = ("QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKA"
                    "WKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG")
        routed = route_opendde_weights(load_opendde_checkpoint())
        real_sd = routed["expander"]
        rf = build_complex_features([(SEQ_7ROA, None, "protein")])
        real_ifd = build_structural_token_features(rf)
        real_n_res = rf["restype"].shape[0]
        print("real-weights mode: 7ROA n_res=%d Ns=%d" % (real_n_res, real_ifd["parent_residue_idx"].shape[0]), flush=True)

    results = []
    if args.real_weights:
        # single run with real expander weights + real 7ROA ifd; random activations of real shapes.
        Ns = real_ifd["parent_residue_idx"].shape[0]
        nr = real_n_res
        g = torch.Generator().manual_seed(20260712)
        s_inputs_res = torch.randn(nr, C["c_s_inputs"], generator=g)
        s_res = torch.randn(nr, C["c_s"], generator=g)
        z_res = torch.randn(nr, nr, C["c_z"], generator=g)
        for chunk in (C["pair_chunk_size"], Ns):
            mod = TimedExpander(real_sd, ckc, c_s=C["c_s"], c_z=C["c_z"], c_s_inputs=C["c_s_inputs"],
                                n_roles=C["n_roles"], pair_chunk_size=chunk)
            _run(mod, real_ifd, s_inputs_res, s_res, z_res)
            mod.totals.clear(); mod.calls.clear()
            t, out = _run(mod, real_ifd, s_inputs_res, s_res, z_res)
            rec = {"mode": "real_weights", "Ns": Ns, "n_res": nr, "pair_chunk_size": chunk,
                   "total_s": round(t, 4),
                   "device_op_s": {k: round(v, 4) for k, v in mod.totals.items()},
                   "device_op_calls": dict(mod.calls),
                   "n_chunks": (Ns + chunk - 1) // chunk}
            _, out2 = _run(mod, real_ifd, s_inputs_res, s_res, z_res)
            zs1 = torch.Tensor(ttnn.to_torch(out[2])).float()
            zs2 = torch.Tensor(ttnn.to_torch(out2[2])).float()
            rec["determinism_z_pcc"] = round(_pcc(zs1, zs2), 6)
            rec["finite"] = bool(torch.isfinite(zs1).all().item())
            print(json.dumps(rec, sort_keys=True), flush=True)
            results.append(rec)
        print("ALL " + json.dumps(results))
        return

    results = []
    for size in args.sizes:
        n_res = max(size // 2, 6)  # Ns ~= 2*n_res; pick n_res so requested Ns is approx hit
        s_inputs_res, s_res, z_res, ifd = _make_inputs(n_res, C["c_s"], C["c_z"], C["c_s_inputs"])
        Ns = ifd["parent_residue_idx"].shape[0]
        # adjust z_res/s_res/s_inputs_res to the actual n_res of the synthesized feats
        nr = feats_n_res = s_res.shape[0]
        z_res = z_res[:nr, :nr]
        sd = _rand_state_dict(C["c_s"], C["c_z"], C["c_s_inputs"])
        mod = TimedExpander(sd, ckc, c_s=C["c_s"], c_z=C["c_z"], c_s_inputs=C["c_s_inputs"],
                            n_roles=C["n_roles"], pair_chunk_size=C["pair_chunk_size"])
        # warm
        _, _ = _run(mod, ifd, s_inputs_res, s_res, z_res)
        mod.totals.clear(); mod.calls.clear()
        t, out = _run(mod, ifd, s_inputs_res, s_res, z_res)
        rec = {"Ns": Ns, "n_res": nr, "total_s": round(t, 4),
               "device_op_s": {k: round(v, 4) for k, v in mod.totals.items()},
               "device_op_calls": dict(mod.calls),
               "n_chunks": (Ns + C["pair_chunk_size"] - 1) // C["pair_chunk_size"]}
        # chunk sweep: single chunk vs 128
        if args.chunk_sweep:
            mod1 = TimedExpander(sd, ckc, c_s=C["c_s"], c_z=C["c_z"], c_s_inputs=C["c_s_inputs"],
                                 n_roles=C["n_roles"], pair_chunk_size=Ns)
            _run(mod1, ifd, s_inputs_res, s_res, z_res)
            mod1.totals.clear(); mod1.calls.clear()
            t1, _ = _run(mod1, ifd, s_inputs_res, s_res, z_res)
            rec["single_chunk_total_s"] = round(t1, 4)
            rec["single_chunk_speedup"] = round(t / t1, 3) if t1 > 0 else None
        # determinism / finiteness check
        _, out2 = _run(mod, ifd, s_inputs_res, s_res, z_res)
        si1, ss1, zs1, ab1 = out
        si2, ss2, zs2, ab2 = out2
        zs_h1 = torch.Tensor(ttnn.to_torch(zs1)).float()
        zs_h2 = torch.Tensor(ttnn.to_torch(zs2)).float()
        rec["determinism_z_pcc"] = round(_pcc(zs_h1, zs_h2), 6)
        rec["finite"] = bool(torch.isfinite(zs_h1).all().item())
        print(json.dumps(rec, sort_keys=True), flush=True)
        results.append(rec)
        # cleanup
        for o in (si1, ss1, zs1, ab1, si2, ss2, zs2, ab2):
            try:
                if hasattr(o, "value"):
                    ttnn.deallocate(o)
            except Exception:
                pass
        del mod, out, out2
    print("ALL " + json.dumps(results))


if __name__ == "__main__":
    main()
