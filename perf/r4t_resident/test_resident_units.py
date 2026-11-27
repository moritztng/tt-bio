"""Hardware verification for u32 fp32 ops, DeviceHeads + device_insertion_loss, DeviceFolds, and DeviceOptimizer.

Runs on one Blackhole chip (anx-glx or qb2) and verifies:
0. u32 bitcast op smoke: verifies uint32 bitcast + clone/slice/concat/repeat preserve exact fp32 bits.
1. DeviceHeads + device_insertion_loss on real batch08 (all 5 loss terms active):
   - FD-validated float64 reference (fd_check < 2e-6)
   - Grad-off byte-identical A/A + forward parity vs host FlowGlue.heads
   - Taped VJP graded per D28 on d_s + all 8 head weights with 0 pair-sized PCIe transfers
   - Broken arms (gross + subtle x1.10) both FAIL
2. DeviceFolds on full RFD4 Trunk + prewarm(256,1/4) + DeviceHeads:
   - refold(): 0 PCIe bytes, exact fp32/bf16 parity across all 714 weights vs fresh host build, warm wall time
   - tape_backward(): 0 PCIe bytes, exact parity (<1e-4) across all leaf grads vs host Folds.backward, warm wall time
3. DeviceOptimizer on exact 558 trained RFD4 params:
   - Zero-grad decay rule, reduce_fn hook, EMA 0.999 shadows, warm step wall times for split2/split2_hifi4/fp32.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np


def _aiclk_sampler(node: int):
    clk, stop = [], threading.Event()
    path = f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk"

    def run():
        while not stop.is_set():
            try:
                clk.append(int(open(path).read().split()[0]))
            except Exception:
                pass
            time.sleep(0.05)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return clk, stop, t


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, np.float64).ravel()
    b64 = np.asarray(b, np.float64).ravel()
    den = float(np.linalg.norm(b64))
    num = float(np.linalg.norm(a64 - b64))
    return num if den == 0.0 else num / den


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, np.float64).ravel()
    b64 = np.asarray(b, np.float64).ravel()
    na, nb = float(np.linalg.norm(a64)), float(np.linalg.norm(b64))
    if na == 0.0 and nb == 0.0:
        return 1.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a64, b64) / (na * nb))


def _load_npz_batch(path: str) -> dict:
    import torch
    z = np.load(path, allow_pickle=False)
    batch = {"x_1": {}, "x_t": {}, "t": {}}
    for k in z.files:
        v = z[k]
        if k.startswith("in/"):
            rest = k[3:]
            parts = rest.split("/")
            if len(parts) == 2 and parts[0] in ("x_1", "x_t", "t"):
                batch[parts[0]][parts[1]] = torch.from_numpy(v.copy())
            elif len(parts) == 1:
                batch[parts[0]] = torch.from_numpy(v.copy()) if v.ndim > 0 else v.item()
    return batch


def test_u32_smoke(dev):
    """Verify bit-exact fp32 helpers using uint32 bitcast on ttnn==0.68.0."""
    import torch
    import ttnn
    from tt_bio.train.folds import (
        _dev_clone_f32, _dev_concat_f32, _dev_expand_f32,
        _dev_mul_f32, _dev_reduce_to_f32, _dev_reshape_f32, _dev_slice_f32,
    )

    print("\n=== [0/3] Bit-exact uint32 fp32 DeviceFolds op smoke ===", flush=True)
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((64, 64)).astype(np.float32)
    x_tt = ttnn.from_torch(torch.from_numpy(x_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

    # 1. clone
    c_tt = _dev_clone_f32(x_tt)
    c_np = ttnn.to_torch(c_tt).numpy()
    assert np.array_equal(c_np, x_np), f"clone bit mismatch: max={np.max(np.abs(c_np - x_np))}"
    ttnn.deallocate(c_tt)

    # 2. aligned slice + concat
    s0 = _dev_slice_f32(x_tt, 1, 0, 32)
    s1 = _dev_slice_f32(x_tt, 1, 32, 64)
    cat_tt = _dev_concat_f32([s0, s1], 1)
    cat_np = ttnn.to_torch(cat_tt).numpy()
    assert np.array_equal(cat_np, x_np), f"aligned slice/concat mismatch: max={np.max(np.abs(cat_np - x_np))}"
    for t in (s0, s1, cat_tt, x_tt):
        ttnn.deallocate(t)

    # 3. unaligned slice/concat (384, 12) + (384, 12) -> (384, 24) (bias_ws)
    a_np = rng.standard_normal((384, 12)).astype(np.float32)
    b_np = rng.standard_normal((384, 12)).astype(np.float32)
    a_tt = ttnn.from_torch(torch.from_numpy(a_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    b_tt = ttnn.from_torch(torch.from_numpy(b_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    ab_tt = _dev_concat_f32([a_tt, b_tt], 1)
    ab_np = ttnn.to_torch(ab_tt).numpy()
    assert np.array_equal(ab_np, np.concatenate([a_np, b_np], axis=1)), (
        f"unaligned concat mismatch: max={np.max(np.abs(ab_np - np.concatenate([a_np, b_np], axis=1)))}"
    )
    ab_s0 = _dev_slice_f32(ab_tt, 1, 0, 12)
    ab_s1 = _dev_slice_f32(ab_tt, 1, 12, 24)
    assert np.array_equal(ttnn.to_torch(ab_s0).numpy(), a_np), "unaligned slice 0 mismatch"
    assert np.array_equal(ttnn.to_torch(ab_s1).numpy(), b_np), "unaligned slice 1 mismatch"

    # 4. width-1 reshape (384,) -> (384, 1) and broadcast mul (384, 12) * (384, 1) + reduce back to (384, 1)
    w_np = rng.standard_normal((384,)).astype(np.float32)
    w_tt = ttnn.from_torch(torch.from_numpy(w_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    w_col = _dev_reshape_f32(w_tt, (384, 1))
    assert np.array_equal(ttnn.to_torch(w_col).numpy(), w_np[:, None]), "width-1 reshape mismatch"
    mul_tt = _dev_mul_f32(a_tt, w_col, (384, 12))
    mul_rel = _rel_l2(ttnn.to_torch(mul_tt).numpy(), a_np * w_np[:, None])
    assert mul_rel < 1e-6, f"broadcast mul (384,12)*(384,1) mismatch: {mul_rel}"
    red_tt = _dev_reduce_to_f32(mul_tt, (384, 1))
    red_rel = _rel_l2(ttnn.to_torch(red_tt).numpy(), (a_np * w_np[:, None]).sum(axis=1, keepdims=True))
    assert red_rel < 1e-6, f"reduce_to (384,1) mismatch: {red_rel}"

    # 5. expand (384, 1) -> (384, 12)
    exp_tt = _dev_expand_f32(w_col, (384, 12))
    assert np.array_equal(ttnn.to_torch(exp_tt).numpy(), np.broadcast_to(w_np[:, None], (384, 12))), "expand mismatch"

    # 6. fp32 square vs multiply(a, a) precision check
    sq_tt = ttnn.square(a_tt)
    mm_tt = ttnn.multiply(a_tt, a_tt)
    sq_rel = _rel_l2(ttnn.to_torch(sq_tt).numpy(), a_np * a_np)
    mm_rel = _rel_l2(ttnn.to_torch(mm_tt).numpy(), a_np * a_np)
    ttnn.deallocate(sq_tt)
    ttnn.deallocate(mm_tt)
    assert sq_rel < 1e-3, f"ttnn.square(f32) rel_l2={sq_rel:.2e} >= 1e-3"

    for t in (a_tt, b_tt, ab_tt, ab_s0, ab_s1, w_tt, w_col, mul_tt, red_tt, exp_tt):
        ttnn.deallocate(t)
    print(f"  u32 fp32 smoke PASS (mul_rel={mul_rel:.2e}, red_rel={red_rel:.2e}, sq_rel={sq_rel:.2e}, mm_f32_rel={mm_rel:.2e})", flush=True)
    return {"mul_rel": mul_rel, "red_rel": red_rel, "sq_rel": sq_rel, "mm_f32_rel": mm_rel}


def test_heads(dev, glue, batch8_path: str, pcie_meter_mod):
    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.rfd4.heads import DeviceHeads
    from tt_bio.train.rfd4_heads import device_insertion_loss
    from tt_bio.train import rfd4_objective as OBJ

    print("\n=== [1/3] DeviceHeads + device_insertion_loss (batch08, all 5 terms) ===", flush=True)
    batch = _load_npz_batch(batch8_path)
    b1 = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            b1[k] = {sk: sv[:1] for sk, sv in v.items()}
        elif isinstance(v, torch.Tensor):
            b1[k] = v[:1]
        else:
            b1[k] = v
    b1["exploration_k"] = 1
    b, n = b1["padding_mask"].shape
    n_pad = ((n + 31) // 32) * 32
    print(f"  batch08 b={b}, n={n}, n_pad={n_pad}, ins_rows={(b1['insertion_source_idx'] >= 0).sum().item()}", flush=True)

    rng = np.random.default_rng(42)
    s_np = (rng.standard_normal((b, n, 768)).astype(np.float32) * 0.5)
    s_np *= b1["padding_mask"].numpy()[..., None].astype(np.float32)
    s_pad_np = np.zeros((b, n_pad, 768), dtype=np.float32)
    s_pad_np[:, :n, :] = s_np

    def host_f64_loss_and_grads(s64_t, head_w64):
        from tt_bio._vendor.rfd4_prep._cif.model_masks import compute_flow_emit_masks
        emit = compute_flow_emit_masks(b1)

        def rms_norm(x, w):
            inv = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-5)
            return x * inv * w

        eca = emit["bb_ca"].to(torch.float64)[..., None]
        ell = emit["local_latents"].to(torch.float64)[..., None]
        pm = b1["padding_mask"].to(torch.float64)[..., None]

        v_ca = torch.matmul(rms_norm(s64_t, head_w64["ca_norm_w"]), head_w64["ca_w"]) * eca
        v_ll = torch.matmul(rms_norm(s64_t, head_w64["ll_norm_w"]), head_w64["ll_w"]) * ell
        v_sq = torch.matmul(rms_norm(s64_t, head_w64["seq_norm_w"]), head_w64["seq_w"]) * pm
        q = torch.matmul(s64_t, head_w64["ins_q_w"])
        k = torch.matmul(s64_t, head_w64["ins_k_w"])
        v_ins = torch.bmm(q, k.transpose(1, 2)) * (768.0 ** -0.5)

        outs_np = {
            "v_bb_ca": v_ca.detach().numpy(),
            "v_local_latents": v_ll.detach().numpy(),
            "sequence_logits": v_sq.detach().numpy(),
            "insertion_logits": v_ins.detach().numpy(),
        }
        tot, brk, seeds = OBJ.rfd4_loss(b1, outs_np, OBJ.WEIGHTS)
        loss_proxy = (
            (v_ca * torch.as_tensor(seeds["v_bb_ca"])).sum()
            + (v_ll * torch.as_tensor(seeds["v_local_latents"])).sum()
            + (v_sq * torch.as_tensor(seeds["sequence_logits"])).sum()
            + (v_ins * torch.as_tensor(seeds["insertion_logits"])).sum()
        )
        grads = torch.autograd.grad(loss_proxy, [s64_t] + list(head_w64.values()))
        gmap = {"s": grads[0].numpy()}
        for nm, gv in zip(head_w64.keys(), grads[1:]):
            gmap[nm] = gv.numpy()
        return tot, brk, outs_np, seeds, gmap

    head_w64 = {
        "ca_norm_w": glue.ca_linear[0].weight.detach().to(torch.float64).clone().requires_grad_(True),
        "ca_w": glue.ca_linear[1].weight.detach().t().to(torch.float64).clone().requires_grad_(True),
        "ll_norm_w": glue.local_latents_linear[0].weight.detach().to(torch.float64).clone().requires_grad_(True),
        "ll_w": glue.local_latents_linear[1].weight.detach().t().to(torch.float64).clone().requires_grad_(True),
        "seq_norm_w": glue.sequence_readout[0].weight.detach().to(torch.float64).clone().requires_grad_(True),
        "seq_w": glue.sequence_readout[1].weight.detach().t().to(torch.float64).clone().requires_grad_(True),
        "ins_q_w": glue.insertion_q_proj.weight.detach().t().to(torch.float64).clone().requires_grad_(True),
        "ins_k_w": glue.insertion_k_proj.weight.detach().t().to(torch.float64).clone().requires_grad_(True),
    }
    s64 = torch.from_numpy(s_np.astype(np.float64)).requires_grad_(True)
    tot64, brk64, outs64, seeds64, g64 = host_f64_loss_and_grads(s64, head_w64)

    fd_errs = []
    for key_fd in ("s", "ca_w", "ins_q_w"):
        eps = 1e-6
        t_orig = s64 if key_fd == "s" else head_w64[key_fd]
        d = rng.standard_normal(t_orig.shape)
        d = d / np.linalg.norm(d)
        dt = torch.from_numpy(d)

        def eval_at(delta):
            s_e = (s64.detach() + (delta * dt if key_fd == "s" else 0.0)).requires_grad_(True)
            w_e = {
                k: (v.detach() + (delta * dt if k == key_fd else 0.0)).requires_grad_(True)
                for k, v in head_w64.items()
            }
            tot_e, _, _, _, _ = host_f64_loss_and_grads(s_e, w_e)
            return tot_e

        fd_val = (eval_at(+eps) - eval_at(-eps)) / (2.0 * eps)
        an_val = float(np.sum(g64[key_fd] * d))
        fd_err = abs(fd_val - an_val) / max(abs(an_val), 1e-12)
        fd_errs.append((key_fd, fd_err))
    max_fd = max(e for _, e in fd_errs)
    print(f"  FD check on float64 ref: {fd_errs} (max={max_fd:.2e})", flush=True)
    assert max_fd < 2e-6, f"FD check failed: {fd_errs}"

    dheads = DeviceHeads(glue, dev)
    glue._device_heads = dheads
    s_tt = ttnn.from_torch(torch.from_numpy(s_pad_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

    with ag.no_grad():
        out_aa1 = glue.heads(s_tt, b1, insertion=True)
        out_aa2 = glue.heads(s_tt, b1, insertion=True)
    aa_diffs = {}
    for mod_k, sub_k, ref_k in (
        ("bb_ca", "v", "v_bb_ca"),
        ("local_latents", "v", "v_local_latents"),
        ("sequence", "logits", "sequence_logits"),
        ("insertion", "logits", "insertion_logits"),
    ):
        a1 = ttnn.to_torch(out_aa1[mod_k][sub_k]).numpy()[:, :n, ...]
        a2 = ttnn.to_torch(out_aa2[mod_k][sub_k]).numpy()[:, :n, ...]
        if mod_k == "insertion":
            a1 = a1[:, :n, :n]
            a2 = a2[:, :n, :n]
        aa_diffs[ref_k] = (float(np.max(np.abs(a1 - a2))), _rel_l2(a1, outs64[ref_k]))
        ttnn.deallocate(out_aa1[mod_k][sub_k])
        ttnn.deallocate(out_aa2[mod_k][sub_k])
    print(f"  Grad-off A/A max_abs & fwd rel_l2 vs f64: {aa_diffs}", flush=True)
    assert all(d[0] == 0.0 for d in aa_diffs.values()), f"A/A not bit-identical: {aa_diffs}"
    assert all(d[1] < 1e-2 for d in aa_diffs.values()), f"Fwd parity failed: {aa_diffs}"

    def run_taped_heads(broken_mode: str = "none"):
        ag.forget_parameters()
        head_handles = {
            "ca_norm_w": dheads.ca_norm_w, "ca_w": dheads.ca_w,
            "ll_norm_w": dheads.ll_norm_w, "ll_w": dheads.ll_w,
            "seq_norm_w": dheads.seq_norm_w, "seq_w": dheads.seq_w,
            "ins_q_w": dheads.ins_q_w, "ins_k_w": dheads.ins_k_w,
        }
        p_leaves = {k: ag.parameter(h) for k, h in head_handles.items()}
        for pl in p_leaves.values():
            pl.grad = None

        if pcie_meter_mod is not None:
            pcie_meter_mod.take()
            pcie_meter_mod.phase("heads_taped")

        s_in = ag.Tensor(s_tt, requires_grad=True)
        with ag.tape():
            o = glue.heads(s_in, b1, insertion=True)
        ca_t = next(iter(o["bb_ca"].values()))
        ll_t = next(iter(o["local_latents"].values()))
        seq_t = o["sequence"]["logits"]
        ins_t = o["insertion"]["logits"]

        ok = {
            "v_bb_ca": ttnn.to_torch(ca_t.value).float().numpy()[:, :n, :3].astype(np.float64),
            "v_local_latents": ttnn.to_torch(ll_t.value).float().numpy()[:, :n, :32].astype(np.float64),
            "sequence_logits": ttnn.to_torch(seq_t.value).float().numpy()[:, :n, :147].astype(np.float64),
        }
        tot_non_ins, _bd, seeds = OBJ.rfd4_loss(b1, ok, OBJ.WEIGHTS)
        ins_w = float(OBJ.WEIGHTS["insertion_prediction"])
        ins_loss_np, dz_ins = device_insertion_loss(ins_t, b1, weight=ins_w)
        if broken_mode == "gross":
            ttnn.deallocate(dz_ins)
            dz_ins = None
        elif broken_mode == "subtle_110":
            dz_old = dz_ins
            dz_ins = ttnn.multiply(dz_old, 1.10)
            ttnn.deallocate(dz_old)

        def up_seed(arr, d_out):
            buf = np.zeros((b, n_pad, d_out), dtype=np.float32)
            buf[:, :n, :d_out] = np.asarray(arr, dtype=np.float32)
            return ttnn.from_torch(torch.from_numpy(buf), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

        outs_bwd = [ca_t, ll_t, seq_t]
        seeds_bwd = [up_seed(seeds["v_bb_ca"], 3), up_seed(seeds["v_local_latents"], 32), up_seed(seeds["sequence_logits"], 147)]
        if dz_ins is not None:
            outs_bwd.append(ins_t)
            seeds_bwd.append(dz_ins)
        ag.backward(outs_bwd, seeds_bwd)
        for sd in seeds_bwd:
            ttnn.deallocate(sd)

        pcie_snap = pcie_meter_mod.take() if pcie_meter_mod is not None else None

        res_g = {}
        res_g["s"] = ttnn.to_torch(s_in.grad).float().numpy()[:, :n, :]
        ttnn.deallocate(s_in.grad)
        for k, pl in p_leaves.items():
            if pl.grad is None:
                res_g[k] = np.zeros_like(g64[k], dtype=np.float32)
            else:
                res_g[k] = ttnn.to_torch(pl.grad).float().numpy().reshape(g64[k].shape)
                ttnn.deallocate(pl.grad)
                pl.grad = None
        ag.release_pins()
        ag.forget_parameters()

        rels = {k: _rel_l2(res_g[k], g64[k]) for k in g64}
        coss = {k: _cos(res_g[k], g64[k]) for k in g64}
        return tot_non_ins + ins_w * float(ins_loss_np[0]), rels, coss, pcie_snap

    tot_dev, rels_ok, coss_ok, pcie_heads = run_taped_heads("none")
    worst_rel = max(rels_ok.values())
    min_cos = min(coss_ok.values())
    loss_rel = abs(tot_dev - tot64) / max(abs(tot64), 1e-12)
    print(f"  Good arm: loss_rel={loss_rel:.3e}, worst_rel={worst_rel:.3e}, min_cos={min_cos:.6f}", flush=True)
    for k in sorted(rels_ok):
        print(f"    {k:12s}: rel_l2={rels_ok[k]:.3e}  cos={coss_ok[k]:.6f}", flush=True)
    print(f"  Heads PCIe: {pcie_heads}", flush=True)

    _, rels_gross, _, _ = run_taped_heads("gross")
    _, rels_sub, _, _ = run_taped_heads("subtle_110")
    gross_worst = max(rels_gross.values())
    sub_worst = max(rels_sub.values())
    print(f"  Broken arm (gross drop insertion): worst_rel={gross_worst:.3e} (>1e-2 FAIL)", flush=True)
    print(f"  Broken arm (subtle x1.10 ins grad): worst_rel={sub_worst:.3e} (>1e-2 FAIL)", flush=True)

    dheads.deallocate()
    glue._device_heads = None
    ttnn.deallocate(s_tt)

    assert worst_rel <= 1e-2 and min_cos >= 0.9999, f"Heads VJP D28 failed: worst_rel={worst_rel}, min_cos={min_cos}"
    assert gross_worst > 1e-2 and sub_worst > 1e-2, f"Broken arms failed to fail: {gross_worst}, {sub_worst}"
    return {
        "fd_max": max_fd, "aa_max_abs": max(d[0] for d in aa_diffs.values()),
        "loss_rel": loss_rel, "worst_rel": worst_rel, "min_cos": min_cos,
        "broken_gross": gross_worst, "broken_subtle_110": sub_worst, "pcie": pcie_heads,
    }


def test_device_folds(dev, state, cfg, glue, pcie_meter_mod):
    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4.heads import DeviceHeads
    from tt_bio.train.folds import DeviceFolds, Folds

    print("\n=== [2/3] DeviceFolds (full RFD4 Trunk + prewarm + DeviceHeads, 0 PCIe) ===", flush=True)
    full_state = dict(state)
    for nm, p in glue.named_parameters():
        full_state[f"glue.{nm}"] = p.detach()

    def find_trimuls(root):
        seen, out, todo = set(), [], [root]
        while todo:
            o = todo.pop()
            if id(o) in seen:
                continue
            seen.add(id(o))
            if hasattr(o, "prewarm") and hasattr(o, "width_variants"):
                out.append(o)
            if isinstance(o, (list, tuple)):
                todo.extend(o)
            elif isinstance(o, dict):
                todo.extend(o.values())
            elif hasattr(o, "__dict__") and not isinstance(o, (torch.Tensor, ttnn.Tensor, type)):
                todo.extend(v for v in vars(o).values() if not isinstance(v, (torch.Tensor, ttnn.Tensor)))
        return out

    dfolds = DeviceFolds(full_state, dev)
    with dfolds.recording() as leaves:
        trunk = E.load_trunk(glue, leaves, cfg, training=True, attn_fp32=True)
        dheads = DeviceHeads(glue, dev, state=leaves)
        for m in find_trimuls(trunk):
            m.prewarm(256, 1)
            m.prewarm(256, 4)
    n_bound = dfolds.bind(trunk, dheads)
    unreached = set(dfolds.unreached())
    used_leaf_names = [nm for nm in dfolds.leaves if nm not in unreached]
    print(f"  Bound {n_bound} specs, {len(dfolds.nodes)} IR nodes, {len(used_leaf_names)} used leaves", flush=True)

    rng = np.random.default_rng(99)
    perturbed_host = {}
    dev_masters = {}
    for nm in used_leaf_names:
        base = full_state[nm].detach().float().contiguous()
        delta = torch.from_numpy((rng.standard_normal(base.shape) * 0.02).astype(np.float32))
        w_new = (base + delta).contiguous()
        perturbed_host[nm] = w_new
        dev_masters[nm] = ttnn.from_torch(w_new, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

    # Warmup refold x2, then timed refold under pcie_meter
    for _ in range(2):
        dfolds.refold(dev_masters)
    ttnn.synchronize_device(dev)
    if pcie_meter_mod is not None:
        pcie_meter_mod.take()
        pcie_meter_mod.phase("refold")
    t0 = time.time()
    dfolds.refold(dev_masters)
    ttnn.synchronize_device(dev)
    refold_s = time.time() - t0
    refold_pcie = pcie_meter_mod.take() if pcie_meter_mod is not None else None
    print(f"  refold() warm time={refold_s*1000:.1f} ms, pcie={refold_pcie}", flush=True)

    host_ref_state = dict(full_state)
    host_ref_state.update(perturbed_host)
    hfolds = Folds(host_ref_state)
    with hfolds.recording() as hleaves:
        htrunk = E.load_trunk(glue, hleaves, cfg, training=True, attn_fp32=True)
        hdheads = DeviceHeads(glue, dev, state=hleaves)
        for m in find_trimuls(htrunk):
            m.prewarm(256, 1)
            m.prewarm(256, 4)

    h_handles = [h for h, _ in hfolds.sources.values()]
    assert len(h_handles) == len(dfolds.specs), (len(h_handles), len(dfolds.specs))
    worst_by_dt = {}
    for spec, hh in zip(dfolds.specs, h_handles):
        a = ttnn.to_torch(spec.handle).float().numpy()
        b = ttnn.to_torch(hh).float().numpy()
        rel = _rel_l2(a, b)
        dt = str(spec.dtype)
        worst_by_dt[dt] = max(worst_by_dt.get(dt, 0.0), rel)
    print(f"  refold() max rel_l2 vs fresh host build ({len(dfolds.specs)} weights): {worst_by_dt}", flush=True)

    # Warmup tape_backward x1, then timed parity pass
    def seed_all_grads():
        pairs = []
        for spec, hh in zip(dfolds.specs, h_handles):
            g_np = (rng.standard_normal(spec.shape) * 0.1).astype(np.float32)
            g_t = torch.from_numpy(g_np)
            g_tt = ttnn.from_torch(g_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            leaf = ag.parameter_for(spec.handle)
            assert leaf is not None
            leaf.grad = g_tt
            pairs.append((hh, g_t))
        return pairs

    seed_all_grads()
    warm_g, _ = dfolds.tape_backward(dev_masters)
    for t in warm_g.values():
        ttnn.deallocate(t)

    host_grad_pairs = seed_all_grads()
    for p in hfolds.leaves.values():
        p.grad = None
    hfolds.backward(host_grad_pairs)
    host_leaf_grads = {
        nm: hfolds.leaves[nm].grad.detach().numpy().copy()
        for nm in used_leaf_names if hfolds.leaves[nm].grad is not None
    }

    ttnn.synchronize_device(dev)
    if pcie_meter_mod is not None:
        pcie_meter_mod.take()
        pcie_meter_mod.phase("fold_bwd")
    t1 = time.time()
    dev_grads, idle = dfolds.tape_backward(dev_masters)
    ttnn.synchronize_device(dev)
    bwd_s = time.time() - t1
    bwd_pcie = pcie_meter_mod.take() if pcie_meter_mod is not None else None
    print(f"  tape_backward() warm time={bwd_s*1000:.1f} ms, grads={len(dev_grads)}, idle={len(idle)}, pcie={bwd_pcie}", flush=True)

    assert set(dev_grads.keys()) == set(host_leaf_grads.keys()), (
        set(dev_grads.keys()) ^ set(host_leaf_grads.keys()))
    worst_g_rel, worst_g_name = 0.0, ""
    for nm, g_tt in dev_grads.items():
        g_dev = ttnn.to_torch(g_tt).float().numpy().reshape(host_leaf_grads[nm].shape)
        rel = _rel_l2(g_dev, host_leaf_grads[nm])
        if rel > worst_g_rel:
            worst_g_rel, worst_g_name = rel, nm
        ttnn.deallocate(g_tt)
    print(f"  tape_backward() max rel_l2 across {len(host_leaf_grads)} leaves: {worst_g_rel:.3e} ({worst_g_name})", flush=True)

    for spec in dfolds.specs:
        ttnn.deallocate(spec.handle)
    for h in h_handles:
        ttnn.deallocate(h)
    for c in list(dfolds.consts.values()) + list(hfolds.consts.values()):
        ttnn.deallocate(c)
    for m in dev_masters.values():
        ttnn.deallocate(m)
    ag.release_pins()
    ag.forget_parameters()

    if refold_pcie:
        tot_r = sum(v.get("up_n", 0) + v.get("down_n", 0) for v in refold_pcie.values() if isinstance(v, dict))
        assert tot_r == 0, f"refold had non-zero PCIe bytes: {refold_pcie}"
    if bwd_pcie:
        tot_b = sum(v.get("up_n", 0) + v.get("down_n", 0) for v in bwd_pcie.values() if isinstance(v, dict))
        assert tot_b == 0, f"fold_bwd had non-zero PCIe bytes: {bwd_pcie}"
    assert worst_g_rel < 1e-4, f"DeviceFolds tape_backward parity failed: {worst_g_rel} ({worst_g_name})"
    assert worst_by_dt.get("DataType.FLOAT32", 0.0) < 1e-6, f"fp32 refold mismatch: {worst_by_dt}"
    assert worst_by_dt.get("DataType.BFLOAT16", 0.0) < 1e-2, f"bf16 refold mismatch: {worst_by_dt}"
    return {
        "n_specs": n_bound, "refold_ms": round(refold_s * 1000, 2),
        "fold_bwd_ms": round(bwd_s * 1000, 2), "refold_worst_by_dtype": worst_by_dt,
        "fold_bwd_worst_rel": worst_g_rel, "fold_bwd_worst_name": worst_g_name,
    }


def test_device_optim(dev, state, glue, pcie_meter_mod):
    import torch
    import ttnn
    from tt_bio.train.device_optim import rfd4_device_optimizer

    print("\n=== [3/3] DeviceOptimizer (exact 558 RFD4 params, warm timing + 0-pair PCIe) ===", flush=True)
    # Match rfd4_run.py exact 558 params (trunk nn.* excluding glue-owned + all glue.*)
    glue_keys = {f"nn.{k}" for k, _ in glue.named_parameters()}
    params = {k: v.detach().float() for k, v in state.items() if k.startswith("nn.") and k not in glue_keys}
    for k, p in glue.named_parameters():
        if p.requires_grad:
            params[f"glue.{k}"] = p.detach().float()
    print(f"  Parameter count: {len(params)} ({sum(t.numel() for t in params.values())} elems)", flush=True)

    rng = np.random.default_rng(123)
    host_grads = {
        nm: torch.from_numpy((rng.standard_normal(t.shape) * 0.05).astype(np.float32))
        for nm, t in params.items()
    }
    omitted_adam = next(nm for nm, t in params.items() if t.ndim < 2)
    omitted_muon = next(nm for nm, t in params.items() if t.ndim == 2 and nm.startswith("nn."))

    timings = {}
    for ns_mode in ("split2", "split2_hifi4", "fp32"):
        opt = rfd4_device_optimizer(params, dev, lr=1e-4, ema_decay=0.999, ns_mode=ns_mode)
        reduced_count = [0]

        def _red(g):
            reduced_count[0] += 1
            return g

        step_ms = []
        pcie_step = None
        for it in range(4):
            dev_grads = {
                nm: ttnn.from_torch(g, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
                for nm, g in host_grads.items()
                if nm not in (omitted_adam, omitted_muon)
            }
            ttnn.synchronize_device(dev)
            if it == 3 and pcie_meter_mod is not None:
                pcie_meter_mod.take()
                pcie_meter_mod.phase("opt")
            t0 = time.time()
            st = opt.step(dev_grads, reduce_fn=(_red if (ns_mode == "split2" and it == 0) else None), free_grads=True)
            ttnn.synchronize_device(dev)
            step_ms.append((time.time() - t0) * 1000.0)
            if it == 3 and pcie_meter_mod is not None:
                pcie_step = pcie_meter_mod.take()

        timings[ns_mode] = {
            "cold_ms": round(step_ms[0], 1),
            "warm_ms": round(float(np.median(step_ms[2:])), 1),
            "muon_norm": st["grad_norm_muon"], "adam_norm": st["grad_norm"],
            "n_muon": len(opt.muon_names), "n_adam": len(opt.adam_names),
            "n_muon_buckets": len(opt._muon_buckets), "n_adam_buckets": len(opt._adam_buckets),
            "pcie": pcie_step,
        }
        if ns_mode == "split2":
            assert reduced_count[0] == 1, f"expected reduce_fn called once, got {reduced_count[0]}"
            hm = opt.host_master()
            om_a_diff = float(np.max(np.abs(hm[omitted_adam] - params[omitted_adam].numpy())))
            om_m_diff = float(np.max(np.abs(hm[omitted_muon] - params[omitted_muon].numpy())))
            assert om_a_diff == 0.0, f"Omitted Adam param moved: {om_a_diff}"
            assert om_m_diff == 0.0, f"Omitted Muon param moved: {om_m_diff}"
            assert opt.ema is not None and opt.ema.n_averaged == 4 and len(opt.ema.host_shadow()) == len(params)
        print(
            f"  {ns_mode:14s}: cold={timings[ns_mode]['cold_ms']} ms, warm={timings[ns_mode]['warm_ms']} ms "
            f"(muon={len(opt.muon_names)} in {len(opt._muon_buckets)} bkts, "
            f"adam={len(opt.adam_names)} in {len(opt._adam_buckets)} bkts, "
            f"norms={st['grad_norm_muon']:.3f}/{st['grad_norm']:.3f}, pcie={pcie_step})",
            flush=True,
        )
        opt.deallocate()

    return timings


def test_resident_pair(dev, glue, batch8_path: str, pcie_meter_mod):
    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.rfd4.pair import PairSplit
    from tt_bio.train.bridge import Bridge
    from tt_bio.train.rfd4_pair import ResidentPairBuilder

    print("\n=== [4/4] ResidentPairBuilder (zero pair-sized PCIe, fwd + D28 VJP parity) ===", flush=True)
    bt = _load_npz_batch(batch8_path)
    b1 = {
        k: ({sk: sv[:1] for sk, sv in v.items()} if isinstance(v, dict) else (v[:1] if torch.is_tensor(v) else v))
        for k, v in bt.items()
    }
    n = int(bt["padding_mask"].shape[1])
    c = int(glue.pair_repr_builder.init_repr_factory.linear_out.weight.shape[0])

    # Float64 reference on winner=0
    ps = PairSplit(glue.pair_repr_builder)
    fac = glue.pair_repr_builder.init_repr_factory
    feats64 = ps.features(b1).double()  # [1, n, n, 107]
    W64 = fac.linear_out.weight.detach().double().clone().requires_grad_(True)
    lnw64 = fac.ln_out.weight.detach().double().clone().requires_grad_(True)
    g0, b0 = ps.modulation(b1)
    gam64 = g0.detach().double().clone().requires_grad_(True)
    bet64 = b0.detach().double().clone().requires_grad_(True)
    e1, e2 = float(fac.ln_out.eps), float(glue.pair_repr_builder.adaln.norm.eps)

    x64 = feats64 @ W64.t()
    n1_64 = x64 * torch.rsqrt((x64 * x64).mean(dim=-1, keepdim=True) + e1)
    a64 = n1_64 * lnw64
    bb64 = a64 * torch.rsqrt((a64 * a64).mean(dim=-1, keepdim=True) + e2)
    z64 = bb64 * gam64[:, None, None, :] + bet64[:, None, None, :]

    rng = np.random.default_rng(77)
    dz_np = (rng.standard_normal((1, n, n, c)) * 0.05).astype(np.float32)
    dz64 = torch.from_numpy(dz_np.astype(np.float64))
    gW64, glnw64, ggam64, gbet64 = torch.autograd.grad((z64 * dz64).sum(), [W64, lnw64, gam64, bet64])
    ref_g = {
        "W": gW64.numpy(), "lnw": glnw64.numpy(),
        "gamma": ggam64.numpy(), "beta": gbet64.numpy(),
    }

    rpb = ResidentPairBuilder(glue.pair_repr_builder, dev)
    enc = rpb.encode_step(bt)
    assert enc.fallback_const is None, "expected compact integer encoding on batch08"

    if pcie_meter_mod is not None:
        pcie_meter_mod.take()
        pcie_meter_mod.phase("res_pair_cand")

    t0 = time.time()
    zk, cache = rpb.candidates(enc, dtype=ttnn.float32)
    pcie_cand = pcie_meter_mod.take() if pcie_meter_mod is not None else None
    zk0_np = ttnn.to_torch(zk).float().numpy()[:1]
    ttnn.deallocate(zk)
    cand_fwd_rel = _rel_l2(zk0_np, z64.detach().numpy())

    # Taped winner pass + backward
    for p in glue.pair_repr_builder.parameters():
        p.grad = None
    br = Bridge(dev, ttnn.float32)
    dz_tt = ttnn.from_torch(torch.from_numpy(dz_np), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    if pcie_meter_mod is not None:
        pcie_meter_mod.take()
        pcie_meter_mod.phase("res_pair_win_fwd")
    with ag.tape():
        gam_w, bet_w = rpb.modulation(b1)
        z_t = rpb.winner_taped(cache, 0, gam_w, bet_w, br, dtype=ttnn.float32)
    pcie_win_fwd = pcie_meter_mod.take() if pcie_meter_mod is not None else None
    zw_np = ttnn.to_torch(z_t.value).float().numpy()
    win_fwd_rel = _rel_l2(zw_np, z64.detach().numpy())
    if pcie_meter_mod is not None:
        pcie_meter_mod.take()
        pcie_meter_mod.phase("res_pair_win_bwd")
    ag.backward([z_t], [dz_tt])
    ttnn.synchronize_device(dev)
    pcie_win_bwd = pcie_meter_mod.take() if pcie_meter_mod is not None else None
    pair_ms = (time.time() - t0) * 1000.0
    ttnn.deallocate(dz_tt)
    pcie_snap = {"cand": pcie_cand, "win_fwd": pcie_win_fwd, "win_bwd": pcie_win_bwd}

    # Extract device gradients from bridge.ups
    leaf_names = ["W", "lnw", "gamma", "beta"]
    dev_g = {}
    for nm, (h_t, d_t) in zip(leaf_names, br.ups):
        assert d_t.grad is not None, f"missing grad for {nm}"
        dev_g[nm] = ttnn.to_torch(d_t.grad).float().numpy().reshape(ref_g[nm].shape)
        ttnn.deallocate(d_t.grad)
        d_t.grad = None
    br.ups.clear()
    ag.release_pins()
    # Broken arm: perturb upstream dz by 1.10x -> VJP parity must FAIL > 1e-2
    enc_brk = rpb.encode_step(bt)
    zk_brk, cache_brk = rpb.candidates(enc_brk, dtype=ttnn.float32)
    ttnn.deallocate(zk_brk)
    br_brk = Bridge(dev, ttnn.float32)
    dz_brk = ttnn.from_torch(torch.from_numpy((dz_np * 1.10).astype(np.float32)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    with ag.tape():
        gam_b, bet_b = rpb.modulation(b1)
        z_b = rpb.winner_taped(cache_brk, 0, gam_b, bet_b, br_brk, dtype=ttnn.float32)
    ag.backward([z_b], [dz_brk])
    ttnn.synchronize_device(dev)
    ttnn.deallocate(dz_brk)
    brk_rels = {}
    for nm, (_, d_t) in zip(leaf_names, br_brk.ups):
        brk_rels[nm] = _rel_l2(ttnn.to_torch(d_t.grad).float().numpy().reshape(ref_g[nm].shape), ref_g[nm])
        ttnn.deallocate(d_t.grad)
        d_t.grad = None
    br_brk.ups.clear()
    ag.release_pins()
    brk_worst = max(brk_rels.values())

    # Warm timing (3 bf16 iterations, matching production step)
    warm_enc, warm_cand, warm_wfwd, warm_wbwd, warm_tot = [], [], [], [], []
    for it in range(4):
        t_a = time.time()
        enc_w = rpb.encode_step(bt)
        t_b = time.time()
        zk_w, cache_w = rpb.candidates(enc_w, dtype=ttnn.bfloat16)
        ttnn.synchronize_device(dev)
        t_c = time.time()
        ttnn.deallocate(zk_w)
        br_w = Bridge(dev, ttnn.float32)
        dz_w = ttnn.from_torch(torch.from_numpy(dz_np), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        ttnn.synchronize_device(dev)
        t_d = time.time()
        with ag.tape():
            gam_ww, bet_ww = rpb.modulation(b1)
            z_w = rpb.winner_taped(cache_w, 0, gam_ww, bet_ww, br_w, dtype=ttnn.bfloat16)
        ttnn.synchronize_device(dev)
        t_e = time.time()
        ag.backward([z_w], [dz_w])
        ttnn.synchronize_device(dev)
        t_f = time.time()
        ttnn.deallocate(dz_w)
        for _, d_t in br_w.ups:
            if d_t.grad is not None:
                ttnn.deallocate(d_t.grad)
                d_t.grad = None
        br_w.ups.clear()
        ag.release_pins()
        if it >= 1:
            warm_enc.append((t_b - t_a) * 1e3)
            warm_cand.append((t_c - t_b) * 1e3)
            warm_wfwd.append((t_e - t_d) * 1e3)
            warm_wbwd.append((t_f - t_e) * 1e3)
            warm_tot.append(((t_c - t_a) + (t_f - t_d)) * 1e3)

    rpb.deallocate()

    rels = {nm: _rel_l2(dev_g[nm], ref_g[nm]) for nm in leaf_names}
    coss = {nm: _cos(dev_g[nm], ref_g[nm]) for nm in leaf_names}
    worst_rel = max(rels.values())
    min_cos = min(coss.values())
    warm_med = {
        "enc_ms": round(float(np.median(warm_enc)), 1),
        "cand_ms": round(float(np.median(warm_cand)), 1),
        "win_fwd_ms": round(float(np.median(warm_wfwd)), 1),
        "win_bwd_ms": round(float(np.median(warm_wbwd)), 1),
        "total_ms": round(float(np.median(warm_tot)), 1),
    }
    print(
        f"  ResidentPairBuilder: cand_fwd_rel={cand_fwd_rel:.3e}, win_fwd_rel={win_fwd_rel:.3e}, "
        f"bwd worst_rel={worst_rel:.3e}, min_cos={min_cos:.6f}, cold={pair_ms:.1f} ms, "
        f"warm={warm_med}, broken_x1.10={brk_worst:.3e}, pcie={pcie_snap}",
        flush=True,
    )
    for nm in leaf_names:
        print(f"    {nm:8s}: rel_l2={rels[nm]:.3e}  cos={coss[nm]:.6f}", flush=True)
    assert cand_fwd_rel < 2e-3 and win_fwd_rel < 2e-3, (cand_fwd_rel, win_fwd_rel)
    assert worst_rel <= 1e-2 and min_cos >= 0.9999, (rels, coss)
    assert brk_worst > 1e-2, f"broken ResidentPairBuilder arm failed to trip: {brk_worst}"
    return {
        "cand_fwd_rel": cand_fwd_rel, "win_fwd_rel": win_fwd_rel,
        "worst_rel": worst_rel, "min_cos": min_cos, "rels": rels, "broken_x1_10": brk_worst,
        "wall_ms": round(pair_ms, 1), "warm_ms": warm_med,
        "pcie": pcie_snap,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch8", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    from tt_bio import tenstorrent as T
    from tt_bio.rfd4 import engine as E, network as N
    from tt_bio.train.provenance import open_nodes

    pm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r4t_perf")
    sys.path.insert(0, pm_dir)
    import pcie_meter as PM
    PM.install(sites=False)

    dev = T.get_device()
    card = int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0])
    nodes = open_nodes()
    clk_node = nodes[0] if nodes else card
    clk, stop, t_clk = _aiclk_sampler(clk_node)

    state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
    cfg = config["nn"]
    glue = N.FlowGlue(state, config)
    glue.requires_grad_(True)

    from tt_bio import autograd as ag
    ag.install()

    res = {"card": card, "clk_node": clk_node}
    res["u32_smoke"] = test_u32_smoke(dev)
    res["heads"] = test_heads(dev, glue, args.batch8, PM)
    res["device_folds"] = test_device_folds(dev, state, cfg, glue, PM)
    res["device_optim"] = test_device_optim(dev, state, glue, PM)
    res["resident_pair"] = test_resident_pair(dev, glue, args.batch8, PM)

    stop.set()
    t_clk.join(timeout=1.0)
    res["aiclk_min"] = min(clk) if clk else 0
    res["aiclk_max"] = max(clk) if clk else 0
    res["aiclk_med"] = sorted(clk)[len(clk) // 2] if clk else 0
    res["aiclk_n"] = len(clk)
    print(f"\nALL HARDWARE UNIT SUITES PASSED! AICLK={res['aiclk_min']}..{res['aiclk_max']} (med={res['aiclk_med']}) MHz (n={res['aiclk_n']})", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
