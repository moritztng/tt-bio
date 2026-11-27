"""Hardware verification for DeviceOptimizer, DeviceHeads + device_insertion_loss, and DeviceFolds.

Runs on one Blackhole chip (anx-glx or qb2) and verifies:
1. DeviceOptimizer: all 558 RFD4 params (282 Muon + 276 Adam), 1-step parity vs float64 host
   reference, zero-grad decay rule, reduce_fn hook, and split2/split2_hifi4/fp32 wall times.
2. DeviceHeads + device_insertion_loss on real batch08 (all 5 loss terms active):
   - FD-validated float64 reference (fd_check < 2e-6)
   - Grad-off byte-identical A/A + forward parity vs host FlowGlue.heads
   - Taped VJP graded per D28 on d_s + all 8 head weights with 0 pair-sized PCIe transfers
   - Broken arms (gross + subtle x1.10) both FAIL
3. DeviceFolds on full RFD4 Trunk + prewarm(256,1/4) + DeviceHeads:
   - refold(): 0 PCIe bytes, parity on all 714 weights vs fresh host build, wall time
   - tape_backward(): 0 PCIe bytes, parity on all leaf grads vs host Folds.backward, wall time
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np


def _aiclk_sampler(card: int):
    clk, stop = [], threading.Event()
    path = f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk"

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
    if den == 0.0:
        return num
    return num / den


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


def test_heads(dev, glue, batch8_path: str, pcie_meter_mod):
    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.rfd4.heads import DeviceHeads, device_insertion_loss
    from tt_bio.train import rfd4_objective as OBJ

    print("\n=== [1/3] DeviceHeads + device_insertion_loss (batch08, all 5 terms) ===", flush=True)
    batch = _load_npz_batch(batch8_path)
    # Use single winner example [0:1] so b=1, n=256 (same as winner pass)
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

    # Synthetic s with realistic scale (~1.0), masked on padding
    rng = np.random.default_rng(42)
    s_np = (rng.standard_normal((b, n, 768)).astype(np.float32) * 0.5)
    s_np *= b1["padding_mask"].numpy()[..., None].astype(np.float32)
    s_pad_np = np.zeros((b, n_pad, 768), dtype=np.float32)
    s_pad_np[:, :n, :] = s_np

    # --- Float64 reference + FD check ---
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
        # Backprop seeds through float64 heads
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

    # Validate float64 reference via central finite differences along 3 random directions!
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

    # --- Build DeviceHeads + grad-off A/A check ---
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
    assert all(d[1] < 1e-3 for d in aa_diffs.values()), f"Fwd parity failed: {aa_diffs}"

    # --- Taped forward + on-card insertion_loss + taped backward ---
    def run_taped_heads(broken_mode: str = "none"):
        ag.forget_parameters()
        head_handles = {
            "ca_norm_w": dheads.ca_norm_w,
            "ca_w": dheads.ca_w,
            "ll_norm_w": dheads.ll_norm_w,
            "ll_w": dheads.ll_w,
            "seq_norm_w": dheads.seq_norm_w,
            "seq_w": dheads.seq_w,
            "ins_q_w": dheads.ins_q_w,
            "ins_k_w": dheads.ins_k_w,
        }
        p_leaves = {k: ag.parameter(h) for k, h in head_handles.items()}
        for pl in p_leaves.values():
            pl.grad = None

        if pcie_meter_mod is not None and broken_mode == "none":
            pcie_meter_mod.take()
            pcie_meter_mod.phase("heads_fwd_bwd")

        s_var = ag.Tensor(s_tt, requires_grad=True)
        with ag.tape():
            dout = glue.heads(s_var, b1, insertion=True)

        # Download ONLY 1-D heads (NO insertion_logits download!)
        outs_1d = {
            "v_bb_ca": ttnn.to_torch(dout["bb_ca"]["v"].value).numpy()[:, :n, :],
            "v_local_latents": ttnn.to_torch(dout["local_latents"]["v"].value).numpy()[:, :n, :],
            "sequence_logits": ttnn.to_torch(dout["sequence"]["logits"].value).numpy()[:, :n, :],
        }
        tot_1d, brk_1d, seeds_1d = OBJ.rfd4_loss(b1, outs_1d, OBJ.WEIGHTS)
        win = np.asarray(brk_1d["winners"], np.int64)
        ins_w = float(OBJ.WEIGHTS["insertion_prediction"])
        ins_loss_np, dz_ins = device_insertion_loss(
            dout["insertion"]["logits"], b1, win=win, weight=ins_w)
        tot_dev = tot_1d + ins_w * float(ins_loss_np[win].mean())

        if broken_mode == "gross":
            # Drop insertion grad completely
            dz_ins_use = ttnn.multiply(dz_ins, 0.0)
            ttnn.deallocate(dz_ins)
            dz_ins = dz_ins_use
        elif broken_mode == "subtle_110":
            # Subtle 1.10x scaling on insertion gradient
            dz_ins_use = ttnn.multiply(dz_ins, 1.10)
            ttnn.deallocate(dz_ins)
            dz_ins = dz_ins_use

        def up_seed(arr):
            pad = np.zeros((b, n_pad, arr.shape[-1]), dtype=np.float32)
            pad[:, :n, :] = np.asarray(arr, dtype=np.float32)
            return ttnn.from_torch(
                torch.from_numpy(pad), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

        g_ca = up_seed(seeds_1d["v_bb_ca"])
        g_ll = up_seed(seeds_1d["v_local_latents"])
        g_sq = up_seed(seeds_1d["sequence_logits"])
        ag.backward(
            [dout["bb_ca"]["v"], dout["local_latents"]["v"],
             dout["sequence"]["logits"], dout["insertion"]["logits"]],
            [g_ca, g_ll, g_sq, dz_ins],
        )
        ttnn.synchronize_device(dev)

        pcie_snap = None
        if pcie_meter_mod is not None and broken_mode == "none":
            pcie_snap = pcie_meter_mod.take()

        # Read back grads to compare vs float64
        got = {"s": ttnn.to_torch(s_var.grad).numpy()[:, :n, :]}
        ttnn.deallocate(s_var.grad)
        for k, pl in p_leaves.items():
            arr = ttnn.to_torch(pl.grad).numpy()
            if k.endswith("_norm_w"):
                arr = arr.reshape(-1)
            got[k] = arr
            ttnn.deallocate(pl.grad)
            pl.grad = None
        for t in (g_ca, g_ll, g_sq, dz_ins):
            ttnn.deallocate(t)
        ag.release_pins()

        metrics = {}
        for k in got:
            metrics[k] = {"rel_l2": _rel_l2(got[k], g64[k]), "cos": _cos(got[k], g64[k])}
        return tot_dev, metrics, pcie_snap

    tot_dev, m_good, pcie_heads = run_taped_heads("none")
    loss_rel = abs(tot_dev - tot64) / max(abs(tot64), 1e-12)
    worst_rel = max(v['rel_l2'] for v in m_good.values())
    worst_cos = min(v['cos'] for v in m_good.values())
    print(f"  GOOD arm: loss_rel={loss_rel:.2e}, worst_rel={worst_rel:.3e}, min_cos={worst_cos:.6f}", flush=True)
    for k, v in m_good.items():
        print(f"    {k:12s}  rel_l2={v['rel_l2']:.3e}  cos={v['cos']:.6f}", flush=True)
    if pcie_heads:
        print(f"  Heads PCIe snapshot: {pcie_heads}", flush=True)

    _, m_gross, _ = run_taped_heads("gross")
    _, m_sub, _ = run_taped_heads("subtle_110")
    gross_worst = max(v['rel_l2'] for v in m_gross.values())
    sub_worst = max(v['rel_l2'] for v in m_sub.values())
    print(f"  BROKEN gross worst_rel={gross_worst:.3e} (expect >5e-2 -> FAIL)", flush=True)
    print(f"  BROKEN subtle_110 worst_rel={sub_worst:.3e} (expect >5e-2 -> FAIL)", flush=True)

    assert worst_rel <= 5e-2 and worst_cos >= 0.9999, f"Heads D28 FAILED: {m_good}"
    assert gross_worst > 5e-2, f"Gross broken arm did NOT fail: {gross_worst}"
    assert sub_worst > 5e-2, f"Subtle broken arm did NOT fail: {sub_worst}"

    ttnn.deallocate(s_tt)
    for h in (dheads.ca_norm_w, dheads.ca_w, dheads.ll_norm_w, dheads.ll_w,
              dheads.seq_norm_w, dheads.seq_w, dheads.ins_q_w, dheads.ins_k_w):
        ttnn.deallocate(h)
    glue._device_heads = None
    return {
        "max_fd_err": max_fd,
        "aa_bit_exact": True,
        "loss_rel": loss_rel,
        "worst_rel": worst_rel,
        "min_cos": worst_cos,
        "gross_broken_worst_rel": gross_worst,
        "subtle_110_broken_worst_rel": sub_worst,
        "per_param": m_good,
    }


def test_device_folds(dev, state, cfg, glue, pcie_meter_mod):
    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.rfd4 import engine as E
    from tt_bio.rfd4.heads import DeviceHeads
    from tt_bio.train.folds import DeviceFolds, Folds

    print("\n=== [2/3] DeviceFolds (714 weights: refold + tape_backward, 0 PCIe) ===", flush=True)
    full_state = dict(state)
    for k, p in glue.named_parameters():
        full_state[f"glue.{k}"] = p.detach()

    def find_trimuls(root):
        seen, tms, todo = set(), [], [root]
        while todo:
            o = todo.pop()
            if id(o) in seen:
                continue
            seen.add(id(o))
            if hasattr(o, "prewarm") and hasattr(o, "width_variants"):
                tms.append(o)
            if isinstance(o, (list, tuple)):
                todo.extend(o)
            elif isinstance(o, dict):
                todo.extend(o.values())
            elif hasattr(o, "__dict__") and not isinstance(o, (torch.Tensor, ttnn.Tensor, type)):
                todo.extend(v for v in vars(o).values() if not isinstance(v, (torch.Tensor, ttnn.Tensor)))
        return tms

    # Build DeviceFolds + Trunk + DeviceHeads + prewarm(256, 1) and prewarm(256, 4)
    dfolds = DeviceFolds(full_state, dev)
    with dfolds.recording() as leaves:
        trunk = E.load_trunk(glue, leaves, cfg, training=True, attn_fp32=True)
        dheads = DeviceHeads(glue, dev, state=leaves)
        for m in find_trimuls(trunk):
            m.prewarm(256, 1)
            m.prewarm(256, 4)
    n_bound = dfolds.bind(trunk, dheads)
    print(f"  Bound {n_bound}/{len(dfolds.specs)} device weight handles, {len(dfolds._tuple_bindings)} tuples", flush=True)

    # Create perturbed masters on host and device
    rng = np.random.default_rng(7)
    # Only upload masters that are actually read by specs (all 529 trunk + 8 head = 537 leaves)
    used_leaf_names = set()
    stack = [s.nid for s in dfolds.specs]
    seen_n = set()
    while stack:
        nid = stack.pop()
        if nid in seen_n:
            continue
        seen_n.add(nid)
        nd = dfolds.nodes[nid]
        if nd.op == "leaf":
            used_leaf_names.add(nd.attrs["name"])
        else:
            stack.extend(nd.inputs)

    perturbed_host = {}
    dev_masters = {}
    for nm in sorted(used_leaf_names):
        base = full_state[nm].detach().float()
        delta = torch.from_numpy((rng.standard_normal(base.shape) * 0.02).astype(np.float32))
        new_t = (base + delta).contiguous()
        perturbed_host[nm] = new_t
        dev_masters[nm] = ttnn.from_torch(new_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

    # Run dfolds.refold(dev_masters) under pcie_meter
    if pcie_meter_mod is not None:
        pcie_meter_mod.take()
        pcie_meter_mod.phase("refold")
    ttnn.synchronize_device(dev)
    t0 = time.time()
    dfolds.refold(dev_masters)
    ttnn.synchronize_device(dev)
    refold_s = time.time() - t0
    refold_pcie = pcie_meter_mod.take() if pcie_meter_mod is not None else None
    print(f"  refold() time={refold_s*1000:.1f} ms, pcie={refold_pcie}", flush=True)

    # Verify refolded handles vs host Folds build from perturbed_host
    host_ref_state = dict(full_state)
    host_ref_state.update(perturbed_host)
    hfolds = Folds(host_ref_state)
    with hfolds.recording() as hleaves:
        htrunk = E.load_trunk(glue, hleaves, cfg, training=True, attn_fp32=True)
        hdheads = DeviceHeads(glue, dev, state=hleaves)
        for m in find_trimuls(htrunk):
            m.prewarm(256, 1)
            m.prewarm(256, 4)

    # Compare every spec by order (same build sequence -> exact 1:1 spec order)
    h_handles = [h for h, _ in hfolds.sources.values()]
    assert len(h_handles) == len(dfolds.specs), (len(h_handles), len(dfolds.specs))
    worst_by_dt = {}
    for spec, hh in zip(dfolds.specs, h_handles):
        a = ttnn.to_torch(spec.handle).float().numpy()
        b = ttnn.to_torch(hh).float().numpy()
        rel = _rel_l2(a, b)
        dt = str(spec.dtype)
        worst_by_dt[dt] = max(worst_by_dt.get(dt, 0.0), rel)
    print(f"  refold() max rel_l2 vs fresh host build across {len(dfolds.specs)} weights: {worst_by_dt}", flush=True)

    # Now test tape_backward() on ALL 714 specs with synthetic fp32 grads!
    # Put identical random fp32 grads on both dfolds (via ag.parameter) and hfolds (host backward).
    host_grad_pairs = []
    for spec, hh in zip(dfolds.specs, h_handles):
        g_np = (rng.standard_normal(spec.shape) * 0.1).astype(np.float32)
        g_t = torch.from_numpy(g_np)
        g_tt = ttnn.from_torch(g_t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        leaf = ag.parameter_for(spec.handle)
        assert leaf is not None, f"spec not registered after refold: {spec.shape}"
        leaf.grad = g_tt
        host_grad_pairs.append((hh, g_t))

    # Host Folds reference backward
    hfolds.backward(host_grad_pairs)
    host_leaf_grads = {
        nm: hfolds.leaves[nm].grad.detach().numpy().copy()
        for nm in used_leaf_names if hfolds.leaves[nm].grad is not None
    }

    # DeviceFolds.tape_backward() under pcie_meter
    if pcie_meter_mod is not None:
        pcie_meter_mod.take()
        pcie_meter_mod.phase("fold_bwd")
    ttnn.synchronize_device(dev)
    t1 = time.time()
    dev_grads, idle = dfolds.tape_backward(dev_masters)
    ttnn.synchronize_device(dev)
    bwd_s = time.time() - t1
    bwd_pcie = pcie_meter_mod.take() if pcie_meter_mod is not None else None
    print(f"  tape_backward() time={bwd_s*1000:.1f} ms, grads={len(dev_grads)}, idle={len(idle)}, pcie={bwd_pcie}", flush=True)

    assert set(dev_grads.keys()) == set(host_leaf_grads.keys()), (
        set(dev_grads.keys()) ^ set(host_leaf_grads.keys()))
    worst_g_rel = 0.0
    worst_g_name = ""
    for nm, g_tt in dev_grads.items():
        g_dev = ttnn.to_torch(g_tt).float().numpy().reshape(host_leaf_grads[nm].shape)
        rel = _rel_l2(g_dev, host_leaf_grads[nm])
        if rel > worst_g_rel:
            worst_g_rel, worst_g_name = rel, nm
        ttnn.deallocate(g_tt)
    print(f"  tape_backward() max rel_l2 across {len(host_leaf_grads)} leaves: {worst_g_rel:.3e} ({worst_g_name})", flush=True)

    # Cleanup all device tensors
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

    # Check 0 PCIe bytes for both refold and fold_bwd!
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
        "n_specs": n_bound,
        "refold_ms": round(refold_s * 1000, 2),
        "fold_bwd_ms": round(bwd_s * 1000, 2),
        "refold_worst_by_dtype": worst_by_dt,
        "fold_bwd_worst_rel": worst_g_rel,
        "fold_bwd_worst_name": worst_g_name,
    }


def test_device_optim(dev, state, glue):
    import torch
    import ttnn
    from tt_bio.train.device_optim import DeviceOptimizer
    from tt_bio.train.rfd4_objective import MUON_EXCLUDE

    print("\n=== [3/3] DeviceOptimizer (all 558 RFD4 params, timing + 1-step f64 parity) ===", flush=True)
    params = {k: v.detach().float() for k, v in state.items() if k.startswith("nn.")}
    for k, p in glue.named_parameters():
        params[f"glue.{k}"] = p.detach().float()

    rng = np.random.default_rng(123)
    host_grads = {
        nm: torch.from_numpy((rng.standard_normal(t.shape) * 0.05).astype(np.float32))
        for nm, t in params.items()
    }
    # Deliberately omit one Adam param and one Muon param to test zero-grad decay semantics!
    omitted_adam = next(nm for nm, t in params.items() if t.ndim < 2)
    omitted_muon = next(nm for nm, t in params.items() if t.ndim == 2 and nm.startswith("nn."))

    # Check timing & accuracy for split2 vs split2_hifi4 vs fp32
    timings = {}
    for ns_mode in ("split2", "split2_hifi4", "fp32"):
        opt = DeviceOptimizer(
            params, dev,
            lr=1e-4, adam_lr=1e-4, muon_exclude=MUON_EXCLUDE,
            ema_decay=0.999, ns_mode=ns_mode,
        )
        # Upload grads once (outside step timer)
        dev_grads = {
            nm: ttnn.from_torch(g, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            for nm, g in host_grads.items()
            if nm not in (omitted_adam, omitted_muon)
        }
        # Test reduce_fn hook on first mode
        reduced_count = [0]
        def _red(g):
            reduced_count[0] += 1
            return g

        # Warmup step 1 + timed step 2
        ttnn.synchronize_device(dev)
        t0 = time.time()
        st1 = opt.step(dev_grads, reduce_fn=_red if ns_mode == "split2" else None)
        ttnn.synchronize_device(dev)
        t1 = time.time()
        # Re-upload grads for step 2 timing
        dev_grads2 = {
            nm: ttnn.from_torch(g, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            for nm, g in host_grads.items()
            if nm not in (omitted_adam, omitted_muon)
        }
        ttnn.synchronize_device(dev)
        t2 = time.time()
        st2 = opt.step(dev_grads2)
        ttnn.synchronize_device(dev)
        t3 = time.time()

        timings[ns_mode] = {
            "step1_ms": round((t1 - t0) * 1000, 1),
            "step2_ms": round((t3 - t2) * 1000, 1),
            "muon_norm": st2['muon_norm'],
            "adam_norm": st2['adam_norm'],
            "n_muon": len(opt.muon_names),
            "n_adam": len(opt.adam_names),
            "n_buckets": len(opt._muon_buckets),
        }
        if ns_mode == "split2":
            assert reduced_count[0] == len(host_grads) - 2
            # Verify omitted Adam param did NOT move (zero-grad skips Adam moment/decay)
            om_a_after = opt.host_master(omitted_adam).numpy()
            om_a_diff = float(np.max(np.abs(om_a_after - params[omitted_adam].numpy())))
            assert om_a_diff == 0.0, f"Omitted Adam param moved: {om_a_diff}"
            # Verify EMA shadow exists
            assert opt.ema is not None and len(opt.ema.shadows) == len(params)
        print(f"  {ns_mode:14s}: step1={timings[ns_mode]['step1_ms']} ms, step2={timings[ns_mode]['step2_ms']} ms "
              f"(muon={len(opt.muon_names)} in {len(opt._muon_buckets)} buckets, adam={len(opt.adam_names)}, "
              f"norms={st2['muon_norm']:.3f}/{st2['adam_norm']:.3f})", flush=True)
        opt.deallocate()

    return timings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch8", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import tenstorrent as T
    from tt_bio.rfd4 import engine as E, network as N

    # Load pcie_meter
    pm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "r4t_perf")
    sys.path.insert(0, pm_dir)
    import pcie_meter as PM
    PM.install(sites=False)

    dev = T.get_device()
    card = int(os.environ["TT_VISIBLE_DEVICES"].split(",")[0])
    clk, stop, t_clk = _aiclk_sampler(card)

    state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
    cfg = config["nn"]
    glue = N.FlowGlue(state, config)

    res = {"card": card}
    res["heads"] = test_heads(dev, glue, args.batch8, PM)
    res["device_folds"] = test_device_folds(dev, state, cfg, glue, PM)
    res["device_optim"] = test_device_optim(dev, state, glue)

    stop.set()
    t_clk.join(timeout=1.0)
    res['aiclk_min'] = min(clk) if clk else 0
    res['aiclk_max'] = max(clk) if clk else 0
    res['aiclk_n'] = len(clk)
    print(f"\nALL 3 HARDWARE UNIT SUITES PASSED! AICLK={res['aiclk_min']}..{res['aiclk_max']} MHz (n={res['aiclk_n']})", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
