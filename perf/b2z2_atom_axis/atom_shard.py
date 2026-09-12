#!/usr/bin/env python3
"""The atom track, window-sharded over a real two-chip mesh: bit-exact, and what it costs.

`atom_curve.py` measured the thing that decides whether this is worth building: the atom encoder
and decoder fit t = 0.617 ms + 0.001256 ms/atom, so only 9.9 % of the atom track is independent
of the atom count, against the token DiT's 56.3 %. A perfect two-chip window split is therefore
worth 1.820x on the track. This file builds the split and prices what is not perfect about it.

## The halo, which is the whole design

An atom's queries live in one window of W=32; its keys are the H=128 rows centred on that window,
so window k reads atoms [32k-48, 32k+80). A contiguous slab of windows therefore needs its own
rows plus **48 atoms at each open edge** -- and needs them AFTER the layer's AdaLN, which is why
no amount of input replication removes the exchange: the key source is the layer's own output.

Rounded to the window grid that is 2 windows of halo per open edge. Per layer, per chip:

    edge      = concat(first 2 windows, last 2 windows)          4 windows, 32 KB
    all_gather over the mesh                                     one collective per layer
    L, R      = mesh_partition of the assembled halo columns     each chip takes its own
    s_widened = concat(L, own slab, R)                           74 windows on each chip

The per-chip `keys_indexing` is (2*74, 8*70) and is THE SAME MATRIX on both chips: with 2 windows
of left pad the local half-window index of window k's c-th key block is 2k + c + 1 on either
side, so the shard needs no per-device constant at all. The out-of-range edges of the real atom
axis are zero columns in the unsharded matrix and zero rows in the sharded source, which is the
same value by construction, not by luck.

## Arms

    whole     the atom transformer over all 140 windows, replicated on both chips. The reference
              output, and the one-chip time the ratio is taken against.
    shard     70 windows per chip, halo exchanged per layer, output all_gathered.
    nohalo    CONTROL: the same shard with the halo dropped (zeros instead of the neighbour's
              rows). Must differ from `whole`, or the check is not reading the halo.
    perturb   CONTROL: `whole` on a perturbed input. Must differ.

Run (whglx, the two chips verified to open as a 1x2 mesh):
    TT_VISIBLE_DEVICES=24,25 TT_BIO_LEASE_CARDS=24,25 TT_BIO_LEASE_HOLDER=worker:... \
    TT_BIO_TRACE_REGION_SIZE=536870912 python3 perf/b2z2_atom_axis/atom_shard.py --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from atom_curve import DEPTH, DIM, H, HEADS, W, build_inputs, make_weights  # noqa: E402

HALO_W = 2          # windows of halo per open edge; 2*32 = 64 >= the 48 rows a window reaches back


def open_mesh(tt, ttnn, n):
    """Install a 1xN mesh as tt_bio's device, at `_open_device_locked` (the injection point
    `perf/b2z2_pairchain/chain_bitexact.py` uses). Setting the module global by hand instead
    opens a second device underneath a live fabric mesh."""
    opens = [0]

    def _mesh_open(device_id, kwargs):
        opens[0] += 1
        with tt._device_init_lock():
            if n > 1:
                ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            dev = (ttnn.open_mesh_device(ttnn.MeshShape(1, n), **kwargs) if n > 1
                   else ttnn.open_device(device_id=device_id, **kwargs))
            tt._configure_active_compute_grid(dev)
            dev.enable_program_cache()
            return dev

    tt._open_device_locked = _mesh_open
    dev = tt.get_device()
    assert opens[0] == 1, "the mesh was opened more than once"
    return dev


def shard_keys_indexing(torch, NW_loc):
    """(2*(NW_loc+2*HALO_W), 8*NW_loc) one-hot, identical on every chip.

    Local half-window index of window k's c-th key block is 2k + c + 1: the unsharded rule is
    j = 2(k0+k) + c - 3 and the local source starts at global half-window 2*k0 - 2*HALO_W."""
    rows = 2 * (NW_loc + 2 * HALO_W)
    m = torch.zeros(rows, 8 * NW_loc)
    for k in range(NW_loc):
        for c in range(8):
            m[2 * k + c + 2 * HALO_W - 3, 8 * k + c] = 1.0
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--windows", type=int, default=140)
    ap.add_argument("--mesh", type=int, default=2)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parity-only", action="store_true")
    args = ap.parse_args()
    NW, MESH = args.windows, args.mesh
    assert NW % MESH == 0, "the window axis must split evenly"
    NWL = NW // MESH

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    assert Path(T.__file__).resolve().parents[1] == ROOT, f"wrong tt_bio: {T.__file__}"

    dev = open_mesh(T, ttnn, MESH)
    kernel_cls = (ttnn.types.WormholeComputeKernelConfig
                  if dev.arch() == ttnn.Arch.WORMHOLE_B0
                  else ttnn.types.BlackholeComputeKernelConfig)
    ckc = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                     fp32_dest_acc_en=True, packer_l1_acc=True)
    REPL = ttnn.replicate_tensor_to_mesh_mapper(dev)
    COMP = ttnn.concat_mesh_to_tensor_composer(dev, 0)

    def up(t, dt=None):
        return ttnn.from_torch(t, dtype=dt or ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=dev, mesh_mapper=REPL)

    def down_all(t):
        return ttnn.to_torch(t, mesh_composer=COMP)

    out = {"env": {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
                   "arch": T.arch_name(), "grid": str(T.CORE_GRID_MAIN), "mesh": f"1x{MESH}",
                   "NW": NW, "n_atoms": W * NW, "halo_windows": HALO_W,
                   "commit": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip(),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "loadavg_start": open("/proc/loadavg").read().split()[:3],
                   "reps": args.reps, "blocks": args.blocks},
           "parity": {}, "timing": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sd = make_weights(torch, args.seed)
    a_pt, s_pt, z_pt, ki_pt = build_inputs(torch, NW, args.seed + 7)
    ki_loc_pt = shard_keys_indexing(torch, NWL)

    # ---- whole: 140 windows, replicated ---------------------------------------------------
    whole = T.DiffusionTransformer(n_layers=DEPTH, dim=DIM, n_heads=HEADS, atom_level=True,
                                   state_dict=sd, compute_kernel_config=ckc)
    a_w, s_w = up(a_pt), up(s_pt)
    z_w = [up(x) for x in z_pt]
    ki_w = up(ki_pt, ttnn.bfloat4_b)
    ref = down_all(whole(a_w, s_w, z_w, ki_w))[0:1]

    # ---- shard: 70 windows per chip, halo exchanged per layer ------------------------------
    shard = T.DiffusionTransformer(n_layers=DEPTH, dim=DIM, n_heads=HEADS, atom_level=True,
                                   state_dict=sd, compute_kernel_config=ckc)
    zeros_pad = up(torch.zeros(1, HALO_W, W, DIM))
    COLLECTIVES = [0]

    def make_key_source(with_halo: bool):
        def key_source(s):
            """s: (1, NWL, W, DIM) on each chip -> (1, NWL + 2*HALO_W, W, DIM) with the
            neighbour's edge windows in place. One all_gather of 4 windows per layer."""
            if not with_halo:
                return ttnn.concat([zeros_pad, s, zeros_pad], dim=1)
            edge = ttnn.concat([s[:, :HALO_W], s[:, -HALO_W:]], dim=1)      # (1, 2*HALO_W, ...)
            allg = ttnn.all_gather(edge, dim=1)                             # (1, 2*MESH*HALO_W,...)
            COLLECTIVES[0] += 1
            # Column d of the assembled left/right halo is what device d must prepend/append;
            # mesh_partition hands each device its own column, so the slice below is the same
            # op on every device (SPMD) and still device-dependent in effect.
            lefts, rights = [], []
            for d in range(MESH):
                lefts.append(zeros_pad if d == 0 else allg[:, (2 * (d - 1) + 1) * HALO_W:
                                                           (2 * (d - 1) + 2) * HALO_W])
                rights.append(zeros_pad if d == MESH - 1 else
                              allg[:, 2 * (d + 1) * HALO_W:(2 * (d + 1) + 1) * HALO_W])
            L = ttnn.mesh_partition(ttnn.concat(lefts, dim=1), dim=1)
            R = ttnn.mesh_partition(ttnn.concat(rights, dim=1), dim=1)
            return ttnn.concat([L, s, R], dim=1)
        return key_source

    a_s = ttnn.mesh_partition(up(a_pt), dim=1)
    s_s = ttnn.mesh_partition(up(s_pt), dim=1)
    z_s = [ttnn.mesh_partition(up(x), dim=0) for x in z_pt]     # bias is (NW, heads, W, H)
    ki_s = up(ki_loc_pt, ttnn.bfloat4_b)

    def run_shard(with_halo=True):
        for layer in shard.layers:
            layer.attn_pair_bias.key_source = make_key_source(with_halo)
        o = shard(a_s, s_s, z_s, ki_s)
        return ttnn.all_gather(o, dim=1)

    got = down_all(run_shard(True))[0:1]
    nohalo = down_all(run_shard(False))[0:1]
    again = down_all(run_shard(True))[0:1]
    pert = down_all(whole(ttnn.add(a_w, 1.0), s_w, z_w, ki_w))[0:1]

    eq = lambda x, y: {"bit_exact": bool(x.shape == y.shape and torch.equal(x, y)),
                       "max_abs": float((x.float() - y.float()).abs().max())}
    out["parity"] = {
        "shard_vs_whole": eq(got, ref),
        "self_repeat": eq(got, again),
        "control_nohalo_differs": not torch.equal(nohalo, ref),
        "control_nohalo_max_abs": float((nohalo.float() - ref.float()).abs().max()),
        "control_perturbed_differs": not torch.equal(pert, ref),
        "collectives_per_run": COLLECTIVES[0] // 3,
        "both_chips_agree_after_gather": bool(torch.equal(
            down_all(run_shard(True))[0:1], down_all(run_shard(True))[1:2])),
    }
    print("PARITY", json.dumps(out["parity"], indent=1), flush=True)
    args.out.write_text(json.dumps(out, indent=1))
    if args.parity_only:
        ok = (out["parity"]["shard_vs_whole"]["bit_exact"]
              and out["parity"]["control_nohalo_differs"]
              and out["parity"]["control_perturbed_differs"]
              and out["parity"]["both_chips_agree_after_gather"])
        print("PARITY-ONLY", "PASS" if ok else "FAIL", flush=True)
        return 0 if ok else 1

    # ---- timing: traced, both arms in one process ------------------------------------------
    def timed(fn, tag):
        fn(); fn()
        ttnn.synchronize_device(dev)
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        fn()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        for _ in range(3):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev)
        ms = []
        for _ in range(args.blocks):
            t0 = time.perf_counter()
            for _ in range(args.reps):
                ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(dev)
            ms.append(1e3 * (time.perf_counter() - t0) / args.reps)
        ttnn.release_trace(dev, tid)
        r = {"ms": round(st.median(ms), 5),
             "spread_pct": round(100 * (max(ms) - min(ms)) / st.median(ms), 3),
             "blocks_ms": [round(x, 5) for x in ms]}
        out["timing"][tag] = r
        print(f"  {tag:16s} {r['ms']:9.4f} ms (spread {r['spread_pct']:.2f} %)", flush=True)
        args.out.write_text(json.dumps(out, indent=1))
        return r["ms"]

    for layer in shard.layers:
        layer.attn_pair_bias.key_source = make_key_source(True)
    t_whole = timed(lambda: whole(a_w, s_w, z_w, ki_w), "whole")
    t_shard_nog = timed(lambda: shard(a_s, s_s, z_s, ki_s), "shard_no_outgather")
    t_shard = timed(lambda: run_shard(True), "shard")
    # what the halo exchange itself costs: the same slab with the halo replaced by a local pad
    for layer in shard.layers:
        layer.attn_pair_bias.key_source = make_key_source(False)
    t_nohalo = timed(lambda: shard(a_s, s_s, z_s, ki_s), "shard_no_halo")

    out["result"] = {
        "track_ratio": round(t_whole / t_shard, 5),
        "track_ratio_no_outgather": round(t_whole / t_shard_nog, 5),
        "perfect_ratio_from_curve": 1.81995,
        "halo_cost_ms": round(t_shard_nog - t_nohalo, 5),
        "halo_cost_us_per_layer": round(1e3 * (t_shard_nog - t_nohalo) / DEPTH, 2),
        "outgather_cost_ms": round(t_shard - t_shard_nog, 5),
    }
    out["env"]["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["result"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
