#!/usr/bin/env python3
"""Screen 3: the two TriangleMultiplication levers, timed at OpenDDE's own 512 aa shapes.

opendde's trimul is 36.311 s of a 100.292 s fold (36.2 %), 1216 calls at 29.862 ms. The byte model
in the state doc says the module is DRAM-bound and names two deletions. This screen measures BOTH of
them at the exact shapes the fold presents, so the predicted landing is measured-per-op rather than
derived-from-bytes, and it checks the parity of each.

Shapes: c_z = 384, hidden = 384 (checkpoint `tri_mul_in.linear_a_p.weight` is [384, 384]),
TRIANGLE_MULT_CHUNK_SIZE = 32, S = 512 > TRIANGLE_MULT_L1_MAX_SEQ = 352 so the DRAM path,
`_trimul_inproj_group` halves 8 -> 4 because 12 % 8 != 0, giving group = 4 and THREE loop
iterations. The fused in-projection output is [1, 512, 512, 4*group*32].

LEVER 1 -- E6 (`reblock_permute_gated`): ships OFF and served 0 of 1216 in the measured fold. It
replaces `ttnn.chunk` + two sigmoid-gated `multiply_` + the forward channel move with one kernel.
Measured to WIN 1.214x on openfold3's trimul body and to LOSE on boltz2, where only 64 of 560 calls
were eligible. This screen reports `eligible_gated` at opendde's shape AND the timing.

LEVER 2 -- the in-projection group. `_trimul_inproj_group` HALVES from `_TRIMUL_INPROJ_GROUP` = 8,
so a channel loop with 12 pairs can only reach 4, and the whole normed pair tensor is re-read once
per iteration -- three times per call. A divisor search instead of a halving search reaches 12, one
iteration, which also drops `ttnn.reallocate` (guarded by `n_pairs // group > 1`) and the concat
(`_acc_concat` short-circuits a single block). Bit-exact by the docstring's own argument: the group
is a partition of an independent-channel sum.

GO/NO-GO, pre-committed before the numbers exist:
  E6      GO iff `eligible_gated` is True at opendde's gp shape AND the fused pair is `torch.equal`
          to the sequence it replaces AND it is >= 1.10x on that sequence.
  group   GO iff the g=12 leg is `torch.equal` to the g=4 leg AND the summed per-call op time falls
          by >= 2 ms (>= 2.4 s/fold, 28x the 0.086 s A/A floor).
Either failing is a NO-GO for that lever alone; they are independent.
"""
from __future__ import annotations
import json, os, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch, ttnn                                                              # noqa: E402
import tt_bio.tenstorrent as T                                                  # noqa: E402
import tt_bio.reblock_permute as RB                                             # noqa: E402
from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor  # noqa: E402

if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

dev = T.get_device()
GRID = tuple(T.COMPUTE_GRID_MAIN)
CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
MC = ttnn.DRAM_MEMORY_CONFIG
S, CZ, HID, CHUNK = 512, 384, 384, 32
torch.manual_seed(0)


def bench(fn, n=7, warm=2):
    outs = None
    for _ in range(warm):
        r = fn(); ttnn.synchronize_device(dev)
        for t in (r if isinstance(r, (list, tuple)) else [r]):
            ttnn.deallocate(t)
    ts = []
    for i in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        if i == 0:
            outs = [ttnn.to_torch(t) for t in (r if isinstance(r, (list, tuple)) else [r])]
        for t in (r if isinstance(r, (list, tuple)) else [r]):
            ttnn.deallocate(t)
    return st.median(ts) * 1e3, outs


def main():
    res = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
           "grid": list(GRID), "S": S, "c_z": CZ, "hidden": HID,
           "trimul_l1_max_seq": T.TRIANGLE_MULT_L1_MAX_SEQ,
           "seq_len_more_chunking": T.SEQ_LEN_MORE_CHUNKING,
           "is_small_grid": T._IS_SMALL_GRID,
           "inproj_group_const": T._TRIMUL_INPROJ_GROUP,
           "inproj_fused_bytes": T._TRIMUL_INPROJ_FUSED_BYTES,
           "group_today": T._trimul_inproj_group(S, CHUNK, 1, HID // CHUNK),
           "legs": []}
    print("group today =", res["group_today"], " n_pairs =", HID // CHUNK, flush=True)
    RB.set_enabled(True); RB.set_enabled_back(True); RB.set_enabled_gated(True)

    xn = ttnn.from_torch(torch.randn(1, S, S, CZ, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                         dtype=ttnn.bfloat16, device=dev, memory_config=MC)
    pc = T._triangle_mul_program_config((S + 31) // 32)

    for group in (4, 12):
        W = group * CHUNK                      # channels per role in one iteration
        n_iter = (HID // CHUNK) // group
        w = ttnn.from_torch(torch.randn(CZ, 4 * W, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat16, device=dev, memory_config=MC)
        leg = {"group": group, "chan_per_iter": W, "n_iter": n_iter, "ops": {}}
        print("\n=== group %d: %d iteration(s), %d channels/iter, gp width %d ==="
              % (group, n_iter, W, 4 * W), flush=True)

        mm_ms, _ = bench(lambda: ttnn.experimental.minimal_matmul(
            xn, w, memory_config=MC, dtype=ttnn.bfloat16, compute_kernel_config=CKC))
        leg["ops"]["inproj_mm"] = round(mm_ms, 4)
        print("  in-proj matmul                 %8.4f ms" % mm_ms, flush=True)

        gp = ttnn.experimental.minimal_matmul(xn, w, memory_config=MC, dtype=ttnn.bfloat16,
                                              compute_kernel_config=CKC)
        sc = int(gp.shape[-1]) // 4
        leg["slice_c"] = sc
        leg["eligible_gated"] = bool(RB.eligible_gated(gp, sc, MC))
        print("  eligible_gated(slice_c=%d) = %s" % (sc, leg["eligible_gated"]), flush=True)

        def plain():
            g_a, g_b, p_a, p_b = ttnn.chunk(gp, chunks=4, dim=-1)
            a = ttnn.multiply_(p_a, g_a, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            b = ttnn.multiply_(p_b, g_b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            ttnn.deallocate(g_a); ttnn.deallocate(g_b)
            a2 = T._channel_move(a, MC); ttnn.deallocate(a)
            b2 = T._channel_move(b, MC); ttnn.deallocate(b)
            b3 = ttnn.transpose(b2, -2, -1, memory_config=MC); ttnn.deallocate(b2)
            return [a2, b3]

        def fused():
            a = RB.reblock_permute_gated(gp, 2 * sc, 0, sc, memory_config=MC)
            b = RB.reblock_permute_gated(gp, 3 * sc, sc, sc, memory_config=MC)
            b3 = ttnn.transpose(b, -2, -1, memory_config=MC); ttnn.deallocate(b)
            return [a, b3]

        # `fused` only READS gp; `plain` calls `multiply_` on a `ttnn.chunk` result, which would
        # corrupt gp across bench repeats if chunk ever aliased. Run fused first, rebuild gp, then
        # plain -- so a hypothetical alias shows up as a parity failure instead of a silent drift.
        f_ms, f_out = bench(fused)
        ttnn.deallocate(gp)
        gp = ttnn.experimental.minimal_matmul(xn, w, memory_config=MC, dtype=ttnn.bfloat16,
                                              compute_kernel_config=CKC)
        p_ms, p_out = bench(plain)
        leg["ops"]["ab_plain"] = round(p_ms, 4)
        leg["ops"]["ab_fused_e6"] = round(f_ms, 4)
        leg["e6_ratio"] = round(p_ms / f_ms, 4)
        leg["e6_equal"] = [bool(torch.equal(x, y)) for x, y in zip(p_out, f_out)]
        print("  a,b plain (chunk+mult+move)    %8.4f ms" % p_ms, flush=True)
        print("  a,b fused  E6                  %8.4f ms   %.4fx  equal=%s"
              % (f_ms, p_ms / f_ms, leg["e6_equal"]), flush=True)

        a2, b3 = fused()
        ra_ms, _ = bench(lambda: ttnn.reallocate(a2, memory_config=MC))
        leg["ops"]["reallocate_one"] = round(ra_ms, 4)
        print("  reallocate (one chunk)         %8.4f ms" % ra_ms, flush=True)

        mm2_ms, _ = bench(lambda: ttnn.matmul(a2, b3, compute_kernel_config=CKC, memory_config=MC,
                                              program_config=pc, dtype=ttnn.bfloat16))
        leg["ops"]["triangle_matmul"] = round(mm2_ms, 4)
        print("  triangle matmul a@b            %8.4f ms" % mm2_ms, flush=True)

        xc = ttnn.matmul(a2, b3, compute_kernel_config=CKC, memory_config=MC,
                         program_config=pc, dtype=ttnn.bfloat16)
        mb_ms, _ = bench(lambda: T._channel_move_back(xc, MC))
        leg["ops"]["channel_move_back"] = round(mb_ms, 4)
        print("  channel move back              %8.4f ms" % mb_ms, flush=True)

        back = T._channel_move_back(xc, MC)
        if n_iter > 1:
            parts = [back] + [ttnn.clone(back, memory_config=MC) for _ in range(n_iter - 1)]
            cc_ms, _ = bench(lambda: ttnn.concat(parts, dim=-1))
            leg["ops"]["concat"] = round(cc_ms, 4)
            print("  concat %d chunks                %8.4f ms" % (n_iter, cc_ms), flush=True)
            for p in parts[1:]:
                ttnn.deallocate(p)
        else:
            leg["ops"]["concat"] = 0.0
            print("  concat                         %8.4f ms (single block, short-circuited)"
                  % 0.0, flush=True)

        o = leg["ops"]
        per_iter_plain = o["inproj_mm"] + o["ab_plain"] + o["triangle_matmul"] + o["channel_move_back"]
        per_iter_e6 = o["inproj_mm"] + o["ab_fused_e6"] + o["triangle_matmul"] + o["channel_move_back"]
        realloc = 2 * o["reallocate_one"] if n_iter > 1 else 0.0
        leg["loop_total_plain_ms"] = round(n_iter * (per_iter_plain + realloc) + o["concat"], 4)
        leg["loop_total_e6_ms"] = round(n_iter * (per_iter_e6 + realloc) + o["concat"], 4)
        print("  LOOP TOTAL plain %8.4f ms | with E6 %8.4f ms"
              % (leg["loop_total_plain_ms"], leg["loop_total_e6_ms"]), flush=True)

        ttnn.deallocate(gp); ttnn.deallocate(w); ttnn.deallocate(xc); ttnn.deallocate(back)
        ttnn.deallocate(a2); ttnn.deallocate(b3)
        res["legs"].append(leg)

    base = res["legs"][0]["loop_total_plain_ms"]
    res["deltas_ms_per_call"] = {
        "e6_only": round(base - res["legs"][0]["loop_total_e6_ms"], 4),
        "group12_only": round(base - res["legs"][1]["loop_total_plain_ms"], 4),
        "both": round(base - res["legs"][1]["loop_total_e6_ms"], 4)}
    print("\nper-call channel-loop deltas (ms):", res["deltas_ms_per_call"], flush=True)
    print("projected fold deltas at 1216 calls (s):",
          {k: round(v * 1216 / 1000, 3) for k, v in res["deltas_ms_per_call"].items()}, flush=True)

    p = Path(__file__).with_name("screen3.json")
    p.write_text(json.dumps(res, indent=1))
    print("wrote", p)


main()
