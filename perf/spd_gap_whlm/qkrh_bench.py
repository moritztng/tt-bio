"""ESMC's q/k path after the qkv projection: today's six-op chain against the qk_rope_heads program, both vs float64.

Today (Attention._qk_heads + _rope): layer_norm(q), layer_norm(k), concat with v, nlp_create_qkv_heads, two
rotary_embedding calls. tt_bio.qk_rope_heads does the norms, the RoPE and the head split of q, k and v in one program.
Reports rel_l2 / max_abs of q and k heads against float64, whether v's heads equal the input bit for bit, and the
median ms of each arm (sync, N calls, sync).

    python3 perf/spd_gap_whlm/qkrh_bench.py --out qkrh.jsonl
"""
import argparse
import json
import math

import torch
import ttnn

from qkfuse_bench import SHAPES, S, err, ref_heads, rope64, timed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--inner", type=int, default=10)
    ap.add_argument("--fidelity", default="HiFi4", help="qk_rope_heads.FIDELITY arms")
    ap.add_argument("--block", default="1,4", help="qk_rope_heads.BLOCK arms (tiles a dest acquire)")
    ap.add_argument("--bf8", action="store_true", help="add arm qkrh_bf8: block-fp8 q, k, v out (v graded vs float64)")
    ap.add_argument("--wflush", action="store_true", help="add arm qkrh_bf8_wbar: --bf8 with a write barrier per head")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio import tenstorrent as T
    from tt_bio import qk_rope_heads as QR
    from tt_bio.esmc import _rope, rope_tables
    dev = T.get_device()
    ck = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
    up = lambda t: ttnn.from_torch(t.contiguous(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)  # noqa: E731
    out = open(args.out, "w")
    gen = torch.Generator().manual_seed(0)
    for name in args.shapes.split(","):
        B, d, H = SHAPES[name]
        hd = d // H
        rec = {"shape": name, "bsd_h": [B, S, d, H], "l1_bytes": QR._l1_bytes(d // 32)}
        x = torch.randn(B, S, d, generator=gen).to(torch.bfloat16)
        ws = [(torch.randn(d, d, generator=gen) / math.sqrt(d) + 0.02 * torch.randn(d, 1, generator=gen)).to(torch.bfloat16)
              for _ in range(3)]
        gq, gk = ((1 + 0.1 * torch.randn(d, generator=gen)).to(torch.bfloat16) for _ in range(2))
        x_tt = up(x)
        q, k, v = (ttnn.linear(x_tt, up(w), compute_kernel_config=ck, dtype=ttnn.bfloat16) for w in ws)
        # the reference starts from the bf16 projections the device holds, so both arms are graded on the same input
        q64, k64, v64 = (ttnn.to_torch(t).double().reshape(B, S, d) for t in (q, k, v))
        c64, s64 = rope64(S, hd)
        ref_q, ref_k = ref_heads(q64, gq.double(), H, c64, s64), ref_heads(k64, gk.double(), H, c64, s64)
        ref_v = v64.view(B, S, H, hd).transpose(1, 2)
        cos, sin = rope_tables(S, hd, device=dev)
        gq_tt, gk_tt = up(gq.view(1, d)), up(gk.view(1, d))

        def chain():
            qn = ttnn.layer_norm(q, weight=gq_tt, epsilon=1e-5, compute_kernel_config=ck)
            kn = ttnn.layer_norm(k, weight=gk_tt, epsilon=1e-5, compute_kernel_config=ck)
            qkv = ttnn.unsqueeze(ttnn.concat([qn, kn, v], dim=-1), 1)
            qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False,
                                                                memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for t in (qn, kn, qkv):
                ttnn.deallocate(t)
            qr, kr = _rope(qh, kh, cos, sin, l1=False)
            for t in (qh, kh):
                ttnn.deallocate(t)
            return qr, kr, vh

        def fused():
            return QR.qk_rope_heads(q, k, v, gq_tt, gk_tt, cos, sin)

        def fused_bf8():
            return QR.qk_rope_heads(q, k, v, gq_tt, gk_tt, cos, sin, out_dtype=ttnn.bfloat8_b)

        def rope_only():   # the same program and bytes with the norm compiled out: what the LayerNorm costs
            return QR.rope_heads_packed(qkv_packed, cos, sin)

        def copy():        # q, k, v read and written once by stock ops: the DRAM floor of the program
            return [ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG) for t in (q, k, v)]

        qkv_packed = ttnn.concat([q, k, v], dim=-1)
        blocks, first = args.block.split(","), {}
        arms = [("chain", chain, None, "1")]
        arms += [(f"qkrh_{f}_b{b}", fused, f, b) for f in args.fidelity.split(",") for b in blocks]
        arms += [("qkrh_bf8", fused_bf8, None, "4")] if args.bf8 else []
        arms += [("qkrh_bf8_wbar", fused_bf8, None, "4")] if args.wflush else []
        arms += [(f"rope_only_b{b}", rope_only, None, b) for b in blocks] + [("copy", copy, None, "1")]
        for arm, fn, fid, blk in arms:
            QR.FIDELITY = getattr(ttnn.MathFidelity, fid or "HiFi4")
            QR.BLOCK = int(blk)
            QR.WRITE_FLUSH = not arm.endswith("_wbar")
            try:
                if arm.startswith(("rope_only", "copy")):
                    o = fn()
                    if arm.startswith("rope_only"):   # bit-identity across blocks (SaProt's packed path)
                        got = torch.cat([ttnn.to_torch(t).float().flatten() for t in o])
                        rec[arm] = {"same_as_first": bool(torch.equal(got, first.setdefault("rope", got)))}
                    for t in o:
                        ttnn.deallocate(t)
                    rec.setdefault(arm, {})["ms"] = timed(lambda fn=fn: [ttnn.deallocate(t) for t in fn()], dev, args.repeat, args.inner)
                    continue
                o = fn()
                if o is None:
                    rec[arm] = {"declined": True}
                    continue
                got = [ttnn.to_torch(t).double().reshape(B, H, S, hd) for t in o]
                rec[arm] = {"q": err(got[0], ref_q), "k": err(got[1], ref_k),
                            "v_exact": bool(torch.equal(got[2], ref_v))}
                if arm.startswith("qkrh_bf8"):
                    rec[arm]["v"] = err(got[2], ref_v)
                elif arm != "chain":
                    flat = torch.cat([g.flatten() for g in got[:2]])
                    rec[arm]["same_as_first"] = bool(torch.equal(flat, first.setdefault("qkrh", flat)))
                for t in o:
                    ttnn.deallocate(t)
                rec[arm]["ms"] = timed(lambda fn=fn: [ttnn.deallocate(t) for t in fn()], dev, args.repeat, args.inner)
            except Exception as exc:   # a kernel that fails to build or run is a result here, not a crash
                rec[arm] = {"error": str(exc).splitlines()[0][:300]}
        QR.FIDELITY, QR.BLOCK, QR.WRITE_FLUSH = ttnn.MathFidelity.HiFi4, 1, True
        ttnn.deallocate(qkv_packed)
        print(json.dumps(rec), flush=True)
        out.write(json.dumps(rec) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
