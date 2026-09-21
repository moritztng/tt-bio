#!/usr/bin/env python3
"""Where exactly does the head-major write land wrong?

`c12-diffusion-head-major` pass 10 read `torch.equal(A1, B) = False` at `apb_trunk` with
max_abs 0.0703125 and stopped there. That number cannot tell an ADDRESS bug from a VALUE bug, and
the two have nothing in common: an address bug is a permutation of correct tiles, a value bug is
arithmetic. This asks the buffer which one it is.

Reference is the raw `minimal_matmul` output brought to host and sliced/permuted in torch -- the
same tensor both arms compute from, so nothing here compares one device arm against another.

    python3 localize.py --sig apb_trunk
"""
import argparse
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]

# name, batch, seq, c_in, heads, head_dim, padded_head_dim
SIGS = {
    "apb_trunk": (1, 512, 384, 16, 32, 32),
    "dit_token": (1, 512, 768, 16, 48, 64),
}
TILE = 32


def _say(*a):
    print(*a, flush=True)


def run(sig_name, device_id):
    import torch
    import ttnn
    sys.path.insert(0, str(ROOT))
    from tt_bio import tenstorrent as TT
    from tt_bio import triatt_qkv as TQ

    b, s, c_in, heads, hd, phd = SIGS[sig_name]
    _say(f"{sig_name}: b={b} s={s} c_in={c_in} heads={heads} hd={hd} phd={phd}")
    _say("opening device via tt_bio.tenstorrent.get_device()")
    dev = TT.get_device()
    _say(f"device open, arch={dev.arch()}")

    torch.manual_seed(0)
    x_t = torch.randn(b, s, c_in, dtype=torch.bfloat16)
    w_t = torch.randn(c_in, 3 * heads * phd, dtype=torch.bfloat16) * 0.02
    bias_t = torch.randn(3 * heads * phd, dtype=torch.bfloat16) * 0.02
    mk = dict(layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
              memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x, w, bias = (ttnn.from_torch(t, **mk) for t in (x_t, w_t, bias_t))

    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    ckc = cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    cfg = TT._qkv_mm_config(x, w, "apb")
    assert cfg is not None

    _say("arm REF: raw minimal_matmul, no split")
    qkv = ttnn.experimental.minimal_matmul(
        input_tensor=x, weight_tensor=w, bias_tensor=bias, compute_kernel_config=ckc,
        dtype=ttnn.bfloat16, config=cfg)
    qkv_t = ttnn.to_torch(qkv)                       # [b, s, 3*heads*phd]
    _say(f"  qkv {tuple(qkv_t.shape)}")
    ref = [qkv_t[..., c * heads * phd:(c + 1) * heads * phd]
           .reshape(b, s, heads, phd).permute(0, 2, 1, 3).contiguous() for c in range(3)]

    _say("arm A1: nlp_create_qkv_heads on the same buffer")
    a1 = [ttnn.to_torch(t) for t in ttnn.experimental.nlp_create_qkv_heads(
        ttnn.unsqueeze(qkv, 1), num_heads=heads, num_kv_heads=heads, transpose_k_heads=False)]

    _say("arm B: head-major destination")
    TQ._APB_ENABLED = True
    outs = TQ.qkv_heads(x, w, ckc, heads, phd, ttnn.bfloat16, cfg, bias=bias,
                        allow_m_le_n=True, site="apb")
    assert outs is not None, f"declined: {TQ.APB_REJECTS}"
    bb = [ttnn.to_torch(t) for t in outs]
    _say("  B read back")

    res = {"sig": sig_name, "device_id": device_id, "heads": heads, "padded_head_dim": phd,
           "seq": s, "chunks": []}
    for c in range(3):
        r, a, bv = ref[c], a1[c], bb[c]
        d_a1 = (a.float() - r.float()).abs().max().item()
        d_b = (bv.float() - r.float()).abs().max().item()
        rec = {"chunk": c, "max_abs_A1_vs_ref": d_a1, "max_abs_B_vs_ref": d_b,
               "A1_equals_ref": bool(torch.equal(a, r)), "B_equals_ref": bool(torch.equal(bv, r))}
        if not rec["B_equals_ref"]:
            # tile-resolved: which (head, row_tile, chan_tile) tiles differ
            mt, dt = s // TILE, phd // TILE
            rt = r.reshape(b, heads, mt, TILE, dt, TILE)
            bt = bv.reshape(b, heads, mt, TILE, dt, TILE)
            bad = []
            for h in range(heads):
                for m in range(mt):
                    for d in range(dt):
                        if not torch.equal(bt[0, h, m, :, d, :], rt[0, h, m, :, d, :]):
                            bad.append((h, m, d))
            rec["n_tiles"] = heads * mt * dt
            rec["n_bad_tiles"] = len(bad)
            rec["bad_tiles_first20"] = bad[:20]
            # is each wrong tile a CORRECT tile from somewhere else? -> address bug
            index = {}
            for h in range(heads):
                for m in range(mt):
                    for d in range(dt):
                        index.setdefault(_key(rt[0, h, m, :, d, :]), []).append((h, m, d))
            perm, alien = [], 0
            for (h, m, d) in bad:
                src = index.get(_key(bt[0, h, m, :, d, :]))
                if src:
                    perm.append({"dst": [h, m, d], "src": src[0]})
                else:
                    alien += 1
            rec["permuted_tiles"] = len(perm)
            rec["alien_tiles"] = alien
            rec["permutation_first12"] = perm[:12]
        res["chunks"].append(rec)
    return res


def _key(t):
    import hashlib
    return hashlib.sha1(t.float().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig", default="apb_trunk", choices=sorted(SIGS))
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    res = run(a.sig, a.device_id)
    path = OUT / (a.out or f"localize_{a.sig}.json")
    path.write_text(json.dumps(res, indent=1) + "\n")
    _say(json.dumps(res, indent=1)[:4000])
    _say(f"WROTE {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
