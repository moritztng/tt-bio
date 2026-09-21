#!/usr/bin/env python3
"""Does the head-major map assume the right channel order? Checked end to end, on the CPU.

`tile_map.py` proves the destination map is the permutation `reshape(B,S,H,D).permute(0,2,1,3)`
performs. That is independent of the values, so it cannot see the other half of the claim: that N
tile index `tidx` of chunk `c` really is (head, channel) = (tidx / DT, tidx % DT) of q, k or v.
That comes from how `AttentionPairBias.__init__` lanes the fused weight, and if it were wrong the
lever would be silently, numerically wrong rather than merely mis-addressed.

Two exact checks that compose, and neither has a tolerance:

  LANES   the laned weight's column `c * n_heads * padded_head_dim + h * padded_head_dim + d` is
          `torch.equal` to the unlaned checkpoint's row `(c * n_heads + h) * head_dim + d` for every
          (c, h, d), and every pad column and pad bias element is exactly zero. That is the channel
          order the map assumes, read off the module's own relaning rather than assumed.
  VALUES  project, add the bias, tile, apply the writer's map, de-tile, and compare with the same
          projection sliced at those column indices and permuted to (batch, head, seq, channel).
          `torch.equal`, because both sides are the same numbers moved differently.

An einsum over the unlaned per-head weights was tried as the reference first and is NOT usable: it
is a different contraction routine, so it disagrees with `x @ w` by 2.2e-15 in fp64 at c_in = 384
and by exactly 0 at 768. That is BLAS blocking, not the lever, and a reference that wobbles with the
input width cannot carry a bit-exactness claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tile_map import TILE, macro_ids, tile_buffer, untile  # noqa: E402

# (name, batch, seq, c_in, n_heads, head_dim) -- head_dim LOGICAL; the padding is derived.
CASES = [
    ("dit_token", 1, 512, 768, 16, 48),     # pads 48 -> 64, DT 2, the 4800-prog signature
    ("apb_trunk", 1, 512, 384, 16, 32),     # no padding, DT 1, the 264-prog signature
]


def lane(qkv_w, q_b, n_heads, head_dim):
    """`AttentionPairBias.__init__`'s relaning, tenstorrent.py:7726-7744, transcribed."""
    pad = -head_dim % TILE
    phd = head_dim + pad
    w = qkv_w.reshape(3 * n_heads, head_dim, -1)
    w = torch.nn.functional.pad(w, (0, 0, 0, pad), mode="constant", value=0)
    w = w.reshape(3 * n_heads * phd, -1)
    b = q_b.reshape(n_heads, head_dim)
    b = torch.nn.functional.pad(b, (0, pad), mode="constant", value=0)
    b = b.reshape(n_heads * phd)
    b = torch.cat([b, torch.zeros(2 * n_heads * phd, dtype=b.dtype)])
    return w.t().contiguous(), b


def check(name, batch, seq, c_in, n_heads, head_dim):
    phd = head_dim + (-head_dim % TILE)
    torch.manual_seed(0)
    # Per-head reference weights, [3 * n_heads * head_dim, c_in] as the checkpoint stores them.
    qkv_w = torch.randn(3 * n_heads * head_dim, c_in, dtype=torch.float64) * 0.02
    q_b = torch.randn(n_heads * head_dim, dtype=torch.float64) * 0.02
    x = torch.randn(batch, seq, c_in, dtype=torch.float64)

    w, b = lane(qkv_w, q_b, n_heads, head_dim)
    out = (x.reshape(batch * seq, c_in) @ w + b)              # the projection, laned width

    mt, dt = seq // TILE, phd // TILE
    d1 = n_heads * dt
    ids = macro_ids(mt, None if dt == 1 else dt, batch * mt, d1)

    # LANES: the channel order, straight off the relaning.
    lanes_ok = True
    for c in range(3):
        for h in range(n_heads):
            base = (c * n_heads + h) * phd
            lanes_ok &= torch.equal(w[:, base:base + head_dim],
                                    qkv_w[(c * n_heads + h) * head_dim:
                                          (c * n_heads + h) * head_dim + head_dim].t())
            if phd > head_dim:
                lanes_ok &= bool((w[:, base + head_dim:base + phd] == 0).all())
                lanes_ok &= bool((b[base + head_dim:base + phd] == 0).all())
            if c == 0:
                lanes_ok &= torch.equal(b[base:base + head_dim],
                                        q_b[h * head_dim:(h + 1) * head_dim])
            else:
                lanes_ok &= bool((b[base:base + phd] == 0).all())

    # VALUES: the writer's destination against the same projection sliced and permuted.
    ok, worst = True, 0.0
    for c in range(3):
        chunk = out[:, c * d1 * TILE:(c + 1) * d1 * TILE].contiguous()
        src = tile_buffer(chunk)
        dst = torch.empty_like(src)
        for row in range(batch * mt):
            for tidx in range(d1):
                dst[ids[row * d1 + tidx]] = src[row * d1 + tidx]
        got = untile(dst, batch * n_heads * seq, phd).reshape(batch, n_heads, seq, phd)
        ref = torch.stack([chunk[:, h * phd:(h + 1) * phd] for h in range(n_heads)], 0)
        ref = ref.reshape(n_heads, batch, seq, phd).permute(1, 0, 2, 3).contiguous()
        ok &= torch.equal(got, ref)
        worst = max(worst, float((got - ref).abs().max()))
        # The pad lanes reach SDPA, so a non-zero there would add bias to every q.k.
        if phd > head_dim:
            ok &= bool((got[..., head_dim:] == 0).all())

    return {"case": name, "batch": batch, "seq": seq, "c_in": c_in, "n_heads": n_heads,
            "head_dim": head_dim, "padded_head_dim": phd, "HEAD_MAJOR_DT": dt,
            "lanes_torch_equal": bool(lanes_ok), "values_torch_equal": bool(ok),
            "max_abs": worst,
            "pad_lanes": phd - head_dim}


def main() -> int:
    rows = [check(*c) for c in CASES]
    for r in rows:
        print(f"{r['case']:<11} head_dim {r['head_dim']} -> {r['padded_head_dim']} "
              f"(pad {r['pad_lanes']}) DT={r['HEAD_MAJOR_DT']} "
              f"lanes={r['lanes_torch_equal']} values={r['values_torch_equal']} "
              f"max_abs={r['max_abs']:.3e}")
    ok = all(r["lanes_torch_equal"] and r["values_torch_equal"] for r in rows)
    out = Path(__file__).resolve().parent / "channel_order.json"
    out.write_text(json.dumps({"all_pass": ok, "cases": rows,
                               "reference": "laned columns vs the unlaned checkpoint rows, and "
                                            "the projection sliced at those columns, fp64, "
                                            "torch.equal on both"}, indent=1)
                   + "\n")
    print(f"\n{'PASS' if ok else 'FAIL'} -- {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
