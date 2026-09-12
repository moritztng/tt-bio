"""Digest every pair-track op's UNSHARDED output, so a shard fix can be proven not to have moved it.

This branch edits code six models share. `_slab_take` gained a rank-4 pad and `_tri_att_sdpa`,
`_tri_att_sdpa_at` and the two fused-HiFi entries gained a `q_len_cfg` argument that selects the
k_chunk. Both are supposed to be inert unless a caller asks for a slab: `_slab_take`'s new branch
only runs when `shard` is set, and `q_len_cfg=None` restores the old expression exactly. That is an
argument from reading the diff, and a diff that reads inert is exactly how a shared-code regression
ships (`tt-bio-shared-diffusion-global-env-default-regression`).

So measure it. No mesh, no slab, no fabric: build the layer at seed 0 and run each of the five ops
over the WHOLE tensor, at the lengths where the SDPA config bands live, and print a sha256 of the
bytes. Run the same file against the commit before the fixes and the digests must match exactly.

    SLAB_S is not read here; the ladder is fixed so the two runs cannot silently differ.
    DIGEST_OUT names the JSON.

One chip:
    TT_VISIBLE_DEVICES=12 TT_BIO_LEASE_CARDS=12 PYTHONPATH=$PWD \\
        python3 perf/b2z2_pairchain/unsharded_digest.py
"""

import hashlib
import json
import os
import pathlib
import sys
import time

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT not in sys.path[:1]:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

OUT_PATH = os.environ.get("DIGEST_OUT", "/tmp/b2z2_unsharded_digest.json")
LENS = [int(x) for x in os.environ.get("DIGEST_LENS", "288,320,384,512").split(",")]
C_Z, C_S = 128, 384

import ttnn  # noqa: E402
from tt_bio import reference as ref  # noqa: E402
from tt_bio import tenstorrent as tt  # noqa: E402


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


dev = tt.get_device()
kernel_cls = (ttnn.types.WormholeComputeKernelConfig
              if dev.arch() == ttnn.Arch.WORMHOLE_B0
              else ttnn.types.BlackholeComputeKernelConfig)
KC = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)
log(f"device={dev} arch={dev.arch()} grid={tt.CORE_GRID_MAIN}")

RES = {"arch": str(dev.arch()), "grid": str(tt.CORE_GRID_MAIN), "lens": LENS, "digests": {},
       "sdpa_picks": {}}


def randomize_(module):
    for name, p in module.named_parameters():
        if p.dim() == 1:
            p.data = (1.0 if name.endswith("weight") else 0.0) + 0.1 * torch.randn_like(p)
        else:
            p.data = torch.randn_like(p) * (p.shape[-1] ** -0.5)


def digest(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


for S in LENS:
    torch.manual_seed(0)
    rl = ref.PairformerLayer(C_S, C_Z, 16, 0.25, 32, 4, v2=True)
    randomize_(rl)
    layer = tt.PairformerLayer(32, 4, 24, 16, True,
                               {k: v.float() for k, v in rl.state_dict().items()}, KC)
    m1 = torch.zeros(1, S)
    m1[:, :S - 32] = 1.0
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    pair_mask = up(m1[:, :, None] * m1[:, None, :])
    attn = up((1 - m1).unsqueeze(1).unsqueeze(1) * -1e9)
    z_t = torch.randn(1, S, S, C_Z, dtype=torch.float32)
    ops = {
        "trimul_start": lambda z: layer.triangle_multiplication_start(z, pair_mask),
        "trimul_end": lambda z: layer.triangle_multiplication_end(z, pair_mask),
        "triatt_start": lambda z: layer.triangle_attention_start(z, attn),
        "triatt_end": lambda z: layer.triangle_attention_end(z, attn),
        "transition_z": lambda z: layer.transition_z(z),
    }
    row = {}
    for name, fn in ops.items():
        z = up(z_t)
        o = fn(z)
        row[name] = digest(ttnn.to_torch(o))
        ttnn.deallocate(o)
        ttnn.deallocate(z)
    RES["digests"][str(S)] = row
    log(f"S={S}: " + "  ".join(f"{k}={v}" for k, v in row.items()))
    for t in (pair_mask, attn):
        ttnn.deallocate(t)

RES["sdpa_picks"] = {f"q{q}_k{k}": v for (q, k), v in tt.SDPA_CHUNK_PICKS.items()}
log(f"SDPA picks: {RES['sdpa_picks']}")
tt.cleanup()
pathlib.Path(OUT_PATH).write_text(json.dumps(RES, indent=1))
log(f"wrote {OUT_PATH}")
