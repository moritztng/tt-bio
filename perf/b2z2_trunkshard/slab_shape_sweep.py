"""Which token lengths does an i-axis row slab reproduce bit-exactly, and which does it not.

`block_split_bitexact.py` found that the split is exact at 512 and 256 aa and not at 288 or 320,
and that exactly one op moves: `triangle_attention_end`. Its qkv and gate projections, and the
fused SDPA behind them, are routed and blocked by shape, and a slab halves M -- `tenstorrent.py`
says so at line 6820 and names a parity script as the arbiter of which shapes are affected. One
shape is an anecdote. This walks the ladder and produces the rule.

Per length: build the real ops once, run each whole, run it as two tile-aligned row slabs,
concatenate, compare with `torch.equal`. The weights do not depend on the token length, so the
layer is built once and every length reuses it.

    python3 perf/b2z2_trunkshard/slab_shape_sweep.py

Env: SWEEP_LENS (comma-separated, default the 128..640 ladder), SWEEP_OUT.
"""

import pathlib
import sys

_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _ROOT not in sys.path[:1]:
    sys.path.insert(0, _ROOT)

import json
import os
import time

import torch

import ttnn
from tt_bio import reference as ref
from tt_bio import row_shard
from tt_bio import tenstorrent as tt

LENS = [int(x) for x in os.environ.get(
    "SWEEP_LENS", "128,160,192,224,256,288,320,384,448,512,576,640").split(",")]
OUT = os.environ.get("SWEEP_OUT", "/tmp/b2z2_slab_shape_sweep.json")
C_Z, C_S = 128, 384


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def randomize_(module):
    for name, p in module.named_parameters():
        if p.dim() == 1:
            p.data = (1.0 if name.endswith("weight") else 0.0) + 0.1 * torch.randn_like(p)
        else:
            p.data = torch.randn_like(p) * (p.shape[-1] ** -0.5)


dev = tt.get_device()
kernel_cls = (ttnn.types.WormholeComputeKernelConfig
              if dev.arch() == ttnn.Arch.WORMHOLE_B0
              else ttnn.types.BlackholeComputeKernelConfig)
KC = kernel_cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)
rl = ref.PairformerLayer(C_S, C_Z, 16, 0.25, 32, 4, v2=True)
randomize_(rl)
layer = tt.PairformerLayer(32, 4, 24, 16, True, {k: v.float() for k, v in rl.state_dict().items()},
                           KC)
log(f"device={dev} arch={dev.arch()} lengths={LENS}")

rows = []
for S in LENS:
    try:
        bounds = row_shard.row_shard_bounds(S, 2)
    except ValueError as e:
        log(f"S={S}: {e}")
        continue
    torch.manual_seed(1)
    z_t = torch.randn(1, S, S, C_Z, dtype=torch.float32)
    m1 = torch.zeros(1, S)
    m1[:, :S - 32] = 1.0

    def up(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    mask = up(m1[:, :, None] * m1[:, None, :])
    am = up((1 - m1).unsqueeze(1).unsqueeze(1) * -1e9)
    ops = {
        "trimul_start": lambda z, sl: layer.triangle_multiplication_start(z, mask, row_slab=sl),
        "trimul_end": lambda z, sl: layer.triangle_multiplication_end(z, mask, row_slab=sl),
        "triatt_start": lambda z, sl: layer.triangle_attention_start(z, am, row_slab=sl),
        "triatt_end": lambda z, sl: layer.triangle_attention_end(z, am, row_slab=sl),
        "transition_z": lambda z, sl: layer.transition_z(z, row_slab=sl),
    }
    row = {"S": S, "tiles": S // 32, "slabs": [list(b) for b in bounds],
           "slab_tiles": [(r1 - r0) // 32 for r0, r1 in bounds],
           "balance_ceiling": row_shard.row_shard_ceiling(S, 2), "ops": {}}
    def picks():
        """Every shape-routed counter, so a divergence names the leg that moved."""
        out = {}
        for mod, attr in (("_triatt_qkv", "STATS"), ("_triatt_sdpa", "STATS"),
                          ("_reblock", "STATS_GATED")):
            m = getattr(tt, mod, None)
            if m is not None and hasattr(m, attr):
                v = getattr(m, attr)
                out[mod] = dict(v) if isinstance(v, dict) else list(v)
        # these are keyed by shape tuples; JSON needs strings
        out["sdpa_picks"] = {str(k): v for k, v in tt.SDPA_CHUNK_PICKS.items()}
        out["sdpa_routes"] = {str(k): v for k, v in tt.SDPA_ROUTE_COUNTS.items()}
        return out

    def moved(a, b):
        out = {}
        for k in set(a) | set(b):
            x, y = a.get(k), b.get(k)
            if x == y:
                continue
            if isinstance(y, dict):
                d = {}
                for kk in set(y) | set(x or {}):
                    yv, xv = y.get(kk), (x or {}).get(kk)
                    if yv == xv:
                        continue
                    # counts subtract; anything else (a pick is a list of block dims) is reported
                    d[kk] = (yv - (xv or 0)) if isinstance(yv, (int, float)) else [xv, yv]
            else:
                d = [bb - aa for aa, bb in zip(x or [0] * len(y), y)]
            if d:
                out[k] = d
        return out

    for name, fn in ops.items():
        zt = up(z_t)
        try:
            p0 = picks()
            whole = ttnn.to_torch(fn(zt, None))
            p1 = picks()
            parts = []
            for b in bounds:
                o = fn(zt, b)
                parts.append(ttnn.to_torch(o))
                ttnn.deallocate(o)
            p2 = picks()
            joined = torch.cat(parts, dim=1)
            d = (joined.float() - whole.float()).abs()
            row["ops"][name] = {
                "bit_exact": bool(joined.shape == whole.shape and torch.equal(joined, whole)),
                "max_abs_diff": d.max().item(), "mean_abs_diff": d.mean().item(),
                "frac_differing": float((d > 0).sum()) / d.numel(),
                "ref_abs_max": whole.float().abs().max().item(),
                "picks_whole": moved(p0, p1), "picks_slab": moved(p1, p2),
            }
        except Exception as e:                                              # noqa: BLE001
            row["ops"][name] = {"bit_exact": None, "error": f"{type(e).__name__}: {e}"[:200]}
        finally:
            ttnn.deallocate(zt)
    ttnn.deallocate(mask)
    ttnn.deallocate(am)
    bad = [n for n, r in row["ops"].items() if r.get("bit_exact") is not True]
    row["all_bit_exact"] = not bad
    log(f"S={S:4d} ({S // 32:2d} tiles -> {row['slab_tiles']}): "
        + ("bit-exact" if not bad else "NOT bit-exact: " + ", ".join(
            f"{n} {row['ops'][n].get('max_abs_diff', row['ops'][n].get('error'))}" for n in bad)))
    rows.append(row)

pathlib.Path(OUT).write_text(json.dumps({"arch": str(dev.arch()), "lengths": rows}, indent=2))
log(f"wrote {OUT}")
ok = [r["S"] for r in rows if r["all_bit_exact"]]
bad = [r["S"] for r in rows if not r["all_bit_exact"]]
print(f"BIT-EXACT at {ok}")
print(f"NOT bit-exact at {bad}")
