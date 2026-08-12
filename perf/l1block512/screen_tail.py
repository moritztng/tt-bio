#!/usr/bin/env python3
"""Screen the trimul output tail in situ: production against `l1norm` and row-blocked `block`.

In situ, because a standalone tail leg is the measurement this site has already been burned by
(standalone legs summed to 16.0 ms against a 15.0 ms call). Every leg is the whole
`TriangleMultiplication.__call__` on real protenix-v2 layer-0 weights at the real 512 aa pair
shape, legs interleaved one sample per round so a thermal or co-tenant drift hits all of them,
median of 7 after 3 warm rounds. Parity is `torch.equal` against production, not a tolerance.
"""
import json, statistics as st, sys, time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "stage_split_298"))
from pf_layer import build_layer  # noqa: E402

import tt_bio.tenstorrent as T  # noqa: E402
from tt_bio.tenstorrent import COMPUTE_GRID_MAIN, get_device  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 7
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
layer, c_z = build_layer(ckc)
tm = layer.triangle_multiplication_start
torch.manual_seed(0)
z = ttnn.from_torch(torch.randn(1, N, N, c_z), layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)

import os
if os.environ.get("SCREEN_LEGS") == "control":
    LEGS = [("A_off", "off", 0), ("block_R128", "block", 128),
            ("blockdram_R128", "blockdram", 128), ("blockdram_R256", "blockdram", 256),
            ("A_off_2", "off", 0)]
else:
    LEGS = [("A_off", "off", 0), ("l1norm", "l1norm", 0),
            ("block_R64", "block", 64), ("block_R128", "block", 128),
            ("block_R256", "block", 256), ("A_off_2", "off", 0)]


def set_leg(mode, rows):
    T._TRIMUL_TAIL_MODE = mode
    if rows:
        T._TRIMUL_TAIL_ROWS = rows


def once():
    return tm(z, None)


def wall(pipe=6):
    """ms per call, `pipe` calls back to back between two syncs -- tape2's own instrument."""
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [once() for _ in range(pipe)]
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) * 1e3 / pipe
    for o in outs:
        ttnn.deallocate(o)
    return ms


def synced():
    """ms for one call with a sync on both sides -- includes the full host dispatch chain."""
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    o = once()
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) * 1e3
    ttnn.deallocate(o)
    return ms


# ------------------------------------------------------------------ parity first, before timing
ref = None
parity = {}
for name, mode, rows in LEGS:
    set_leg(mode, rows)
    try:
        o = once()
        t = ttnn.to_torch(o)
        ttnn.deallocate(o)
    except Exception as e:                                                    # noqa: BLE001
        parity[name] = f"RAISED {type(e).__name__}: {str(e)[:160]}"
        print(f"{name:12s} PARITY {parity[name]}")
        continue
    if ref is None:
        ref, parity[name] = t, "reference"
    else:
        eq = bool(torch.equal(t, ref))
        md = float((t.float() - ref.float()).abs().max())
        parity[name] = f"torch.equal={eq} max_abs_diff={md:.3e}"
    print(f"{name:12s} PARITY {parity[name]}")

live = [l for l in LEGS if not str(parity.get(l[0], "")).startswith("RAISED")]

# ------------------------------------------------------------------ warm, then interleaved timing
for name, mode, rows in live:
    set_leg(mode, rows)
    for _ in range(3):
        ttnn.deallocate(once())
ttnn.synchronize_device(dev)

pipes = {n: [] for n, _, _ in live}
syncs = {n: [] for n, _, _ in live}
for r in range(REPS):
    for name, mode, rows in live:
        set_leg(mode, rows)
        pipes[name].append(wall())
        syncs[name].append(synced())
    print(f"  round {r + 1}/{REPS} " + " ".join(
        f"{n}={st.median(pipes[n]):.3f}" for n, _, _ in live))

base = st.median(pipes["A_off"])
base_s = st.median(syncs["A_off"])
rows_out = []
print(f"\n=== trimul tail screen, N={N}, grid {COMPUTE_GRID_MAIN}, median of {REPS} ===")
print(f"{'leg':12s} {'pipe ms':>9s} {'d ms':>8s} {'sync ms':>9s} {'d ms':>8s}  parity")
for name, mode, rows in live:
    p, sc = st.median(pipes[name]), st.median(syncs[name])
    rows_out.append(dict(leg=name, mode=mode, rows=rows, pipe_ms=round(p, 4),
                         pipe_delta_ms=round(p - base, 4), sync_ms=round(sc, 4),
                         sync_delta_ms=round(sc - base_s, 4),
                         pipe_all=[round(v, 4) for v in pipes[name]],
                         parity=parity[name]))
    print(f"{name:12s} {p:9.4f} {p - base:+8.4f} {sc:9.4f} {sc - base_s:+8.4f}  {parity[name]}")
aa = abs(st.median(pipes["A_off"]) - st.median(pipes["A_off_2"])) if "A_off_2" in pipes else None
print(f"\nA/A (A_off vs A_off_2, pipe): {aa:.4f} ms" if aa is not None else "")
print("s/fold at 1208 trimul calls: " + " ".join(
    f"{r['leg']}={1208 * r['pipe_delta_ms'] / 1e3:+.3f}" for r in rows_out))

OUT = dict(n=N, reps=REPS, grid=list(COMPUTE_GRID_MAIN), c_z=c_z, hidden=tm._hidden,
           trimul_calls_per_fold=1208, aa_pipe_ms=round(aa, 4) if aa is not None else None,
           legs=rows_out, parity={k: str(v) for k, v in parity.items()})
d = HERE
d.mkdir(parents=True, exist_ok=True)
(d / f"tail_screen_{N}_qb2c0{'_control' if os.environ.get('SCREEN_LEGS') == 'control' else ''}.json").write_text(json.dumps(OUT, indent=2))
print("WROTE " + str(d / f"tail_screen_{N}_qb2c0{'_control' if os.environ.get('SCREEN_LEGS') == 'control' else ''}.json"))
