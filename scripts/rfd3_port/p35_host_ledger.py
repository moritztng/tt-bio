"""Where the RFD3 step's HOST time actually goes, measured without adding a single sync.

The brief this pass inherited says "the step is 411.7 ms: 211.7 device + 200.0 host". Those two
numbers come from p29's per-op sync-bracketed profile, where every op is fenced by
ttnn.synchronize_device -- so nothing overlaps and every drain wait is charged to the host. The
shipped step at the same fixture is 225.6 ms, which is LESS than that profile's device half alone,
so "half the step is host" cannot be read off it. This script measures the split the shipped run
actually has: it adds no syncs, wraps named host functions and the ttnn entry points with
perf_counter, and prints a per-step ledger that has to add up to the step wall.

What each row means, and this is the whole point of the instrument:
  * `host.*`   -- pure torch/python work between two device barriers. Fully exposed: nothing is in
                  flight, because whatever was enqueued has already been drained.
  * `tt.to_torch` -- a blocking read. Its wall is the DEVICE time of everything enqueued ahead of
                  it that had not finished yet, i.e. exposed device time, not host cost.
  * every other `tt.*` -- host-side dispatch of a non-blocking enqueue. Overlaps with the device.
  * `residual` -- step wall minus the sum. Python glue plus anything unwrapped.

Arms:
  --plain            no instrumentation at all: the honest wall, and the inflation the ledger costs.
  --freeze-indices   MEASUREMENT ONLY, produces a WRONG design: reuse step 0's neighbour graph for
                     every later step. This is the end-to-end upper bound on every attn_indices
                     lever (host rewrite, device port, caching) -- if the wall does not move, the
                     work is overlapped or small and the lever is capped there, whatever the
                     op-level profile says. Never a shippable arm; see rfd3-p24-measure-before-
                     porting-attn-indices for the three passes that assumed this cost was the lever
                     without ever bounding it end to end.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402

import tt_bio.rfd3.model as R  # noqa: E402
from tt_bio.rfd3.design import build_diffusion_module, build_token_initializer  # noqa: E402
from tt_bio.rfd3.featurize import featurize  # noqa: E402
from tt_bio.rfd3.input import InputSpecification  # noqa: E402
from tt_bio.rfd3.sampler import RFD3Sampler  # noqa: E402

ACC: dict[str, float] = defaultdict(float)
CNT: dict[str, int] = defaultdict(int)
SNAP: list[tuple[float, dict, dict]] = []   # (step wall, ACC snapshot, CNT snapshot)

TT_OPS = ("linear", "matmul", "add", "add_", "multiply", "multiply_", "subtract", "softmax",
          "typecast", "scatter", "embedding", "reshape", "permute", "pad", "to_layout", "concat",
          "rms_norm", "layer_norm", "sigmoid", "clone", "full", "from_torch", "to_torch",
          "deallocate", "transpose", "repeat", "sum", "mean", "exp", "where", "zeros", "arange")


def _wrap(mod, name, label):
    fn = getattr(mod, name, None)
    if fn is None or not callable(fn):
        return
    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            ACC[label] += time.perf_counter() - t0
            CNT[label] += 1
    setattr(mod, name, w)


def instrument():
    for n in TT_OPS:
        _wrap(ttnn, n, f"tt.{n}")
    for n in ("_scatter_mean", "_dense_attention_mask", "_sparse_qk_host", "_sparse_pair_gather",
              "_sparse_attn_index", "_scaled_distogram_bins", "_attention_index_prefix",
              "_extend_with_neighbours", "_grouping_indices"):
        _wrap(R, n, f"host.{n}")
    # attn_indices is called with two different k -- 128 for the atoms, 32 for the DiT token graph
    orig = R._create_attention_indices
    def w(f, X_L, tok_idx, n_keys, n_seq):
        t0 = time.perf_counter()
        try:
            return orig(f, X_L, tok_idx, n_keys, n_seq)
        finally:
            lbl = f"host.attn_indices(k={n_keys})"
            ACC[lbl] += time.perf_counter() - t0
            CNT[lbl] += 1
    R._create_attention_indices = w


def freeze_indices():
    """Reuse the first call's result for every later call at the same (k, L). WRONG DESIGN."""
    orig = R._create_attention_indices
    cache: dict = {}
    def w(f, X_L, tok_idx, n_keys, n_seq):
        key = (n_keys, n_seq, len(tok_idx))
        if key not in cache:
            cache[key] = orig(f, X_L, tok_idx, n_keys, n_seq)
        return cache[key]
    R._create_attention_indices = w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default="scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb")
    ap.add_argument("--contig", default="A1-10,230,A31-40")
    ap.add_argument("--ckpt", default="/home/ttuser/.boltz/rfd3/weights")
    ap.add_argument("--num_timesteps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--freeze-indices", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    if a.freeze_indices:
        freeze_indices()
    if not a.plain:
        instrument()

    spec = InputSpecification.from_dict({"input": a.pdb, "contig": a.contig})
    spec.validate()
    f = featurize(a.pdb, spec)
    cap = Path(a.ckpt)
    ti_w = torch.load(cap / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True)
    dm_w = torch.load(cap / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti_w)
    dev_dm = build_diffusion_module(dm_w)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    L = init["Q_L_init"].shape[0]
    coord0 = f["motif_pos"].float().unsqueeze(0) if "motif_pos" in f else torch.zeros(1, L, 3)
    ACC.clear(); CNT.clear()

    # Per-step snapshots. A 40-step run's MEAN is meaningless here: step 1 registers ~3400
    # programs and cost 3.9 s in the first version of this probe (and 40 s on a cold
    # ~/.cache/ttnn -- rfd3-baseline-seed-cold-cache-trap). Every number below is the MEDIAN
    # over the warm steps, so the compile tail cannot masquerade as dispatch cost.
    # Patch the CLASS: `diffusion_module(...)` resolves __call__ on the type, so an instance
    # attribute is never consulted.
    cls = type(dev_dm)
    dm_call = cls.__call__
    def stepped(self, *ar, **kw):
        t0 = time.perf_counter()
        try:
            return dm_call(self, *ar, **kw)
        finally:
            SNAP.append((time.perf_counter() - t0, dict(ACC), dict(CNT)))
    cls.__call__ = stepped

    sampler = RFD3Sampler(num_timesteps=a.num_timesteps)
    t0 = time.perf_counter()
    with torch.no_grad():
        X, traj = sampler.sample(dev_dm, 1, L, coord0, f, init,
                                 f["is_motif_atom_with_fixed_coord"],
                                 generator=torch.Generator().manual_seed(a.seed))
    wall = time.perf_counter() - t0
    steps = len(traj)
    per = wall / steps * 1e3
    # median per-step deltas
    med_ms: dict[str, float] = {}
    med_cnt: dict[str, float] = {}
    if len(SNAP) > 2:
        keys = set().union(*[s[1].keys() for s in SNAP])
        for k in keys:
            d = [(SNAP[i][1].get(k, 0.0) - SNAP[i - 1][1].get(k, 0.0)) * 1e3
                 for i in range(2, len(SNAP))]
            med_ms[k] = statistics.median(d)
            dc = [SNAP[i][2].get(k, 0) - SNAP[i - 1][2].get(k, 0) for i in range(2, len(SNAP))]
            med_cnt[k] = statistics.median(dc)
    step_walls = [s[0] * 1e3 for s in SNAP]
    med_step = statistics.median(step_walls[2:]) if len(step_walls) > 2 else per
    print(f"\n[step wall] first={step_walls[0]:.0f} second={step_walls[1]:.0f} "
          f"median(warm)={med_step:.1f} min={min(step_walls[2:]):.1f} max={max(step_walls[2:]):.1f} ms",
          flush=True)
    print(f"\n[ledger] L={L} atoms  I={int(f['atom_to_token_map'].max()) + 1}  steps={steps}  "
          f"wall={wall:.2f} s  {per:.1f} ms/step"
          f"{'  [PLAIN]' if a.plain else ''}{'  [FROZEN INDICES -- wrong design]' if a.freeze_indices else ''}",
          flush=True)

    base = med_step if med_ms else per
    rows = sorted(med_ms.items(), key=lambda kv: -kv[1]) if med_ms else \
        [(k, v / steps * 1e3) for k, v in sorted(ACC.items(), key=lambda kv: -kv[1])]
    acc_ms = sum(v for _, v in rows)
    print(f"{'row (median warm step)':34s} {'ms/step':>8s} {'% step':>7s} {'calls/step':>11s}")
    for k, ms in rows:
        if ms < 0.05:
            continue
        print(f"{k:34s} {ms:8.3f} {100 * ms / base:6.1f}% {med_cnt.get(k, 0):11.1f}")
    print(f"{'SUM of rows':34s} {acc_ms:8.3f} {100 * acc_ms / base:6.1f}% "
          f"{sum(med_cnt.values()):11.1f}")
    print(f"{'residual (python glue etc)':34s} {base - acc_ms:8.3f} {100 * (base - acc_ms) / base:6.1f}%")
    drain = med_ms.get("tt.to_torch", 0.0)
    host = sum(v for k, v in med_ms.items() if k.startswith("host."))
    disp = acc_ms - drain - host
    nested = sum(v for k, v in med_ms.items() if k.startswith("host."))
    print(f"\n  exposed device (tt.to_torch drains) {drain:8.3f} ms/step {100 * drain / base:5.1f}%")
    print(f"  host.* (INCLUDES nested tt.* time)  {host:8.3f} ms/step {100 * host / base:5.1f}%")
    print(f"  ttnn dispatch (all other tt.*)      {disp:8.3f} ms/step {100 * disp / base:5.1f}%")
    print(f"  ttnn ops dispatched per step        {sum(v for k, v in med_cnt.items() if k.startswith('tt.')):8.1f}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({
            "atoms": L, "steps": steps, "wall_s": wall, "ms_per_step": per,
            "plain": a.plain, "freeze_indices": a.freeze_indices,
            "median_warm_step_ms": med_step, "step_walls_ms": step_walls,
            "rows_ms_per_step": med_ms or {k: v / steps * 1e3 for k, v in ACC.items()},
            "calls_per_step": med_cnt or {k: CNT[k] / steps for k in CNT},
            "drain_ms": drain, "host_ms": host, "dispatch_ms": disp,
        }, indent=2))
        print(f"[done] {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
