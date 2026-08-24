"""S2: itemise DiffusionTokenEncoder.run_device at the page fixture, op by op.

Leg 2 of `rfd3-close-the-page-gap`. The token encoder is 39.9 % of the step at 6051 atoms and
the decomposition that found it (`state/rfd3-optimize-on-fixture.md` §1.2) attributes drain to
regions, not to ops, so there is no mechanism yet. This replaces `run_device` with a line-for-
line copy that puts `ttnn.synchronize_device` on both sides of every region.

Two rules on reading the output, both from the knowledgebase:

  * `tt-bio-isolated-op-timing-oversync-inflates-cost` -- the sum of sync-bracketed rows
    overshoots the region's true wall by up to ~2x, because each sync drains work that would
    otherwise overlap. Use the RANKING and the RATIOS. The unsynced region wall is printed
    alongside for exactly this reconciliation.
  * `roofline-roof-must-be-measured-not-asserted` -- the GB/s column is against the MEASURED
    385 GB/s clone roof for this card (`state/rfd3-host-half.md` §3), not the 435 GB/s
    datasheet number. A row above the roof is a byte-model error, not a result.

Batch is 1, because that is what ships at this fixture after the speed cap landed.

    ~/.coworker/scripts/benchlock.sh rfd3-close-the-page-gap -- env TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-close-the-page-gap PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p46_token_encoder_itemise.py
"""
import collections
import json
import os
import pathlib
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design                      # noqa: E402
from tt_bio.rfd3 import model as M                                 # noqa: E402

FIXTURE = pathlib.Path("perf/dsfix/fixtures/rfd3_R4.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
# argv[1] overrides the artifact path, so a re-run under different defaults does not
# overwrite the record of the run that justified a landed decision.
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p46/token_encoder_itemise.json")
STEPS = 8                 # 2 discarded as cold, 6 counted; ranking does not need 200
WARM_CALLS = 4            # run_device is called twice per step, so 2 steps of warmup
ROOF_GBS = 385.0          # MEASURED clone roof for this card, not the datasheet 435

T = collections.OrderedDict()      # region -> [total_s, n]
CALLS = [0]
REGION_WALL = [0.0]                # unsynced wall of the whole region, for reconciliation


def _add(name, dt):
    if CALLS[0] < WARM_CALLS:
        return
    e = T.setdefault(name, [0.0, 0])
    e[0] += dt
    e[1] += 1


class timed:
    """sync, start, body, sync, stop -- `ttnn-sync-before-every-timed-region`."""

    def __init__(self, name, dev):
        self.name, self.dev = name, dev

    def __enter__(self):
        ttnn.synchronize_device(self.dev)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        ttnn.synchronize_device(self.dev)
        _add(self.name, time.perf_counter() - self.t0)
        return False


def instrumented_run_device(self, R_L_ca, S_init_I, Z_init_II, D_II_self=None):
    dev, ckc, dt = self.device, self.compute_kernel_config, self.dtype
    f32 = self.fp32_residual
    B, I = R_L_ca.shape[0], R_L_ca.shape[1]
    ttnn.synchronize_device(dev)
    t_region = time.perf_counter()

    with timed("s.upload+transition_1", dev):
        s_in = S_init_I
        if s_in.ndim == 2:
            s_in = s_in.unsqueeze(0).expand(B, -1, -1).contiguous()
        s = M._tt(s_in, dev, dt)
        if f32:
            s = ttnn.typecast(s, ttnn.float32, memory_config=s.memory_config())
        for tr in self.transition_1:
            sc = ttnn.typecast(s, dt, memory_config=s.memory_config()) if f32 else s
            upd = tr(sc)
            s = ttnn.add(s, ttnn.typecast(upd, ttnn.float32, memory_config=upd.memory_config())) if f32 \
                else ttnn.add(s, upd)

    with timed("host._scaled_distogram_bins", dev):
        bins = M._scaled_distogram_bins(R_L_ca, sigma_data=self.sigma_data, n_bins=self.N_BINS)

    # The shipped path, not the pre-_CONCAT_ALIGNED one this file used to copy: ONE combined
    # 160-wide one-hot, a 128+160 concat where both pieces are tile multiples, and a slice back
    # to 258 so rms_norm averages over the right count.
    with timed("_combined_onehot_dev[B,I,I,160]", dev):
        dself = self._combined_onehot_dev(bins, D_II_self, B, I)

    with timed("z.upload+_batched", dev):
        z_in = Z_init_II.unsqueeze(0) if Z_init_II.ndim == 3 else Z_init_II
        z = self._batched(M._tt_cached(z_in, dev, dt), B)

    with timed("ttnn.concat 128+160 -> 288", dev):
        wide = ttnn.concat([z, dself], dim=-1)
    ttnn.deallocate(dself)
    with timed("ttnn.slice 288 -> 258", dev):
        zcat = ttnn.slice(wide, [0, 0, 0, 0], [B, I, I, 2 * self.N_BINS + self.C_Z])
    ttnn.deallocate(wide)
    with timed("ttnn.rms_norm(258)", dev):
        z = ttnn.rms_norm(zcat, weight=self.process_z_n, epsilon=1e-6, compute_kernel_config=ckc)
    with timed("ttnn.linear 258->128", dev):
        z = ttnn.linear(z, self.process_z_w, compute_kernel_config=ckc, dtype=dt,
                        core_grid=M.CORE_GRID_MAIN)
    ttnn.deallocate(zcat)
    if f32:
        z = ttnn.typecast(z, ttnn.float32, memory_config=z.memory_config())

    with timed("transition_2 x2", dev):
        for tr in self.transition_2:
            zc = ttnn.typecast(z, dt, memory_config=z.memory_config()) if f32 else z
            upd = tr(zc)
            z = ttnn.add(z, ttnn.typecast(upd, ttnn.float32, memory_config=upd.memory_config())) if f32 \
                else ttnn.add(z, upd)

    for i, blk in enumerate(self.pairformer_stack):
        with timed("pairformer[%d]" % i, dev):
            s, z = blk(s, z)

    ttnn.synchronize_device(dev)
    if CALLS[0] >= WARM_CALLS:
        REGION_WALL[0] += time.perf_counter() - t_region
    CALLS[0] += 1
    return s, z


def _onehot_itemised(self, bins, B, I, dev, dt, tag):
    """UNUSED since the shipped path moved to _combined_onehot_dev. Kept for the RFD3_CONCAT_ALIGNED=0
    arm only; the instrumented path above no longer calls it.

    The prior plan folded the embedding and the layout conversion together. Split them:
    `ttnn-untilize-single-core-fallback` says a layout conversion can silently fall back to
    one core, and a 65-wide last dim tilizes to 96."""
    eye = self._const.get(("eye", dt))
    if eye is None:
        eye = M._tt(torch.eye(self.N_BINS), dev, dt)
        self._const[("eye", dt)] = eye
    with timed("%s.idx upload" % tag, dev):
        idx = M._tt_idx(bins, dev)
    with timed("%s.ttnn.embedding" % tag, dev):
        oh = ttnn.embedding(idx, eye, layout=ttnn.ROW_MAJOR_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
    with timed("%s.ttnn.reshape" % tag, dev):
        oh = ttnn.reshape(oh, (B, I, I, self.N_BINS))
    with timed("%s.to_layout(TILE)" % tag, dev):
        oh = ttnn.to_layout(oh, ttnn.TILE_LAYOUT)
    return oh


def byte_model(B, I):
    """Bytes moved per call, DRAM read + write, bf16. Tile layout pads a 65-wide last dim to
    96 and a 258-wide one to 288; that padding is real traffic and is counted."""
    p = I * I
    b = B * p
    return {
        "_combined_onehot_dev[B,I,I,160]": b * 4 + b * 160 * 2 + b * 160 * 2 * 2,
        "ttnn.concat 128+160 -> 288": b * (128 + 160) * 2 + b * 288 * 2,
        "ttnn.slice 288 -> 258": b * 288 * 2 + b * 288 * 2,
        "ttnn.rms_norm(258)": b * 288 * 2 * 2,
        "ttnn.linear 258->128": b * 288 * 2 + b * 128 * 2,
    }


def main():
    M.DiffusionTokenEncoder.run_device = instrumented_run_device
    specs = json.loads(FIXTURE.read_text())
    out_dir = "/tmp/rfd3_p46"
    os.system("rm -rf %s" % out_dir)
    t0 = time.perf_counter()
    rfd3_design.run_design(specs, out_dir, checkpoint_dir=CKPT, from_pdb=True,
                           num_timesteps=STEPS, seed=42, num_designs=1, batch_size=8,
                           verbose=True)
    wall = time.perf_counter() - t0

    counted_calls = CALLS[0] - WARM_CALLS
    steps = counted_calls / 2.0
    B, I = 1, 685
    bm = byte_model(B, I)
    rows = []
    for name, (tot, n) in T.items():
        ms_call = 1000 * tot / max(1, n)
        ms_step = 1000 * tot / steps
        by = bm.get(name)
        gbs = (by / (tot / max(1, n))) / 1e9 if by else None
        rows.append({"region": name, "n": n, "ms_per_call": round(ms_call, 3),
                     "ms_per_step": round(ms_step, 3),
                     "MB_per_call": round(by / 1e6, 1) if by else None,
                     "GB_s": round(gbs, 1) if gbs else None,
                     "pct_of_roof": round(100 * gbs / ROOF_GBS, 1) if gbs else None})
    rows.sort(key=lambda r: -r["ms_per_step"])
    synced_total = sum(r["ms_per_step"] for r in rows)
    unsynced = 1000 * REGION_WALL[0] / steps

    print("\n=== S2 token encoder, R4 6051 atoms, B=%d I=%d, %d calls counted (%.1f steps) ===" %
          (B, I, counted_calls, steps))
    print("%-30s %8s %8s %9s %8s %7s" % ("region", "ms/call", "ms/step", "MB/call", "GB/s", "%roof"))
    for r in rows:
        print("%-30s %8.3f %8.3f %9s %8s %7s" % (
            r["region"], r["ms_per_call"], r["ms_per_step"],
            r["MB_per_call"] if r["MB_per_call"] else "-",
            r["GB_s"] if r["GB_s"] else "-",
            r["pct_of_roof"] if r["pct_of_roof"] else "-"))
    print("%-30s %8s %8.3f" % ("SUM of synced rows", "", synced_total))
    print("%-30s %8s %8.3f   <- reconcile against this, not the sum" %
          ("run_device wall, unsynced", "", unsynced))
    print("oversync inflation: %.2fx" % (synced_total / unsynced))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "fixture": str(FIXTURE), "atoms": 6051, "tokens": I, "batch": B,
        "steps_counted": steps, "calls_counted": counted_calls, "num_timesteps": STEPS,
        "roof_GB_s_measured": ROOF_GBS, "rows": rows,
        "synced_sum_ms_per_step": round(synced_total, 3),
        "unsynced_region_ms_per_step": round(unsynced, 3),
        "oversync_inflation": round(synced_total / unsynced, 3),
        "total_wall_s": round(wall, 1),
        "host": "qb2", "card": 0, "ttnn": "0.68.0", "torch": torch.__version__,
    }, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
