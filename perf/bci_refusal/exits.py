#!/usr/bin/env python3
"""One short BindCraft 2 campaign on one chip, for issues #19 and #20.

    exits.py --mode success   one trajectory, then return
    exits.py --mode refusal   two trajectories; DRAM ballast is allocated at the second one's
                              boundary so its fold refuses with memory inherited, the way #18's
                              leak left it for the reporter
    exits.py --mode sigterm   one long trajectory, for the driver to SIGTERM mid-campaign

Prints `EVENT <name> <json>` lines on stdout; `driver.py` timestamps them and the exit. The
boundary probe reads the allocator with `ttnn.get_memory_view` directly, not through tt-bio,
so it is an independent witness for the refusal's "already held" figure.
"""
import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))
sys.path.insert(0, str(ROOT / "perf" / "bcx_predictor"))

import meter as M                                                       # noqa: E402
import bc2_state as B                                                   # noqa: E402
from tt_bio import bindcraft2                                           # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings   # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings               # noqa: E402

SHORT = ["screen_steps=4", "refine_steps=2", "anneal_steps=2", "harden_steps=1", "mutate_steps=1"]


def event(name, **kw):
    print(f"EVENT {name} {json.dumps(kw)}", flush=True)


def signals():
    out = {}
    for line in open("/proc/self/status"):
        if line.startswith(("SigCgt", "SigIgn", "SigBlk")):
            m = int(line.split()[1], 16)
            out[line.split(":")[0]] = [n for n in range(1, 32) if m >> (n - 1) & 1]
    return out


def dram():
    import ttnn
    from tt_bio import tenstorrent
    mv = ttnn.get_memory_view(tenstorrent.get_device(), ttnn.BufferType.DRAM)
    banks = int(mv.num_banks)
    return {"allocated": int(mv.total_bytes_allocated_per_bank) * banks,
            "free": int(mv.total_bytes_free_per_bank) * banks,
            "total": int(mv.total_bytes_per_bank) * banks}


BALLAST = []


def fill_to(leave):
    """Allocate DRAM until `leave` bytes are free, in chunks of at most 512 MB."""
    import ttnn
    from tt_bio import tenstorrent
    dev = tenstorrent.get_device()
    row = 1024 * 2 * 32                       # one 32-row strip of a 1024-wide bf16 tile
    while True:
        spare = dram()["free"] - leave
        if spare < row:
            return
        rows = min(spare, 512 << 20) // row * 32
        BALLAST.append(ttnn.zeros([1, 1, rows, 1024], dtype=ttnn.bfloat16,
                                  layout=ttnn.TILE_LAYOUT, device=dev,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("success", "refusal", "sigterm"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--binder", type=int, default=60)
    ap.add_argument("--leave-gb", type=float, default=1.0)
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    args = ap.parse_args()

    project = args.out
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)
    n = 2 if args.mode == "refusal" else 1
    overrides = [f"max_trajectories={n}", "campaign_seed=100", f"project_folder={project}",
                 f"binder_lengths=[{args.binder}]", "compile_next_length=0"]
    if args.mode != "sigterm":
        overrides += SHORT
    settings = cleaned_campaign_settings(
        read_settings(os.path.join(B.BC2, "examples", "pdl1.json"),
                      parse_setting_overrides(overrides)))

    M.CLOCK = M.Clock(1.0)
    M.CLOCK.start()
    from bindcraft import campaign

    cls = bindcraft2.design_model_class()
    real_grad = cls.sequence_gradients
    rounds = [0]

    def sequence_gradients(self, *a, **kw):
        if not kw.get("compile_only"):
            rounds[0] += 1
            event("round", n=rounds[0], aiclk=M.CLOCK.samples[-1][1] if M.CLOCK.samples else None)
        return real_grad(self, *a, **kw)
    cls.sequence_gradients = sequence_gradients

    with bindcraft2.campaign_predictor(trunk="device", validation="jax", checkpoints=args.params,
                                       exact=False) as build:
        event("opened", memory=build.memory.used, signals=signals(), dram=dram())
        inner = campaign.run_trajectory          # tt-bio's boundary reader, installed above
        seen = [0]

        def run_trajectory(*a, **kw):
            seen[0] += 1
            if args.mode == "refusal" and seen[0] == 2:
                before = dram()
                fill_to(int(args.leave_gb * 1e9))
                event("boundary_probe", trajectory=2, before_ballast=before, probe=dram())
            else:
                event("boundary_probe", trajectory=seen[0], probe=dram())
            return inner(*a, **kw)
        campaign.run_trajectory = run_trajectory
        t0 = time.time()
        try:
            got = bindcraft2.run_campaign(settings, project, trajectories_per_card=1,
                                          af2_weights=args.params,
                                          mpnn_weights=os.path.join(
                                              B.BC2, "bindcraft", "weights", "proteinmpnn",
                                              "weights_neutral"))
        except BaseException as exc:
            M.CLOCK.stop()
            event("raised", type=type(exc).__name__, seconds=round(time.time() - t0, 1),
                  aiclk=[c for _t, c, _l in M.CLOCK.samples][-60:])
            raise
        finally:
            campaign.run_trajectory = inner
        M.CLOCK.stop()
        event("returned", trajectories=got, seconds=round(time.time() - t0, 1),
              aiclk=[c for _t, c, _l in M.CLOCK.samples][-60:])
    print(f"=== campaign finished status={0} ===", flush=True)


if __name__ == "__main__":
    main()
