#!/usr/bin/env python3
"""One real BindCraft 2 campaign through `bindcraft2.run_campaign`, N trajectories on one card.

Not a round harness. `perf/bcx_p10_duotraj/duo_round.py` stops each trajectory after nine
gradient rounds, stubs BindCraft 2's cross-worker compile lock out and never reaches MPNN,
validation or acceptance. This runs the whole thing through the user-facing entry: every design
stage, the sequence redesign, the validation ensemble and the accepted-design writing, with the
compile lock LIVE (`JAX_COMPILATION_CACHE_DIR` is set by the launcher, which is what arms it).

It measures rather than stubs, so the only instrument is a timestamp per gradient round per
trajectory plus the card's own clock, written out as they happen.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
sys.path.insert(0, str(_ROOT / "perf" / "bcx_round"))
sys.path.insert(0, str(_ROOT / "perf" / "bcx_predictor"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import meter as M                                                       # noqa: E402
import bc2_state as B                                                   # noqa: E402
from tt_bio import bindcraft2, duotraj                                  # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings   # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings               # noqa: E402


def host_memory() -> dict:
    out = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS:", "VmHWM:")):
            out[line.split(":")[0].lower()] = int(line.split()[1]) * 1024
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            out["mem_available"] = int(line.split()[1]) * 1024
    return out


class Rounds:
    """A timestamp at the top of every gradient round, per trajectory.

    Written to disk at every round, because a run that is OOM-killed at round 200 still has to
    say what it measured up to there.
    """

    def __init__(self, path):
        self.path, self.lock, self.rows = path, threading.Lock(), []

    def mark(self, slot):
        with self.lock:
            self.rows.append({"t": time.time(), "slot": slot,
                              "round": sum(1 for r in self.rows if r["slot"] == slot) + 1,
                              "load1": round(os.getloadavg()[0], 2), **host_memory()})
            if len(self.rows) % 5 == 0:
                self.dump()

    def dump(self):
        tmp = f"{self.path}.partial"
        pathlib.Path(tmp).write_text(json.dumps(self.rows))
        os.replace(tmp, self.path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", type=int, default=2)
    ap.add_argument("--binder", type=int, default=146)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--params", default="/home/moritz/bcx_shipped/af2_params")
    ap.add_argument("--validation", default="device", choices=("jax", "device"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="K=V")
    args = ap.parse_args()

    n = args.trajectories
    project = args.out
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)
    overrides = [f"campaign_seed={args.seed}", f"max_trajectories={n}",
                 f"project_folder={project}",
                 # Both trajectories at one binder length, so the amortised round is comparable
                 # with the harness figure and not a reading of two different shapes.
                 f"binder_lengths=[{','.join([str(args.binder)] * n)}]",
                 "compile_next_length=0"] + args.sets
    settings = cleaned_campaign_settings(
        read_settings(os.path.join(B.BC2, "examples", "pdl1.json"),
                      parse_setting_overrides(overrides)))

    # The campaign's gating lever configuration, the same one `bcx-p10-tritraj` measured.
    os.environ["TT_BIO_TRIATT_TAPED_SDPA"] = "0"
    os.environ["TT_BIO_SDPA_OWN_FORWARD"] = "1"
    os.environ["TT_BIO_TAPED_KERNELS"] = "tri_att_sdpa_hifi,rne_add"
    os.environ["TT_BIO_TRIATT_DIVIDING_K"] = "1"
    from tt_bio import tenstorrent as _tn
    from tt_bio import triatt_bw as _tbw
    from tt_bio.af2 import AF2PairBlock
    AF2PairBlock.rne_kernel = True
    _tn._TRIATT_FUSED_HIFI = True
    _tbw.FUSED = True

    M.CLOCK = M.Clock(1.0)
    M.CLOCK.start()

    rounds = Rounds(os.path.join(project, "rounds.json"))
    # The gate lives and dies inside `run_campaign`, so catch the instance on the way past.
    gates = []
    real_device_gate = duotraj.DeviceGate

    class RecordedGate(real_device_gate):
        def __init__(self):
            super().__init__()
            gates.append(self)
    duotraj.DeviceGate = RecordedGate

    cls = bindcraft2.design_model_class()
    real_sequence_gradients = cls.sequence_gradients

    def sequence_gradients(self, *a, **kw):
        if not kw.get("compile_only"):
            rounds.mark(duotraj.slot())
        return real_sequence_gradients(self, *a, **kw)
    cls.sequence_gradients = sequence_gradients

    stamp = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
             "trajectories": n, "validation": args.validation, "binder": args.binder,
             "seed": args.seed, "project": project,
             "commit": subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
             "compile_cache": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
             "pci": M.CLOCK.pci, "sysfs": M.CLOCK.path, "nproc": os.cpu_count(),
             "loadavg_start": os.getloadavg(), "mem_start": host_memory(),
             "started_utc": time.strftime("%FT%TZ", time.gmtime())}
    print(json.dumps(stamp, indent=1), flush=True)

    t0 = time.time()
    trajectories = None

    def close(exc=None):
        """Write the run out. Called BEFORE the predictor's teardown, which can abort the
        process on `close_device` and take the whole record with it."""
        M.CLOCK.stop()
        rounds.dump()
        stamp.update({"wall_seconds": round(time.time() - t0, 2),
                      "trajectories_returned": trajectories, "error": exc,
                      "rounds_per_slot": {s: sum(1 for r in rounds.rows if r["slot"] == s)
                                          for s in sorted({r["slot"] for r in rounds.rows})},
                      "gate": gates[-1].report() if gates else None,
                      "aiclk_samples": [[round(t, 2), c, l] for t, c, l in M.CLOCK.samples],
                      "mem_end": host_memory(), "loadavg_end": os.getloadavg(),
                      "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
        pathlib.Path(os.path.join(project, "run.json")).write_text(json.dumps(stamp, indent=1))
        print(json.dumps({k: v for k, v in stamp.items() if k != "aiclk_samples"}, indent=1),
              flush=True)

    with bindcraft2.campaign_predictor(trunk="device", validation=args.validation,
                                       checkpoints=args.params, extra_msa=True,
                                       template=True, exact=False):
        try:
            trajectories = bindcraft2.run_campaign(
                settings, project, trajectories_per_card=n,
                af2_weights=args.params,
                mpnn_weights=os.path.join(B.BC2, "bindcraft", "weights", "proteinmpnn",
                                          "weights_neutral"))
        except BaseException as exc:
            close(repr(exc))
            raise
        close()


if __name__ == "__main__":
    main()
