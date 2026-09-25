#!/usr/bin/env python3
"""What the template embedding and the structure module are MADE of, per HLO instruction.

`perf/bcx_seam/hostmap.py` charges a round's host seconds to AF2 modules and stops there. Two
of its lines decide this row and neither is legible at that grain: "template embedding [bwd]
other" 1.90 s and "structure module" 1.7-2.2 s. This reruns BindCraft 2's own round under
`jax.profiler` with the optimised-HLO dump and groups the same thunks by their FULL `op_name`,
so a line names a haiku module path instead of a bucket.

It runs BindCraft 2's trunk in JAX and opens no card: the two modules in question are host JAX
in every arm, the anatomy does not depend on where the Evoformer runs, and qb2's four nodes are
held by other rows.
"""
import argparse
import functools
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
for _p in (str(_ROOT / "perf" / "bcx_round"), str(_ROOT / "perf" / "bcx_predictor"),
           str(_ROOT), str(_ROOT / "perf" / "bcx_afgrad"), str(_ROOT / "perf" / "bcx_stack")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import meter as M                                                      # noqa: E402
import bc2_state as B                                                  # noqa: E402
import bindcraft.campaign as campaign                                  # noqa: E402
import bindcraft.trajectory as trajectory                              # noqa: E402
import bindcraft.sequence_optimization as seqopt                       # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings   # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings              # noqa: E402
from run_round import MONOMER, git_head                                # noqa: E402


class ProfMeter(M.Meter):
    def __init__(self, rounds, profile, trace_dir):
        super().__init__(rounds)
        self.profile, self.trace_dir, self.tracing = profile, trace_dir, False

    def on_sequence_gradients_enter(self):
        import jax
        r = self.entries + 1
        if self.profile and self.tracing and r > self.profile[1]:
            jax.profiler.stop_trace()
            self.tracing = False
            M.EVENTS.append({"kind": "trace_stop", "phase": "round", "t0": time.time(),
                             "round": r})
        super().on_sequence_gradients_enter()
        if self.profile and r == self.profile[0]:
            jax.profiler.start_trace(self.trace_dir)
            self.tracing = True
            M.EVENTS.append({"kind": "trace_start", "phase": "round", "t0": time.time(),
                             "round": r})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--profile", default="3,4")
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    project = args.out
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)
    overrides = [f"campaign_seed={args.seed}", "max_trajectories=1",
                 "validation_model=monomer", 'design_models=["model_1_ptm"]',
                 'validation_models=["model_2_ptm"]', f"project_folder={project}",
                 "compile_next_length=false"]
    settings = cleaned_campaign_settings(
        read_settings(os.path.join(B.BC2, "examples", "pdl1.json"),
                      parse_setting_overrides(overrides)))
    campaign.MULTIMER_POOL = MONOMER

    import ttbio_predictor as T
    campaign.AlphaFoldDesignModel = functools.partial(
        T.TTBioAlphaFoldDesignModel, trunk="jax")

    profile = tuple(int(x) for x in args.profile.split(","))
    stamp = {"host": os.uname().nodename, "card": None, "trunk": "jax",
             "commit": git_head(), "seed": args.seed, "rounds_requested": args.rounds,
             "profile": profile, "omp": os.environ.get("OMP_NUM_THREADS"),
             "xla_flags": os.environ.get("XLA_FLAGS"),
             "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg(), "nproc": os.cpu_count(), "project": project}
    print(json.dumps(stamp, indent=1), flush=True)

    import splice
    mt = ProfMeter(args.rounds, profile, os.path.join(project, "trace"))
    M.install(mt, splice, T.TTBioAlphaFoldDesignModel, trajectory, seqopt)

    t0 = time.time()
    stopped = None
    try:
        campaign.run_campaign(settings, project, af2_weights=args.params,
                              mpnn_weights=os.path.join(B.BC2, "bindcraft", "weights",
                                                        "proteinmpnn", "weights_neutral"),
                              max_trajectories=1)
    except M.StopAfterRounds as stop:
        stopped = str(stop)
    finally:
        if mt.tracing:
            import jax
            jax.profiler.stop_trace()
        stamp.update({"wall_seconds": round(time.time() - t0, 2), "stopped": stopped,
                      "loadavg_end": os.getloadavg(),
                      "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
        M.dump(os.path.join(project, "round_events.json"), stamp)
        print(json.dumps(stamp, indent=1), flush=True)


if __name__ == "__main__":
    main()
