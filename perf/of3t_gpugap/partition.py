#!/usr/bin/env python3
"""Re-derive every number in `state/of3t-gpugap.md` from artifacts already in git.

No card and no GPU. The inputs are two banked `of3t-stepfloor` runs plus the verb-call
table that row published, so the whole partition is re-checkable on a laptop:

    git show origin/wk/of3t-stepfloor:perf/of3t_stepfloor/out/step_rekey_384.json
    git show origin/wk/of3t-stepfloor:perf/of3t_stepfloor/out/step_rekey_b_384.json
    git show origin/wk/of3t-stepfloor:perf/of3t_stepfloor/out/d164_probeoff_renormoff_384.json

Why the split below is PORTED vs UNPORTED, checked in the source rather than assumed:

  loss heads  `tt_bio/train/losses.py` and `objectives.py` contain zero occurrences of
              `ttnn` -- af3_loss is numpy on the host. The PCIe leg is reported separately
              by the harness as `download_s` and it is milliseconds.
  optimizer   `tt_bio/train/optim.py:153-165` keeps the fp32 masters and both moments in
              numpy on the host, and its docstring records that the placement is FORCED:
              `ttnn.moreh_adamw` accepts `param_in` as bfloat16 or bfloat8_b only.
  the rest    ttnn, sync-bracketed by `perf/of3t_stepfloor/fullstep.py`.

    python3 perf/of3t_gpugap/partition.py            # from any tt-bio checkout with origin
    python3 perf/of3t_gpugap/partition.py --json perf/of3t_gpugap/PARTITION.json
"""
from __future__ import annotations

import argparse
import json
import subprocess

STEPFLOOR = "origin/wk/of3t-stepfloor"
RUNS = ("step_rekey_384.json", "step_rekey_b_384.json")

#: The parts `fullstep.py` times, and where each one runs. `seed_upload` is the seam.
PORTED = ("trunk_s", "diffusion_s", "backward_s")
UNPORTED = ("losses_s", "optimizer_s")
SEAM = ("seed_upload_s",)

#: `of3t-stepfloor.md` D164 pass 2, probe OFF and the renorm lever OFF: the same program's
#: forward and backward out of one warm process on one card, with ttnn verb calls counted.
#: JIT is burned off on this arm -- it is the pass whose forward read 4.04 s against the
#: cold arm's 13.53 s.
D164 = {"forward_s": 4.04, "forward_calls": 84_996,
        "backward_s": 356.00, "backward_calls": 168_922}

#: `of3t/EVIDENCE.md:89` -- a complete Lightning step, trunk and diffusion and every loss
#: head and the optimizer, H200 at 1980 MHz, crop 384, batch 1, bf16-mixed. n=2 steady.
GPU_STEP_S = (7.0, 8.0)

#: Diffusion samples on the reference side. NOT from the shipped `initial_training` stage
#: yaml -- the H200 and CPU arms both ran the runner yaml upstream's `test_training_full.py`
#: generates (`scripts/datasets/pdb_subset_helpers.py:520 build_runner_yaml_config`), which
#: overrides token_budget/batch_size/precision/epoch_len/max_epochs and the diffusion loss
#: chunk_size but NOT `no_samples`. So it falls through to the architecture default at
#: `openfold3/projects/of3_all_atom/config/model_config.py:181`
#: `architecture.shared.diffusion.no_samples: 48` (`num_recycles: 3` at :177). The `train`
#: preset that yaml selects is a MEMORY preset and changes neither. Both routes give 48;
#: this is the one that is checkable, and `--upstream <repo>` re-checks it. The tree on pc
#: (`~/.coworker/scratch/of3-upstream/repo`) is 0.4.6.dev12+g72fc3a953 while `of3t-theirtest`
#: ran 0.5.0, so what the re-check proves is that the default is 48 on 0.4.6 as well as on
#: 0.4.3 (`perf/of3t_perf/their_step_shape_0.4.3.json`). 0.5.0 is not directly re-read here;
#: it is bracketed by two revisions either side that agree.
THEIR_SAMPLES = 48

#: `of3t-theirtest.md:68` -- upstream's OWN Lightning step, same generated runner yaml, run on
#: qb2's HOST CPU (Ryzen 7 9700X, 8 cores, torch 2.8.0+cpu, no card). One step, crop 384,
#: batch 1, bf16-mixed. It is the FIRST step their loop completed, so it carries warmup; on
#: the H200 the first step was 11 s against a 7-8 s steady state.
CPU_STEP_S = 394.61

#: `of3t-perf.md` -- an UNTAPED trunk cycle at crop 384, reproducing to 0.3 % across processes.
UNTAPED_CYCLE_S = 2.40


def load(name):
    blob = subprocess.run(["git", "show", f"{STEPFLOOR}:perf/of3t_stepfloor/out/{name}"],
                          capture_output=True, text=True, check=True).stdout
    return json.loads(blob)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--upstream", help="path to an upstream openfold3 checkout; re-checks "
                                       "no_samples and num_recycles against its source")
    a = ap.parse_args()

    rows, out = [], {}
    for run in RUNS:
        d = load(run)
        arm = "B" if "_b_" in run else "A"
        assert d["config"]["crop"] == 384 and d["config"]["batch"] == 1
        n_samp = d["config"]["diffusion_samples"]
        for r in d["reps"]:
            parts = {k: r[k] for k in PORTED + UNPORTED + SEAM}
            total = sum(parts.values())
            # The parts must BE the step, not merely resemble it: an unattributed remainder
            # is where a partition quietly stops being one.
            assert abs(total - r["step_s"]) < 2e-3, (run, r["rep"], total, r["step_s"])
            unp = sum(r[k] for k in UNPORTED)
            per_root = r["losses"]["host_loss_s"] / n_samp
            rows.append({
                "arm": f'{arm}{r["rep"]}', "cold": r["cold"], "step_s": r["step_s"],
                "backward_share": 100 * r["backward_s"] / r["step_s"],
                "unported_s": round(unp, 3), "unported_share": 100 * unp / r["step_s"],
                "amdahl_ceiling_x": r["step_s"] / unp,
                "host_loss_per_root_s": per_root,
                "host_loss_at_48_s": THEIR_SAMPLES * per_root,
                "nograd_prefix_s": r["trunk_nograd_prefix_s"],
            })

    print(f'{"arm":4} {"step s":>9} {"bwd":>7} {"unported s":>11} {"unp":>6} '
          f'{"ceiling":>8} {"loss/root":>10} {"loss@48":>8}')
    for x in rows:
        print(f'{x["arm"]:4} {x["step_s"]:9.3f} {x["backward_share"]:6.2f}% '
              f'{x["unported_s"]:11.3f} {x["unported_share"]:5.2f}% '
              f'{x["amdahl_ceiling_x"]:7.1f}x {x["host_loss_per_root_s"]:10.4f} '
              f'{x["host_loss_at_48_s"]:8.1f}')

    # The headline. Arm B rep 2 is the steady rep on the quiet host: JIT-warm, and the arm
    # whose loadavg stayed in the 9-12 band. Naming which rep and why is the whole point.
    head = next(x for x in rows if x["arm"] == "B2")
    lo, hi = (head["step_s"] / GPU_STEP_S[1], head["step_s"] / GPU_STEP_S[0])
    print(f'\nHEADLINE {head["step_s"]:.2f} s vs {GPU_STEP_S[0]}-{GPU_STEP_S[1]} s '
          f'= {lo:.1f}-{hi:.1f}x, crop 384, whole step both sides')
    assert all(x["nograd_prefix_s"] == 0.0 for x in rows), "a prefix appeared; re-read the ceiling"
    print(f'  no no_grad prefix on any rep; their draw is U{{0..3}}, mean 1.5 cycles at '
          f'{UNTAPED_CYCLE_S} s = {1.5 * UNTAPED_CYCLE_S:.1f} s, '
          f'{100 * 1.5 * UNTAPED_CYCLE_S / head["step_s"]:.2f} % of the step')

    # The second-level partition: is the backward the model's arithmetic? Price its verb
    # calls at the FORWARD's own warm rate, measured on the same card in the same process.
    fw = D164["forward_s"] / D164["forward_calls"]
    bw = D164["backward_s"] / D164["backward_calls"]
    pred = D164["backward_calls"] * fw
    print(f'\nBACKWARD, D164 trunk scope, one warm process, one card:')
    print(f'  forward  {1000 * fw:.4f} ms/call   backward {1000 * bw:.4f} ms/call   '
          f'{bw / fw:.1f}x per call, {D164["backward_calls"] / D164["forward_calls"]:.3f}x the calls')
    print(f'  at the forward\'s own rate the backward would cost {pred:.2f} s; it costs '
          f'{D164["backward_s"]:.2f} s')
    for k in (2, 3, 5):
        print(f'    a backward verb at {k}x a forward verb: {k * pred:5.1f} s, '
              f'still {D164["backward_s"] / (k * pred):.1f}x short')

    # The ceiling, which has two answers, and the second one is the finding.
    q = [x for x in rows if x["arm"].startswith("B")]
    lo48 = min(x["host_loss_at_48_s"] for x in q) + min(2.989, 3.194)
    hi48 = max(x["host_loss_at_48_s"] for x in q) + 3.194
    print(f'\nCEILING at the measured 4 samples: {head["amdahl_ceiling_x"]:.1f}x '
          f'({min(x["amdahl_ceiling_x"] for x in rows):.1f}-'
          f'{max(x["amdahl_ceiling_x"] for x in rows):.1f}x over eight reps) -- does not block '
          f'{lo:.0f}-{hi:.0f}x')
    print(f'CEILING at their shipped {THEIR_SAMPLES} samples: host loss is linear in root count, '
          f'so unported alone is {lo48:.1f}-{hi48:.1f} s')
    print(f'  with the ENTIRE device side at zero that is still '
          f'{lo48 / GPU_STEP_S[1]:.1f}-{hi48 / GPU_STEP_S[0]:.1f}x slower than the GPU\'s whole step')

    # The calibration that says this is an implementation gap and not a hardware one.
    print(f'\nCALIBRATION vs upstream\'s own step on an 8-core desktop CPU '
          f'({CPU_STEP_S} s, same crop/batch/precision, {THEIR_SAMPLES} samples):')
    print(f'  our p300c step is {head["step_s"] / CPU_STEP_S:.2f}x the CPU\'s, at '
          f'4 samples against its {THEIR_SAMPLES}. A Blackhole losing to eight Zen cores is '
          f'not a FLOPs story')

    if a.upstream:
        import pathlib as _pl
        mc = (_pl.Path(a.upstream) / "openfold3/projects/of3_all_atom/config/model_config.py")
        src = mc.read_text()
        # The runner-yaml generator must not override no_samples, or the default is not what ran.
        gen = (_pl.Path(a.upstream) / "scripts/datasets/pdb_subset_helpers.py").read_text()
        i = gen.index("def build_runner_yaml_config")
        body = gen[i:gen.index("\ndef ", i + 1)]
        assert "no_samples" not in body, "the generator DOES set no_samples -- re-read the axis"
        assert f'"no_samples": {THEIR_SAMPLES},' in src, "architecture default moved"
        assert '"num_recycles": 3,' in src, "recycle default moved"
        print(f'  upstream re-checked at {a.upstream}: architecture default no_samples '
              f'{THEIR_SAMPLES}, num_recycles 3, and build_runner_yaml_config overrides neither')

    out = {"doc": "of3t-gpugap: the OF3 training step partitioned into ported and unported",
           "reps": rows, "headline": {"tt_step_s": head["step_s"], "gpu_step_s": GPU_STEP_S,
                                      "ratio_x": [lo, hi], "crop": 384, "scope": "whole step",
                                      "tt_diffusion_samples": 4, "their_diffusion_samples": 48,
                                      "floor": True},
           "backward_verb_rate": {**D164, "fwd_ms_per_call": 1000 * fw,
                                  "bwd_ms_per_call": 1000 * bw, "ratio_x": bw / fw,
                                  "predicted_at_fwd_rate_s": pred},
           "cpu_calibration": {"upstream_cpu_step_s": CPU_STEP_S,
                               "tt_over_cpu_x": head["step_s"] / CPU_STEP_S,
                               "note": "same crop/batch/precision and 48 samples vs our 4; the "
                                       "CPU figure is their first step, so it carries warmup"},
           "ceiling": {"at_4_samples_x": head["amdahl_ceiling_x"],
                       "at_48_samples_unported_s": [lo48, hi48]}}
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
