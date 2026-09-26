#!/usr/bin/env python3
"""The template pair stack's VJP graded at the COMPOSED configuration.

`perf/bcx_p10_tmplemb/vjp.py` is the campaign's grader for this stack: float64 reference,
bfloat16 CPU arm beside it as the floor no correct device implementation can beat, device arm
must sit inside `--bar` times that floor. It is adopted here unchanged; rewriting it would mean
grading against a second reference.

What it was never run with is the triangle-attention route ON. `wk/bcx-p10-triatt` graded that
route on the Evoformer's own attentions; the template stack's two c=64 pair blocks carry
triangle attentions too, and with `TT_BIO_TRIATT_TAPED_SDPA=1` they go through it as well. That
combination is what the composed stack runs and nothing has graded it.

Note `--triatt-fused` is NOT the lever: that is `set_triatt_fused`, the `generic_op` arm
`wk/bcx-p10-triatt` closed because it reaches zero of a round's taped attentions. The shipped
lever is the pair of environment flags, read live at the call site, so it is set here and its
reach counter printed afterwards -- an arm whose counter is zero graded nothing.

    python3 perf/bcx_p10_stack/grad_compose.py --arm on  --out out/grad_on.json
    python3 perf/bcx_p10_stack/grad_compose.py --arm off --out out/grad_off.json
"""
import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["agtri", "hifi", "off", "on"], required=True,
                    help="which triangle-attention route the graded arm takes. `agtri` is the "
                         "stock fused verb, `hifi` is bcx-p10-tapegen's taped fused HiFi "
                         "kernel, `off` is the materialised path. `on` is the old name for "
                         "`agtri` and is kept so the earlier readings stay reproducible")
    ap.add_argument("--n", type=int, default=288)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--stack", default="template")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arm = "agtri" if a.arm == "on" else a.arm
    os.environ["TT_BIO_TRIATT_TAPED_SDPA"] = "1" if arm == "agtri" else "0"
    os.environ["TT_BIO_SDPA_OWN_FORWARD"] = "1"      # the agtri arm, the one inside the bar
    # Set before tt_bio is imported, which is why this runs above the sys.path insert:
    # `_TRIATT_FUSED_HIFI` is resolved at import and a later assignment would be too late for
    # the module-level default, unlike in run_round.py where tt_bio is already loaded.
    os.environ["TT_BIO_TAPED_KERNELS"] = "tri_att_sdpa_hifi" if arm == "hifi" else ""
    os.environ["TT_BIO_TRIATT_FUSED_HIFI"] = "1" if arm == "hifi" else "0"
    os.environ["TT_BIO_TRIATT_DIVIDING_K"] = "1" if arm == "hifi" else "0"
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "perf" / "bcx_p10_tmplemb"))
    sys.argv = ["vjp.py", "--n", str(a.n), "--reps", str(a.reps), "--stack", a.stack,
                "--out", a.out]

    import vjp
    vjp.main()

    from tt_bio import tenstorrent, taped_ttnn
    reach = {"triatt_sdpa_stats": dict(tenstorrent.TRIATT_TAPED_SDPA_STATS),
             "sdpa_own_forward_stats": dict(taped_ttnn.SDPA_OWN_FORWARD_STATS),
             "fused_hifi_stats": dict(tenstorrent.TRIATT_FUSED_HIFI_STATS),
             "kernel_entry_stats": {k: list(v) for k, v in taped_ttnn.KERNEL_STATS.items()},
             "arm": arm}
    print("REACH " + json.dumps(reach))
    # vjp.py treats --out as a directory prefix, so the reach counter goes beside its file
    # rather than into it. An arm whose counter is zero graded nothing and the JSON alone
    # cannot say so.
    d = pathlib.Path(a.out)
    (d / "reach.json" if d.is_dir() else d.with_suffix(".reach.json")).write_text(
        json.dumps(reach, indent=1))


if __name__ == "__main__":
    main()
