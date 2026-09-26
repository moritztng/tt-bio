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
    ap.add_argument("--arm", choices=["on", "off"], required=True,
                    help="on = the composed stack's triangle-attention route")
    ap.add_argument("--n", type=int, default=288)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--stack", default="template")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    on = a.arm == "on"
    os.environ["TT_BIO_TRIATT_TAPED_SDPA"] = "1" if on else "0"
    os.environ["TT_BIO_SDPA_OWN_FORWARD"] = "1"      # the agtri arm, the one inside the bar
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "perf" / "bcx_p10_tmplemb"))
    sys.argv = ["vjp.py", "--n", str(a.n), "--reps", str(a.reps), "--stack", a.stack,
                "--out", a.out]

    import vjp
    vjp.main()

    from tt_bio import tenstorrent, taped_ttnn
    reach = {"triatt_sdpa_stats": dict(tenstorrent.TRIATT_TAPED_SDPA_STATS),
             "sdpa_own_forward_stats": dict(taped_ttnn.SDPA_OWN_FORWARD_STATS),
             "arm": a.arm}
    print("REACH " + json.dumps(reach))
    # vjp.py treats --out as a directory prefix, so the reach counter goes beside its file
    # rather than into it. An arm whose counter is zero graded nothing and the JSON alone
    # cannot say so.
    d = pathlib.Path(a.out)
    (d / "reach.json" if d.is_dir() else d.with_suffix(".reach.json")).write_text(
        json.dumps(reach, indent=1))


if __name__ == "__main__":
    main()
