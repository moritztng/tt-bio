#!/usr/bin/env python3
"""Does AF2's triangle-attention softmax actually REACH the exact stand-in? Card-free.

`armed.py` proved the six bindings are the host float64 stand-ins inside `tape()`. It checked
`getattr(ttnn, name)` on the real ttnn module and `taped_ttnn._VERBS`. Neither is the
expression BindCraft 2's dominant softmax evaluates. AF2 routes both triangle attentions through
`tenstorrent._fp32_softmax_attention` (`af2.py:546`, `fp32_softmax=True` at
`af2.py:386-391`), whose tail calls `ttnn.softmax_in_place(attn, ...)` where `ttnn` is
`tt_bio.tenstorrent`'s own module global -- and `taped_ttnn._swap` rebinds that name to the
shim in every tt_bio module except `taped_ttnn` and `autograd` (`taped_ttnn.py:1128`).
So the question is whether the shim's resolution of `softmax_in_place` lands on the exact verb,
and whether `host_f64_softmax_site` -- which defaults False at every site and is a DIFFERENT
lever -- has anything to do with it.

Three checks, none of which reads the source:
  1. `dis` the two call sites, to show the name is resolved at CALL time off the module global
     rather than captured once at import. A captured reference would make the rebinding inert.
  2. Resolve `tt_bio.tenstorrent.ttnn.softmax_in_place` inside `tape()` and CALL it, and watch
     `EXACT_SOFTMAX_STATS` move. A counter that moves is the reach.
  3. Show the two levers are independent: `HOST_F64_SOFTMAX_STATS['selected']` stays 0 while
     the exact counters move, so the site flag being off at `af2.tri_att` does not close this
     path.
"""
from __future__ import annotations

import dis
import io
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch                                                           # noqa: E402
import ttnn                                                            # noqa: E402

from tt_bio import autograd as ag                                      # noqa: E402
from tt_bio import taped_ttnn as tt                                    # noqa: E402
from tt_bio import tenstorrent as ts                                   # noqa: E402


def disasm(fn, needle):
    """The instructions around `needle` in `fn`, so the binding's resolution is visible."""
    buf = io.StringIO()
    dis.dis(fn, file=buf)
    lines = buf.getvalue().splitlines()
    keep = [l for i, l in enumerate(lines) if needle in l
            or (i + 1 < len(lines) and needle in lines[i + 1])]
    return keep


def call_through(modname, mod, attr, x, **kw):
    """Resolve `<mod>.ttnn.<attr>` the way that module's own code does, call it, report the
    counter delta. The resolution happens inside the call, not before it, exactly as the
    bytecode does it."""
    before = dict(ag.EXACT_SOFTMAX_STATS), dict(ag.EXACT_LAYER_NORM_STATS)
    raised = None
    try:
        getattr(mod.ttnn, attr)(x, **kw)
    except BaseException as exc:                                       # noqa: BLE001
        raised = f"{type(exc).__name__}: {exc}"[:160]
    after = dict(ag.EXACT_SOFTMAX_STATS), dict(ag.EXACT_LAYER_NORM_STATS)
    return {"module": modname, "attr": attr, "raised": raised,
            "ttnn_global_is_shim": mod.ttnn is tt.taped_ttnn(),
            "softmax_verb_delta": after[0]["verb"] - before[0]["verb"],
            "softmax_raw_delta": after[0]["raw"] - before[0]["raw"],
            "layer_norm_verb_delta": after[1]["verb"] - before[1]["verb"],
            "layer_norm_raw_delta": after[1]["raw"] - before[1]["raw"]}


def main():
    blob = {"host": os.uname().nodename, "when_utc": time.strftime("%FT%TZ", time.gmtime()),
            "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                     capture_output=True, text=True).stdout.strip(),
            "opened_a_device": False, "loadavg": os.getloadavg()}

    blob["call_sites"] = {
        "_fp32_softmax_tail/softmax_in_place": disasm(ts._fp32_softmax_tail, "softmax_in_place"),
        "site_softmax/softmax": disasm(ts.site_softmax, "softmax"),
    }

    # The site flag AF2's triangle attention selects, and what it answers by default.
    blob["site_flag"] = {"af2.tri_att": ts.host_f64_softmax_site("af2.tri_att"),
                         "af2.msa": ts.host_f64_softmax_site("af2.msa"),
                         "selected_before": ts.HOST_F64_SOFTMAX_STATS["selected"]}

    x = torch.zeros(32, 32)
    blob["outside_tape"] = call_through("tt_bio.tenstorrent", ts, "softmax_in_place",
                                        x, dim=-1)
    with tt.tape():
        blob["inside_tape"] = call_through("tt_bio.tenstorrent", ts, "softmax_in_place",
                                           x, dim=-1)
        blob["inside_tape_layer_norm"] = call_through("tt_bio.tenstorrent", ts,
                                                      "layer_norm", x)
    with ag.exact_training(False):
        with tt.tape():
            blob["inside_tape_off"] = call_through("tt_bio.tenstorrent", ts,
                                                   "softmax_in_place", x, dim=-1)
    blob["site_flag"]["selected_after"] = ts.HOST_F64_SOFTMAX_STATS["selected"]
    blob["site_flag"]["served_after"] = ts.HOST_F64_SOFTMAX_STATS["served"]

    # A call that arrives with a plain tensor takes the shim's untaped short circuit in
    # `taped_ttnn._taped_verb` -- `if not _on_tape(...): return shipped(...)` -- and `shipped`
    # is the object `getattr(ttnn, name)` returned AFTER `_install_exact` swapped it. So the
    # exact stand-in serves BOTH branches inside the scope: the `raw` counter for an untaped
    # operand, the `verb` counter for a taped one. REACH is the sum, not either alone. The
    # dummy operand here is a torch tensor, so it lands on `raw`; a real triangle-attention
    # score tensor is on the tape and lands on `verb`. Both are the host float64 round trip,
    # which is why the sum is the right test and why neither branch is an escape.
    def reached(k, which):
        d = blob[k]
        return d[f"{which}_verb_delta"] + d.get(f"{which}_raw_delta", 0) == 1

    blob["ANSWER"] = {
        "tri_att_softmax_reaches_the_exact_path": reached("inside_tape", "softmax"),
        "which_branch_served_it": ("verb" if blob["inside_tape"]["softmax_verb_delta"]
                                   else "raw"),
        "layer_norm_reaches_the_exact_path": reached("inside_tape_layer_norm", "layer_norm"),
        "call_site_resolves_at_call_time":
            any("LOAD_ATTR" in l and "softmax_in_place" in l
                for l in blob["call_sites"]["_fp32_softmax_tail/softmax_in_place"]),
        "off_switch_closes_it": (blob["inside_tape_off"]["softmax_verb_delta"]
                                 + blob["inside_tape_off"]["softmax_raw_delta"] == 0),
        "untaped_call_does_not_reach_it": (blob["outside_tape"]["softmax_verb_delta"]
                                           + blob["outside_tape"]["softmax_raw_delta"] == 0),
        "independent_of_the_site_flag": (not blob["site_flag"]["af2.tri_att"]
                                          and blob["site_flag"]["served_after"] == 0),
    }
    out = HERE / "REACH.json"
    out.write_text(json.dumps(blob, indent=1))
    print(json.dumps(blob, indent=1))
    print(f"-> {out}")
    checks = {k: v for k, v in blob["ANSWER"].items() if isinstance(v, bool)}
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
