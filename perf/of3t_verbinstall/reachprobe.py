#!/usr/bin/env python3
"""Where does the trunk's softmax actually go, with the package install live?

PKG_HF3 scored 0.702981502944001 -- CTRL_B, the device-softmax control, to sixteen digits --
while `EXACT_SOFTMAX_STATS` said `raw` fired 1742 times and `verb` fired 0. So either the patch
was reverted before the taped forward, or it was present and the trunk's softmaxes never
reached it. Those need different fixes, and no amount of reading decides between them.

This asserts the bindings at three points (after install, after the harness patches, at exit)
and counts every call by call site, so the answer is a table rather than an inference.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import traceback


def main() -> int:
    argv = sys.argv[1:]
    out = ""
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == "--probe-out":
            out = argv[i + 1]; i += 2; continue
        rest.append(argv[i]); i += 1

    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trunkceiling"))

    import ttnn
    from tt_bio import autograd as ag
    from tt_bio import taped_ttnn as tt
    import arm as ceiling_arm

    SITES = collections.Counter()
    REPORT = {"identity": {}}

    def snap(when):
        REPORT["identity"][when] = {
            "verb_softmax_is_ours": tt._VERBS.get("softmax") is ag._v_exact_softmax,
            "verb_softmax_in_place_is_ours": tt._VERBS.get("softmax_in_place") is ag._v_exact_softmax,
            "ttnn_softmax_is_ours": ttnn.softmax is ag._exact_softmax_raw,
            "ttnn_softmax_in_place_is_ours": getattr(ttnn, "softmax_in_place", None) is ag._exact_softmax_raw,
            "shim_cached_softmax": "softmax" in tt._SHIM.__dict__,
            "shim_cached_softmax_in_place": "softmax_in_place" in tt._SHIM.__dict__,
            "autograd_module": id(ag), "taped_module": id(tt), "ttnn_module": id(ttnn),
            "installed": ag.exact_softmax_installed(),
        }

    # Count by CALLER, which is what says whether the trunk reaches this at all.
    real_raw = ag._exact_softmax_raw

    def counting_raw(v, dim=-1, **kw):
        st = traceback.extract_stack(limit=4)[-2]
        SITES["RAW %s:%d %s" % (os.path.basename(st.filename), st.lineno, st.name)] += 1
        return real_raw(v, dim, **kw)

    real_verb = ag._v_exact_softmax

    def counting_verb(shipped, args, kwargs):
        st = traceback.extract_stack(limit=4)[-2]
        SITES["VERB %s:%d %s" % (os.path.basename(st.filename), st.lineno, st.name)] += 1
        return real_verb(shipped, args, kwargs)

    ag._exact_softmax_raw = counting_raw
    ag._v_exact_softmax = counting_verb

    def dump():
        # In a finally, and on the signal path too: the first attempt at this probe died in the
        # taped backward and wrote nothing, which is the one outcome a diagnostic may not have.
        try:
            import tt_bio.tenstorrent as T
            REPORT["HOST_F64_SOFTMAX_STATS"] = dict(T.HOST_F64_SOFTMAX_STATS)
        except Exception as e:
            REPORT["HOST_F64_SOFTMAX_STATS"] = "unavailable: %s" % e
        REPORT["counters"] = dict(ag.EXACT_SOFTMAX_STATS)
        REPORT["call_sites"] = dict(SITES.most_common(40))
        print("REACHPROBE " + json.dumps(REPORT), flush=True)
        if out:
            with open(out, "w") as f:
                json.dump(REPORT, f, indent=1)

    import signal

    def _on_term(signum, frame):
        snap("at_signal")
        dump()
        raise SystemExit(4)

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    rc = 1
    with ag.exact_softmax():
        snap("after_install")
        sys.argv = ["arm.py"] + rest
        try:
            rc = ceiling_arm.main()
        except BaseException as e:
            REPORT["raised"] = "%s: %s" % (type(e).__name__, e)
        finally:
            snap("at_exit")
            dump()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
