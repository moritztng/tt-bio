#!/usr/bin/env python3
"""Resolve the shipped default's trajectory count on this box, and record why.

`duotraj.auto_trajectories()` is what `bindcraft2.run_campaign` calls when
`trajectories_per_card` is left at its default, so the number printed here is the number the
product would have chosen for this arm. The token axis is passed when the resolver takes one,
and ignored when it does not: that difference is the whole of what this row is measuring.
"""
import inspect
import json
import sys

from tt_bio import duotraj


def main():
    out, binder = sys.argv[1], int(sys.argv[2])
    tokens = -(-(binder + 129) // 32) * 32
    kw = {}
    if "tokens" in inspect.signature(duotraj.auto_trajectories).parameters:
        kw["tokens"] = tokens
    n, why = duotraj.auto_trajectories(**kw)
    json.dump({"n": n, "why": why, "tokens": tokens, "size_aware": bool(kw),
               "free_host": duotraj.free_host_bytes(),
               "free_device": duotraj.free_device_bytes(),
               "rss": duotraj.host_rss_bytes()}, open(out, "w"), indent=1)
    print(n)


if __name__ == "__main__":
    main()
