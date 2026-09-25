#!/usr/bin/env python3
"""Resolve the OpenFold3 `Modifications:` paragraph of NOTICE between main and `of3t-data`.

main (fde1cff91) added one fact to the old paragraph: `create_paired_from_precomputed` comes from
0.5.0 (upstream PR #373). `of3t-data` rewrote the paragraph for the restored training half. The
resolution is the row's paragraph with main's fact added. Its module count stays 15: on the
composed tree `audit_vendor_provenance.py` still reports 15 files carrying tt-bio changes, with
the two MSA modules now closest to 0.5.0.

Usage: resolve_notice_of3_vendor.py <file>  ->  0 resolved, 2 not this shape (caller must stop).
"""
import pathlib
import re
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
m = re.search(r"<<<<<<< [^\n]*\n(.*?)=======\n(.*?)>>>>>>> [^\n]*\n", s, re.S)
if not m:
    sys.exit(2)
ours, theirs = m.group(1), m.group(2)
if "create_paired_from_precomputed" not in ours:
    sys.exit(2)
pat = re.compile(r"(cyclic-offset helpers)(; everything else is 0\.4\.3\.)")
if len(pat.findall(theirs)) != 1:
    sys.exit(2)
ind = re.search(r"\n( +)cyclic-offset", theirs).group(1)
new = pat.sub(lambda k: k.group(1) + ", and create_paired_from_precomputed\n" + ind
              + "from 0.5.0 (upstream PR #373: 0.4.3 dropped precomputed\n" + ind
              + "paired MSA rows)" + k.group(2), theirs)
p.write_text(s[:m.start()] + new + s[m.end():])
