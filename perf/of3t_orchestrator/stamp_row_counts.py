#!/usr/bin/env python3
"""Stamp the three row-count sites in of3t-orchestrator.md from DISK, not from prose.

The audit requires the counts in words and the ledger moves while the pass is being written --
four hand-restamps in one pass, each one stale before the compose finished. Compute, write,
compose, in that order and with no thinking in between.
"""
import pathlib, re, sys
D = pathlib.Path("/home/moritz/.coworker")
W = {1:"one",2:"two",3:"three",4:"four",5:"five",6:"six",7:"seven",8:"eight",9:"nine",10:"ten",
     11:"eleven",12:"twelve",13:"thirteen",14:"fourteen",15:"fifteen",16:"sixteen",
     17:"seventeen",18:"eighteen",19:"nineteen"}
def word(n):
    if n < 20: return W[n]
    if n < 100: return {2:"twenty",3:"thirty",4:"forty",5:"fifty",6:"sixty",7:"seventy",8:"eighty",9:"ninety"}[n//10] + ("-"+W[n%10] if n%10 else "")
    return word(n//100) + " hundred" + (" " + word(n%100) if n%100 else "")
briefs  = len(list((D/"workstreams").glob("of3t-*.txt")))
markers = [m.name for m in (D/"state"/"concluded").glob("of3t-*")]
own     = [m for m in markers if m.startswith("of3t-orchestrator")]
rows    = len(markers) - len(own)
p = D/"state"/"of3t-orchestrator.md"; t = p.read_text()
t = re.sub(r"ROWS: \*\*[a-z\- ]+ dispatched, [a-z\- ]+ concluded\*\*",
           f"ROWS: **{word(briefs)} dispatched, {word(rows)} concluded**", t, count=1)
t = re.sub(r"\(\d+ `of3t-\*` briefs including this row's own; \d+ markers",
           f"({briefs} `of3t-*` briefs including this row's own; {len(markers)} markers", t, count=1)
t = re.sub(r"One hundred [a-z\- ]+ rows dispatched, one\nhundred [a-z\- ]+ concluded",
           f"{word(briefs).capitalize()} rows dispatched, {word(rows)}\nconcluded", t, count=1)
t = re.sub(r"`state/concluded` holds \*\*[a-z\- ]+\*\* of3t files",
           f"`state/concluded` holds **{word(len(markers))}** of3t files", t, count=1)
t = re.sub(r"so \*\*[a-z\- ]+ rows have concluded\*\*", f"so **{word(rows)} rows have concluded**", t, count=1)
p.write_text(t)
print(f"stamped: {briefs} briefs, {len(markers)} markers, {rows} rows concluded")
