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
# --- defects: INVOKE the generator; never rewrite its output.
#
# This block used to load UNFIXED_TRIAGE.json and reconcile it in place, and a defect that became
# UNFIXED while the stamper was the thing running was appended straight to CAMPAIGN-INTERNAL with
# the reason "classified by stamp_row_counts.py; no user-facing claim made". `_of3t_donecheck.py`
# refuses this campaign's GO while any defect is USER-FACING -- so the one path that could walk a
# NEW user-facing defect past the GO gate in silence was this stamper, not the triage. It also
# wrote `reasons` as a bare string where the generator writes {"class":..., "why":...}, so the
# artifact's schema depended on which program had touched it last.
#
# It survived unnoticed because it kept the artifact looking fresh through several passes in which
# `triage.py` itself was refusing to run (`TRIAGE IS STALE`, rc=1): the stamper absorbed exactly
# the drift the generator exists to SHOUT about.
#
# The generator asserts its classification against the live UNFIXED set and returns 1 rather than
# report a classification of a set that has moved. Delegating inherits that refusal: an unclassified
# defect now stops the stamp instead of being silently filed in the harmless class.
sys.path.insert(0, str(D/"wt"/"of3t-orchestrator"/"perf"/"of3t_orchestrator"))
from defects_union import defects_text
from status_vocab import statuses_by_defect
import json, subprocess
TRIAGE = D/"wt"/"of3t-orchestrator"/"perf"/"of3t_orchestrator"/"defecttriage"/"triage.py"
tp = D/"state"/"of3t"/"UNFIXED_TRIAGE.json"
r = subprocess.run([sys.executable, str(TRIAGE)], capture_output=True, text=True)
if r.returncode != 0:
    sys.stderr.write(r.stdout + r.stderr)
    raise SystemExit(
        "STAMP REFUSED: triage.py returned %d, so the defect classification does not describe the\n"
        "live UNFIXED set. Classify the named defects in triage.py's TABLE -- reading each one's\n"
        "`### D<N> UPDATE` bodies in DEFECTS.md, which is where a defect's class actually moves --\n"
        "and re-run. Do NOT hand-edit UNFIXED_TRIAGE.json: the gate reads it, and a class written\n"
        "by anything but the generator is a class nobody classified." % r.returncode)
j = json.loads(tp.read_text())          # read-only from here; the generator owns this file
n_def = len(statuses_by_defect(defects_text()))

p = D/"state"/"of3t-orchestrator.md"; t = p.read_text()
def ids(c): return ", ".join(sorted(j["classes"][c], key=lambda s: int(s[1:])))

# Every site is applied through `site()`, which RECORDS whether its pattern matched. The previous
# shape ran eight bare `re.sub`s, wrote the doc, and only then checked ONE of the eight -- so a doc
# rewrite that moved the other seven was invisible, and the single check it did make had been
# failing (rc=1) on every run for many passes while the doc drifted into free prose. Combined with
# the artifact rewrite above it, the stamper's whole surviving effect had become the one effect it
# should never have had: it mutated the gate's input and exited 1.
#
# So: compute first, report per site, and WRITE NOTHING unless the doc still has sites to stamp.
_sites = []
def site(name, pat, rep, flags=0):
    global t
    t2, n = re.subn(pat, rep, t, count=1, flags=flags)
    _sites.append((name, n)); t = t2

# A regex site is destroyed by the very doc rewrite that makes restamping necessary -- which is
# exactly how all ten below went to zero. A DELIMITED block survives a rewrite, because the rewrite
# either keeps the markers or visibly drops them. New counts go in blocks; the regex sites below
# are kept only for as long as the sentences they match still exist.
def block(name, body):
    global t
    o, c = f"<!-- STAMPED:{name} -->", f"<!-- /STAMPED:{name} -->"
    t2, n = re.subn(re.escape(o) + r".*?" + re.escape(c), lambda m: f"{o}\n{body}\n{c}",
                    t, count=1, flags=re.S)
    _sites.append((f"block:{name}", n)); t = t2

block("defects",
      f"**{word(n_def).capitalize()} defects filed**, **{j['unfixed_total']} UNFIXED** "
      f"({j['counts']['SCOPE-EXCLUDED']} scope-excluded, {j['counts']['USER-FACING']} "
      f"USER-FACING, {j['counts']['CAMPAIGN-INTERNAL']} campaign-internal). Generated by "
      f"`defecttriage/triage.py`, which refuses to report a classification of a set that has "
      f"moved; nothing else may write `UNFIXED_TRIAGE.json`.\n\n"
      f"    SCOPE-EXCLUDED     {j['counts']['SCOPE-EXCLUDED']:3d}   {ids('SCOPE-EXCLUDED')}\n"
      f"    USER-FACING        {j['counts']['USER-FACING']:3d}   {ids('USER-FACING')}\n"
      f"    CAMPAIGN-INTERNAL  {j['counts']['CAMPAIGN-INTERNAL']:3d}   {ids('CAMPAIGN-INTERNAL')}")
block("rows",
      f"**{briefs} dispatched over the campaign, {rows} concluded** "
      f"({briefs} `of3t-*` briefs including this row's own; {len(markers)} `state/concluded` "
      f"markers, {len(own)} of them this row's).")

# The ten regex sites this script was built on are GONE, not disabled: all ten matched zero after
# the doc was rewritten, and the three that appeared to come back were matching text the blocks
# above had just written -- two writers for one string. A delimiter is the whole mechanism now.

hit = [n for n, k in _sites if k]; miss = [n for n, k in _sites if not k]
if not hit:
    raise SystemExit(
        "STAMP REFUSED, NOTHING WRITTEN: none of the %d sites matched, so this stamper has no\n"
        "hold on the doc at all -- it cannot keep any count honest and a run of it means nothing.\n"
        "Missing: %s\n"
        "Either the doc was rewritten past every site (restore the sentences, or retire this\n"
        "script), or the patterns are stale. A stamper that matches nothing reads exactly like a\n"
        "doc that is already current, which is how this went unnoticed for many passes."
        % (len(_sites), ", ".join(miss)))
p.write_text(t)
print("stamped %d/%d sites: %s" % (len(hit), len(_sites), ", ".join(hit)))
if miss:
    print("  NOT PRESENT in the doc (not an error while others matched): " + ", ".join(miss))
print(f"counts: {briefs} briefs, {len(markers)} markers, {rows} rows concluded, "
      f"{n_def} defects, {j['unfixed_total']} UNFIXED {j['counts']}")
