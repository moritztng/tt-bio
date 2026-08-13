#!/usr/bin/env python3
"""Load the published AbAg-XM release the way a researcher would, from the Hub only.

Run under `env -i` with a throwaway HOME so nothing resolves from a local cache or an
ambient token. `samples` and `targets` download (17 MB, 48 KB); `structures` streams, which
is the point of it being a separate config.
"""
import collections, sys
from datasets import load_dataset

REPO = "Tenstorrent/abag-xm"
EXPECT = {"boltz2": 83968, "esmfold2": 83968, "opendde-abag": 83456, "protenix-v2": 83968}
fail = []

print("== samples (default config) ==")
ds = load_dataset(REPO, split="train")
print(ds)
n = ds.num_rows
print(f"{n} rows")
by = collections.Counter(ds["model"])
print(sorted(by.items()))
if n != sum(EXPECT.values()):
    fail.append(f"row count {n} != {sum(EXPECT.values())}")
if dict(by) != EXPECT:
    fail.append(f"per-model counts {dict(by)} != {EXPECT}")
tg = set(ds["target"])
nod = len({t for t, d in zip(ds["target"], ds["dockq"]) if d is None})
print(f"{len(tg)} targets; {nod} not dockq-scorable")
if len(tg) != 164:
    fail.append(f"{len(tg)} targets != 164")

print("\n== targets ==")
td = load_dataset(REPO, "targets", split="train")
print(f"{td.num_rows} rows, {len(td.column_names)} columns")
if td.num_rows != 164:
    fail.append(f"targets {td.num_rows} != 164")
if set(td["target"]) != tg:
    fail.append("targets config does not cover the same 164 targets as samples")

print("\n== structures (streamed, no 34 GB download) ==")
st = load_dataset(REPO, "structures", split="train", streaming=True)
print("columns:", list(next(iter(st)).keys()))
for r in list(st.take(3)):
    cif = r["cif"]
    sid = r["sample_id"]
    ok = cif.startswith("data_") and "_atom_site." in cif
    print(f"  {sid:>20}  {len(cif):>9,} chars  mmCIF={ok}")
    if not ok:
        fail.append(f"{sid} does not look like mmCIF")

print("\nFAIL: " + "; ".join(fail) if fail else "\nALL CHECKS PASS")
sys.exit(1 if fail else 0)
