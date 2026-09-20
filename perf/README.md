# perf/

Measurement artifacts. A tuning constant in `tt_bio/`, a claim in `docs/`, a published cell on
`site/` and a gate arm in `tests/` all cite the file they were set from, and this is where that
file lives.

One directory per measurement, named for the lever or the campaign, holding the JSON the run
wrote and the script that wrote it. Raw fold outputs (CIF, PDB) belong here only while the claim
that reads them is still being written; once the numbers are in a JSON beside them, the structures
go.

**A directory survives because something names it.** `tests/test_perf_citations.py` fails if a
`perf/...` path named from shipped source, docs or the site does not resolve, which is the half
that catches a lever landing without its evidence. The other half is the tidy's, and it is now
one command:

```
PYTHONPATH=tests python3 tests/test_perf_citations.py
```

That prints the directories nothing outside `perf/` names. They are concluded passes whose answer
already lives in the comment, the doc or the CHANGELOG entry they produced, and they get deleted.
The run is still in the history and on the branch that made it.

So: cite the artifact from the line it sets. An unnamed directory is not protected by being useful.

**Naming is not claiming, and the census reads the wider set.** `tests/test_antibody_rmsd.py`
names `perf/abb3/verify_instrument.py` as the instrument its real validation runs in, and every
probe under `scripts/rfd3_port/` names the tree it writes. Those are not claims -- the two
resolution tests must not assert a script's own output path already exists, and 56 of them do not
-- but deleting the directory orphans the file that named it. So the two resolution tests read
`tt_bio/`, `docs/` and `site/`, and the census reads `tests/`, `scripts/` and `RELEASING.md` on
top. Censusing the claim surfaces alone read 75 directories as unnamed on 2026-09-20, `perf/abb3`
among them.

And a curated JSON is prose. `docs/perf_baselines.json` is the only file in the repo naming
`perf/qb2cardlayer`, `site/data/perf-512aa.json` the only one naming `perf/wh-embed`; both were
held back by hand on 2026-09-11 because the census skipped `.json` everywhere. It skips it under
`tt_bio/`, `tests/` and `scripts/` only, where a JSON is model data or a run dump.

Deleting a directory a SURVIVOR still points at leaves the pointer dangling inside the evidence
tree, which is the one place the history is no help, so the census holds those back too, to a
fixed point.

Census both spellings. A gate that opens a tree builds the path from segments -- the release gate
reaches this one as `REPO_ROOT / "perf" / "ceilrfd3" / "targets"` -- and a census that greps for
`perf/...` as one string reads those directories as uncited and deletes them. The 2026-09-11 tidy
had `perf/ceilrfd3` and `perf/wh-parity` on its delete list for exactly that reason.
`tests/test_perf_citations.py` now collects the joined form too, from `tests/` and `scripts/` as
well as the three claim surfaces, so running it is the census.
