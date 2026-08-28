# perf/

Measurement artifacts. A tuning constant in `tt_bio/`, a claim in `docs/`, a published cell on
`site/` and a gate arm in `tests/` all cite the file they were set from, and this is where that
file lives.

One directory per measurement, named for the lever or the campaign, holding the JSON the run
wrote and the script that wrote it. Raw fold outputs (CIF, PDB) belong here only while the claim
that reads them is still being written; once the numbers are in a JSON beside them, the structures
go.

**A directory survives because something cites it.** `tests/test_perf_citations.py` fails if a
`perf/...` path named from shipped source, docs or the site does not resolve, which is the half
that catches a lever landing without its evidence. The other half is manual and belongs to the
periodic tidy: a directory nothing names is a concluded pass whose answer already lives in the
comment, the doc or the CHANGELOG entry it produced, and it gets deleted. The run is still in the
history and on the branch that made it.

So: cite the artifact from the line it sets. An uncited directory is not protected by being useful.
