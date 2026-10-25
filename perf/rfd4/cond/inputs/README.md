# RFD4 condition-family inputs

The seven structures the eight design-family tests run on, copied out of the upstream clone by
`perf/rfd4/cond/families.py stage` and committed so the tests run in a fresh checkout. They used to
be gitignored, which meant `tests/test_rfd4_families.py` skipped everywhere except a machine that
had staged them by hand -- including the release gate, so eight families counted as covered by a
test nobody ran.

176 KB for all seven. From NVIDIA's RFD4-Proteina, BSD 3-Clause, Copyright (c) 2026 Institute for
Protein Design, University of Washington; the same upstream `tt_bio/_vendor/rfd4_prep/` comes from
(see NOTICE). `families.py stage` still refreshes them from a clone when you have one.

`partial_diffusion` has no file here: it starts from a design a previous run produced, so its input
is named rather than shipped.
