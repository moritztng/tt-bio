#!/bin/bash
# Launcher proga's TTBioOracle shells out to (--ttbio-launcher). proga runs on its own python
# 3.12 venv; tt-bio runs from the pinned tree in the shipflow 3.10 venv. Scrub proga's
# PYTHONPATH so nothing from the reviewed checkout is importable inside the tt-bio process.
set -u
exec env PYTHONPATH="$TREE" "$VENV/bin/python" -m tt_bio.main "$@"
