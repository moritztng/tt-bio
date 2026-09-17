#!/usr/bin/env bash
# Assemble perf/hall800/msa/ from the two reference-fixture alignments hall800.yaml needs.
# The a3ms are not committed here: one is already tracked under docs/, the other ships in the
# parity-fixtures release asset. Run scripts/fetch_parity_fixtures.sh first if 9ncy is missing.
set -euo pipefail
cd "$(dirname "$0")/../.."
D=perf/hall800/msa
mkdir -p "$D"
# chain A, human serum albumin 585 aa -> sha256(seq)[:16]
cp docs/implementation-parity-data/ref-fixtures/protenix-v2/hsa/msa-server_200step_5sample_10cycle_bf16/msa.a3m "$D/a35bb68136a5125a.a3m"
# chain B, 9ncy light chain 212 aa, already named by its own hash
cp docs/implementation-parity-data/ref-fixtures/protenix-v2/9ncy/msa-campaign_200step_5sample_10cycle_bf16/msa/b6e0ae668b173a0c.a3m "$D/"
ls -l "$D"
