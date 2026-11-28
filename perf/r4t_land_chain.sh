#!/bin/bash
# One chip, in order: INFERENCE-AB (r4t_land_ab.sh) then the release arms (r4t_land_rg.sh) on the landing sha.
#   bash r4t_land_chain.sh <chip> <outdir> <main_sha> <step_sha> <land_sha>
set -u
N=${1:?chip}; R=${2:?outdir}; MAIN=${3:?}; STEP=${4:?}; LAND=${5:?}
H=$(cd "$(dirname "$0")" && pwd)
bash "$H/r4t_land_ab.sh" "$N" "$R/ab" "$MAIN" "$STEP" "$LAND"; echo "ab rc=$?"
bash "$H/r4t_land_rg.sh" "$N" "$R/rg" "$LAND"; echo "rg rc=$?"
