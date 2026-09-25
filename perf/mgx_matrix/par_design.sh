#!/bin/bash
# par_design.sh <card> <run_np.sh pid>: the rfd3 and pxdesign cases of run_np.sh's design surface
# on a second chip, into out_np_par/, while the boltzgen cases finish on the first. A watcher
# stops that run_np.sh as soon as boltzgen is done, killing its next case (rfd3 binder) by pid
# while it is still importing, long before a device open, so no case runs twice.
set -u
cd "$(dirname "$0")/../.."
C=$1 NP=$2
O=perf/mgx_matrix/out_np
(
  L=$O/design/boltzgen/bg_small_molecule.log
  until [ -f "$L" ] && grep -q '^EXIT=' "$L"; do kill -0 "$NP" 2>/dev/null || exit 0; sleep 1; done
  kill -STOP "$NP"; kids=$(pgrep -P "$NP"); kill "$NP"; kill -CONT "$NP"
  for p in $kids; do
    grep -q rfd3_binder "/proc/$p/cmdline" 2>/dev/null && kill "$p"
  done
  sleep 5; rm -rf "$O/design/rfd3/binder" "$O/design/rfd3/binder.log"
) &
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:mgx-matrix
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxm
export TT_METAL_LOGGER_LEVEL=FATAL
I=perf/mgx_matrix/np_inputs P=perf/mgx_matrix/out_np_par
case_() {
  local tag=$1; shift
  mkdir -p "$P/$tag"
  local start; start=$(date +%s)
  $HOME/env/bin/python -m tt_bio.main "$@" > "$P/$tag.log" 2>&1
  echo "EXIT=$? WALL=$(( $(date +%s) - start ))s" >> "$P/$tag.log"
}
case_ design/rfd3/binder design "$I/rfd3_binder.json" --model rfd3 --from_pdb --num_designs 1 --out_dir "$P/design/rfd3/binder"
case_ design/rfd3/motif design "$I/rfd3_motif.json" --model rfd3 --from_pdb --num_designs 1 --out_dir "$P/design/rfd3/motif"
case_ design/pxdesign/pdl1 design "$I/px_pdl1.yaml" --model pxdesign --num_designs 1 --out_dir "$P/design/pxdesign/pdl1"
case_ design/pxdesign/nohotspot design "$I/px_nohotspot.yaml" --model pxdesign --num_designs 1 --out_dir "$P/design/pxdesign/nohotspot"
wait
