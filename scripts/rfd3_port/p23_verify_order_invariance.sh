#!/bin/sh
# p23 acceptance: the fixed tree must fold every design identically no matter what ran
# before it in the process, and identically to the pre-fix tree folding it alone.
set -e
WT=/home/ttuser/.coworker/wt/tt-bio-rfdiffusion3-largedesign-gap-p23
MAIN=/tmp/p23_main
OUT=/tmp/p23/verify
mkdir -p $OUT
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:tt-bio-rfdiffusion3-largedesign-gap-p23
export TT_MESH_GRAPH_DESC_PATH=$HOME/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
PY=~/tt-bio-dev/env/bin/python3
H=scripts/rfd3_port/p22_sequenced_contamination.py
C40="A1-10,20,A31-40"
C150="A1-10,130,A31-40"
C250="A1-10,230,A31-40"

run() {           # run <tree> <tag> <extra args...>
  tree=$1; tag=$2; shift 2
  PYTHONPATH=$tree $PY $tree/$H --tree $tree --out $OUT/$tag.pt --batches 1 8 "$@" \
      > $OUT/$tag.log 2>&1 || { echo "FAILED $tag"; tail -25 $OUT/$tag.log; exit 1; }
  echo "  ok $tag: $(grep -c 'D=' $OUT/$tag.log) cells"
}

MODE=${RFD3_SPARSE_QK:-1}
echo "=== isolated references, pre-fix tree (RFD3_SPARSE_QK=$MODE) ==="
run $MAIN ref_40   --contigs "$C40"
run $MAIN ref_150  --contigs "$C150"
run $MAIN ref_250  --contigs "$C250"
run $MAIN ref_mpro --specs $MAIN/scripts/rfd3_port/parity_artifacts/enzyme_mpro/spec.json
echo "=== isolated, fixed tree ==="
run $WT fix_40   --contigs "$C40"
run $WT fix_150  --contigs "$C150"
run $WT fix_250  --contigs "$C250"
run $WT fix_mpro --specs $WT/scripts/rfd3_port/parity_artifacts/enzyme_mpro/spec.json
MPRO=$WT/scripts/rfd3_port/parity_artifacts/enzyme_mpro/spec.json
echo "=== sequenced, fixed tree, four orderings ==="
run $WT seq_asc  --contigs "$C40" "$C150" "$C250" --specs $MPRO
run $WT seq_desc --contigs "$C250" "$C150" "$C40" --specs $MPRO
run $WT seq_mid  --contigs "$C150" "$C40" "$C250" --specs $MPRO
run $WT seq_big1 --contigs "$C250" "$C40" "$C150" --specs $MPRO
echo "=== sequenced, PRE-FIX tree (control: must show the bug) ==="
run $MAIN pre_asc --contigs "$C40" "$C150" "$C250" --specs $MPRO
