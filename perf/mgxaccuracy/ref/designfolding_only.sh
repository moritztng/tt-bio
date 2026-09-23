#!/bin/bash
# Run ONLY the step that produces the scRMSD, against an upstream run that already has its
# inverse-folded designs on disk. $1 = size tag (512 / 1536).
#
# Why this exists. The upstream CPU reference walks six steps and only two of them matter for
# designability: design (step 1) makes the binder, design_folding (step 4) refolds it alone,
# and analysis (step 5) does nothing but Kabsch-align the two -- which
# `perf/mgxaccuracy/scrmsd.py` reproduces exactly, checked to 0.00000 A on the 8-design whglx
# 512 run. Between them sits `folding` (step 3), which refolds the WHOLE COMPLEX and is the
# most expensive step at a large target: measured on qb2, the 512 reference spent over two
# hours in it for one design and had written nothing. At 1616 tokens it is what puts an
# upstream 1536 number out of reach in a day.
#
# Skipping it is sound because step 4 does not consume step 3's output. The two step configs
# differ by exactly two lines -- `return_designfolding: true` and `designfolding: true` -- and
# read the SAME `design_dir`. Diffed, not assumed.
#
# This is not a shortcut on the model's own work: the design is sampled at upstream defaults
# (sampling_steps 500, recycling_steps 3) and refolded by the same checkpoint at fp32. What is
# dropped is a complex-prediction metric that this comparison never reads.
#
# The staged copy leaves the original run untouched, so it can be started while the six-step
# run is still walking step 3: both would otherwise write `metrics_tmp/data_<id>.npz`.
set -u
size=${1:?size tag}
thr=${2:-8}
src=$HOME/bgref-work/out${size}
dst=$HOME/bgref-work/out${size}_df
inv=intermediate_designs_inverse_folded

[ -d "$src/$inv" ] || { echo "REFUSED: $src/$inv missing — inverse_folding has not run"; exit 1; }
ls "$src/$inv"/*.cif >/dev/null 2>&1 || { echo "REFUSED: no inverse-folded design in $src/$inv"; exit 1; }

rm -rf "$dst"
mkdir -p "$dst/config" "$dst/$inv" "$dst/intermediate_designs"
cp "$src/$inv"/*.cif "$src/$inv"/*.npz "$dst/$inv/" 2>/dev/null
# scrmsd.py aligns the refold to the DESIGNED backbone, which lives in intermediate_designs/.
# Staging only the inverse-folded inputs runs the fold fine and then cannot score it.
cp "$src/intermediate_designs"/*.cif "$src/intermediate_designs"/*.npz \
   "$dst/intermediate_designs/" 2>/dev/null
sed "s#${src}#${dst}#g" "$src/config/design_folding.yaml" > "$dst/config/design_folding.yaml"

export OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr NUMEXPR_NUM_THREADS=$thr
export CUDA_VISIBLE_DEVICES="" TT_VISIBLE_DEVICES=""
t0=$(date +%s)
$HOME/bgref-env/bin/python3 \
    $HOME/bgref-env/lib/python3.12/site-packages/boltzgen/resources/main.py \
    "$dst/config/design_folding.yaml"
rc=$?
echo "DF_DONE size=${size} rc=${rc} wall_s=$(( $(date +%s) - t0 )) out=${dst}"
echo "score with: python3 perf/mgxaccuracy/scrmsd.py ${dst} --json"
