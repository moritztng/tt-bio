#!/bin/bash
# Run ONLY the step that produces the scRMSD, on qb2, against designs already on disk.
#
#   bash designfolding_only.sh --size 1536                 # upstream's own designs
#   bash designfolding_only.sh --from DIR --name devrefold # designs made somewhere else
#
# Why this exists. The upstream CPU reference walks six steps and designability needs two of
# them: design (step 1) makes the binder, design_folding (step 4) refolds it alone, and
# analysis (step 5) only Kabsch-aligns the two -- which `perf/mgxaccuracy/scrmsd.py`
# reproduces exactly, checked to 0.00000 A on the 8-design whglx 512 run. Between them sits
# `folding` (step 3), which refolds the WHOLE COMPLEX and is the expensive step at a large
# target: measured on qb2, the 512 reference sat in it over two hours on one design with
# nothing written, and at 1616 tokens it is what puts an upstream 1536 number out of reach.
#
# Skipping it is sound because step 4 does not consume step 3's output. The two step configs
# differ by exactly two lines -- `return_designfolding: true` and `designfolding: true` -- and
# read the SAME `design_dir`. Diffed, not assumed.
#
# This is not a shortcut on the model's own work: the design is sampled at upstream defaults
# (sampling_steps 500, recycling_steps 3) and refolded by the same checkpoint at fp32. What is
# dropped is a complex-prediction metric this comparison never reads.
#
# --from is the second use and the more informative one. Step 4's cost is set by the BINDER,
# 80 residues, not by the target, so refolding the DEVICE's designs here costs the same at
# 1536 as at 512 -- about 17 minutes per design, measured. That gives a paired device-against-
# upstream comparison on identical designs, which separates "the designs are worse" from "the
# refolder that scores them is worse", at a price the design step itself never reaches. The
# device's inverse-folded output is accepted by the upstream pipeline as-is; verified.
#
# Always into a staged copy, never in place: the six-step run and this one both write
# `metrics_tmp/data_<id>.npz`, so sharing a directory corrupts whichever finishes second.
set -u
size=""; from=""; name=""; thr=8
while [ $# -gt 0 ]; do
    case "$1" in
        --size) size=$2; shift 2;;
        --from) from=$2; shift 2;;
        --name) name=$2; shift 2;;
        --threads) thr=$2; shift 2;;
        *) echo "unknown argument $1"; exit 2;;
    esac
done

inv=intermediate_designs_inverse_folded
# The config template is any completed upstream CPU step-4 config: it carries accelerator=cpu,
# precision=32 and the checkpoint paths, and only its design_dir is rewritten.
tmpl=""
for c in "$HOME"/bgref-work/out*/config/design_folding.yaml; do [ -f "$c" ] && tmpl=$c; done
[ -n "$tmpl" ] || { echo "REFUSED: no upstream design_folding.yaml to use as a template"; exit 1; }

if [ -n "$size" ]; then
    src=$HOME/bgref-work/out${size}
    dst=$HOME/bgref-work/out${size}_df
elif [ -n "$from" ]; then
    [ -n "$name" ] || { echo "REFUSED: --from needs --name"; exit 1; }
    src=$from
    dst=$HOME/bgref-work/out_${name}
else
    echo "REFUSED: need --size or --from"; exit 1
fi

[ -d "$src/$inv" ] || { echo "REFUSED: $src/$inv missing — inverse_folding has not run"; exit 1; }
ls "$src/$inv"/*.cif >/dev/null 2>&1 || { echo "REFUSED: no inverse-folded design in $src/$inv"; exit 1; }

rm -rf "$dst"
mkdir -p "$dst/config" "$dst/$inv" "$dst/intermediate_designs"
cp "$src/$inv"/*.cif "$src/$inv"/*.npz "$dst/$inv/" 2>/dev/null
# scrmsd.py aligns the refold to the DESIGNED backbone, which lives in intermediate_designs/.
# Staging only the inverse-folded inputs runs the fold fine and then cannot score it.
cp "$src/intermediate_designs"/*.cif "$src/intermediate_designs"/*.npz \
   "$dst/intermediate_designs/" 2>/dev/null
sed "s#^\(\s*design_dir:\).*#\1 ${dst}/${inv}#; s#^\(output:\).*#\1 ${dst}/${inv}#" \
    "$tmpl" > "$dst/config/design_folding.yaml"
grep -q "$dst" "$dst/config/design_folding.yaml" || {
    echo "REFUSED: the config rewrite did not take — $tmpl does not have the expected keys"
    exit 1; }

export OMP_NUM_THREADS=$thr MKL_NUM_THREADS=$thr OPENBLAS_NUM_THREADS=$thr NUMEXPR_NUM_THREADS=$thr
export CUDA_VISIBLE_DEVICES="" TT_VISIBLE_DEVICES=""
t0=$(date +%s)
$HOME/bgref-env/bin/python3 \
    $HOME/bgref-env/lib/python3.12/site-packages/boltzgen/resources/main.py \
    "$dst/config/design_folding.yaml"
rc=$?
echo "DF_DONE rc=${rc} wall_s=$(( $(date +%s) - t0 )) out=${dst}"
echo "score with: python3 perf/mgxaccuracy/scrmsd.py ${dst} --json"
