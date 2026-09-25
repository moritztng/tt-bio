#!/usr/bin/env bash
# of3t-inproj: inference fold A/B for the tenstorrent.py change. base = this tree with
# tenstorrent.py from dce82cc17 (the row's starting point); fix = HEAD. Interleaved base, fix,
# base, fix per model, so base/base is the A/A floor. Same card, same seed, cached MSAs.
# AICLK is polled on the card during each fold. -> $S/ab/<model>_<arm><rep>/ and ab.log
set -u
W=/home/ttuser/.coworker/wt/of3t-inproj
S=/home/ttuser/of3t_inproj
CARD=${1:-3}
MSA=/home/ttuser/.coworker/artifacts/tt-bio-full-gate-post-k10/gate-f072ae02f/msa
SMI=/home/ttuser/.local/bin/tt-smi
PY=/home/ttuser/tt-bio-dev/env/bin/python3
rm -rf $S/ab_base $S/ab && mkdir -p $S/ab_base $S/ab
rsync -a --exclude __pycache__ $W/tt_bio $S/ab_base/
git -C $W show dce82cc17:tt_bio/tenstorrent.py > $S/ab_base/tt_bio/tenstorrent.py
while pgrep -f "tt_bio.main predict examples/prot.yaml" >/dev/null; do sleep 5; done
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:of3t-inproj
for model in openfold3 protenix-v2; do
  case $model in
    openfold3) M=$MSA/openfold3__prot__msa-colabfold_200step_5sample_4cycle_fp32cpu ;;
    protenix-v2) M=$MSA/protenix-v2__prot__msa-server_200step_5sample_10cycle_bf16 ;;
  esac
  for rep in 1 2; do
    for arm in base fix; do
      root=$([ $arm = base ] && echo $S/ab_base || echo $W)
      out=$S/ab/${model}_${arm}${rep}; rm -rf $out; mkdir -p $out
      ( while :; do env -u TT_VISIBLE_DEVICES $SMI -s 2>/dev/null | python3 -c "
import sys,json;print(json.load(sys.stdin)['device_info'][$CARD]['telemetry']['aiclk'].strip())" >> $out/aiclk.txt; sleep 1; done ) &
      poll=$!
      t0=$(date +%s.%N)
      # cwd = the arm's own tree: `python -m` puts the cwd ahead of PYTHONPATH.
      (cd $root && $PY -m tt_bio.main predict $W/examples/prot.yaml --model $model \
        --out_dir $out --msa_dir $M --msa_cache_only --diffusion_samples 5 --sampling_steps 200 \
        --seed 0 --output_format cif --override > $out/log.txt 2>&1)
      rc=$?
      t1=$(date +%s.%N)
      kill $poll 2>/dev/null; wait $poll 2>/dev/null
      dig=$(find $out -name "*.cif" | sort | xargs cat | sha256sum | cut -c1-16)
      imp=$(cd $root && $PY -c "import importlib.util as u;print(u.find_spec('tt_bio').origin)" 2>/dev/null)
      tsha=$(sha256sum $root/tt_bio/tenstorrent.py | cut -c1-12)
      clk=$(python3 -c "
import statistics as s;v=[int(x) for x in open('$out/aiclk.txt').read().split() if x.isdigit()]
print(f'min {min(v)} median {int(s.median(v))} max {max(v)} n {len(v)}' if v else 'none')")
      echo "$model $arm$rep rc=$rc wall=$(python3 -c "print(f'{$t1-$t0:.1f}')")s cif=$dig ncif=$(find $out -name '*.cif' | wc -l) aiclk[$clk] pkg=$imp tenstorrent.py=$tsha" | tee -a $S/ab.log
    done
  done
done
echo "=== ab done $(date -u +%FT%TZ)" >> $S/ab.log
