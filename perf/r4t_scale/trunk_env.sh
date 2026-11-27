# Shared by the r4t-scale trunk arms: r4t-step real-step settings (run_real.sh) and its nine crop-256 batches in its order.
WT=$HOME/.coworker/wt/r4t-scale; cd $WT; export PYTHONPATH=$WT TT_BIO_SDPA_OWN_FORWARD=1 TT_BIO_LEASE_HOLDER=worker:r4t-scale
PY=$HOME/scratch/shipflow/venv-c4f99e695d34/bin/python; S=$HOME/r4t/step
B=$S/batch08.npz,$S/batch01.npz,$S/batch05.npz,$S/batch00.npz,$S/batch03.npz,$S/batch06.npz,$S/batch02.npz,$S/batch04.npz,$S/batch07.npz
trunk(){ timeout 2400 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --batch $B --exempt perf/r4t_step/g2_exempt.tsv "$@"; }
