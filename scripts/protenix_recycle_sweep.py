#!/usr/bin/env python3
"""Decisive test: does running Protenix-v2's trunk at its SPEC recycle count (N_CYCLES=10)
instead of the shipped CLI default (--recycling_steps=3) fix the delivered-RMSD gap?

Root cause found (docs/protenix-confidence-head-rootcause.md's anti-ranking was a SYMPTOM):
the shared CLI flag --recycling_steps defaults to 3 (the Boltz/AF3 convention), but
protenix.Trunk.N_CYCLES=10 is protenix-v2's spec. Under-recycling leaves the trunk pair
repr unconverged -> a bimodal ensemble (some samples in the wrong basin) that the confidence
head then anti-ranks. The prior investigation's own harness omitted n_cycles (used 10) and
saw a good tight ensemble (pTM 0.83); the CLI used 3 and saw the hard bimodal one (pTM 0.73).

Folds each target at n_cycles in {3,10}, n_sample=5, seed=0 (matching the committed CLI run),
mirrors worker._predict_protenix_one's feats + _score exactly, and reports DELIVERED RMSD
(best-by-confidence pick) vs oracle. Reuses tests/test_structure.compute_rmsd verbatim.
"""
import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES', '0'); os.environ.setdefault('TT_LOGGER_LEVEL', 'FATAL')
WT = '/home/ttuser/.coworker/wt/tt-bio-protenix-recycling-revisit'
sys.path.insert(0, WT)
import importlib.util, shutil
from pathlib import Path
import numpy as np, torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import Protenix
from tt_bio.protenix_data import build_complex_features
from tt_bio.main import _read_bio_chains, _read_bio_constraints, _resolve_a3m_text, _write_protenix_structure

TARGETS = {
    'prot':       ('examples/prot.yaml',       '/tmp/recycle_revisit/msa'),      # 7ROA monomer, shallow MSA (hard)
    'hemoglobin': ('examples/hemoglobin.yaml', '/tmp/recycle_revisit/msa_hemo'), # a2b2 tetramer, deep MSA (easy control)
}
CYCLES = [3, 10]
N_SAMPLE, N_STEP, SEED = 5, 200, 0
# argv: [targets_csv] [cycles_csv] [n_step] — e.g. "prot" "3,10" 200
if len(sys.argv) > 1 and sys.argv[1]:
    TARGETS = {k: TARGETS[k] for k in sys.argv[1].split(',')}
if len(sys.argv) > 2 and sys.argv[2]:
    CYCLES = [int(x) for x in sys.argv[2].split(',')]
if len(sys.argv) > 3 and sys.argv[3]:
    N_STEP = int(sys.argv[3])

def _score(c):  # verbatim from worker._predict_protenix_one
    ptm, iptm = c.get("ptm", 0.0), c.get("iptm", 0.0)
    if iptm > 0.0:
        return 0.8 * iptm + 0.2 * ptm
    return ptm if ptm > 0.0 else c["plddt"]

def spearman(x, y):
    def rk(a):
        o = np.argsort(a); r = np.empty(len(a)); r[o] = np.arange(len(a)); return r
    rx, ry = rk(np.array(x, float)), rk(np.array(y, float)); rx -= rx.mean(); ry -= ry.mean()
    d = np.linalg.norm(rx) * np.linalg.norm(ry); return float((rx * ry).sum() / d) if d else 0.0

spec = importlib.util.spec_from_file_location('ts', f'{WT}/tests/test_structure.py')
ts = importlib.util.module_from_spec(spec); spec.loader.exec_module(ts)

dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
model = Protenix.load_from_checkpoint('/home/ttuser/protenix_ckpt/protenix-v2.pt',
                                      compute_kernel_config=ckc, device=dev)

def gt_rmsd(name, coords, feats, confs):
    stage = Path(f'/tmp/recycle_revisit/sweep_stage_{name}')
    sdir = stage / f'boltz_results_{name}' / 'structures'
    shutil.rmtree(stage, ignore_errors=True); sdir.mkdir(parents=True)
    (stage / 'examples' / 'ground_truth_structures').mkdir(parents=True)
    shutil.copy(f'{WT}/examples/ground_truth_structures/{name}.cif',
                stage / 'examples' / 'ground_truth_structures' / f'{name}.cif')
    for k in range(len(coords)):
        out = sdir / (f'{name}.cif' if k == 0 else f'{name}_model_{k}.cif')
        _write_protenix_structure(coords[k], feats, None, out, 'mmcif',
                                  b_factors=confs[k]['plddt_atom'] * 100.0)
    cwd = os.getcwd(); os.chdir(stage)
    try:
        return [ts.compute_rmsd(name, k)[0] for k in range(len(coords))]
    finally:
        os.chdir(cwd)

results = {}
for name, (yaml_rel, msa_dir) in TARGETS.items():
    path = Path(WT) / yaml_rel
    chains = _read_bio_chains(path)
    bonds = _read_bio_constraints(path)
    chain_specs = [(cseq, _resolve_a3m_text(spec_, cseq, Path(msa_dir)) if mt == 'protein' else None, mt)
                   for _cid, cseq, spec_, mt in chains]
    feats = build_complex_features(chain_specs, mol_dir=None,
                                   chain_ids=[cid for cid, _s, _sp, _mt in chains], bonds=bonds)
    for nc in CYCLES:
        coords, conf = model.fold(feats, n_step=N_STEP, n_sample=N_SAMPLE, seed=SEED,
                                  return_confidence=True, n_cycles=nc)
        confs = conf if isinstance(conf, list) else [conf]
        gt = gt_rmsd(name, coords, feats, confs)
        sc = [_score(c) for c in confs]
        deliver_k = int(np.argmax(sc))
        results[(name, nc)] = (gt, sc, deliver_k)
        print(f"\n=== {name}  n_cycles={nc} ===", flush=True)
        print(f"{'k':>2}{'score':>9}{'ptm':>8}{'iptm':>8}{'plddt':>8}{'gt_rmsd':>9}", flush=True)
        for k, (c, g) in enumerate(zip(confs, gt)):
            star = ' <-delivered' if k == deliver_k else ''
            print(f"{k:>2}{sc[k]:>9.4f}{c.get('ptm',0):>8.4f}{c.get('iptm',0):>8.4f}"
                  f"{c['plddt']:>8.4f}{g:>9.3f}{star}", flush=True)
        print(f"  DELIVERED (best-by-score) = {gt[deliver_k]:.3f}A   oracle = {min(gt):.3f}A   "
              f"spread {min(gt):.3f}-{max(gt):.3f}A   spearman(score,-rmsd) = {spearman(sc,[-x for x in gt]):+.3f}",
              flush=True)

print("\n" + "=" * 72, flush=True)
print(f"{'target':>12}{'delivered@3':>14}{'delivered@10':>14}{'oracle@10':>12}", flush=True)
for name in TARGETS:
    g3, _, k3 = results[(name, 3)]; g10, _, k10 = results[(name, 10)]
    print(f"{name:>12}{g3[k3]:>13.3f}A{g10[k10]:>13.3f}A{min(g10):>11.3f}A", flush=True)
print("RECYCLE_SWEEP_DONE", flush=True)
