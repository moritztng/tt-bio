"""Stock AF2 initial guess on a GPU: ColabDesign's binder protocol, the input tt-bio's af2ig reproduces.

model_1_ptm, single sequence, the designed complex's own coordinates as the initial guess, 3 recycles (4 passes),
template = backbone+CB with the target sequence and the interchain template removed (tt-bio af2_data.complex_features
defaults). One process; per complex one cold predict (JIT) then --warm timed predicts. Prints one JSON line per rep.
usage: python af2ig_gpu.py --params DIR(with params/) --warm 5 design1.pdb [design2.pdb ...]
"""
import argparse, json, subprocess, threading, time

import numpy as np
from colabdesign import mk_afdesign_model

AA3 = dict(ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G", HIS="H", ILE="I", LEU="L", LYS="K",
           MET="M", PHE="F", PRO="P", SER="S", THR="T", TRP="W", TYR="Y", VAL="V")


def chain_seq(pdb, chain):
    return "".join(AA3[l[17:20]] for l in open(pdb) if l.startswith("ATOM") and l[12:16] == " CA " and l[21] == chain)


def sm_sampler(out, stop):
    while not stop.is_set():
        r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip()
        if r.isdigit():
            out.append(int(r))
        time.sleep(0.5)


ap = argparse.ArgumentParser()
ap.add_argument("--params", required=True)
ap.add_argument("--warm", type=int, default=5)
ap.add_argument("pdbs", nargs="+")
a = ap.parse_args()
model = mk_afdesign_model(protocol="binder", use_multimer=False, use_initial_guess=True, num_recycles=3,
                          data_dir=a.params, model_names=["model_1_ptm"])
for pdb in a.pdbs:
    seq = chain_seq(pdb, "B")
    model.prep_inputs(pdb_filename=pdb, chain="A", binder_chain="B", binder_len=len(seq), use_binder_template=True,
                      rm_target_seq=True, rm_target_sc=False, rm_template_ic=True)
    for rep in range(a.warm + 1):
        sm, stop = [], threading.Event()
        th = threading.Thread(target=sm_sampler, args=(sm, stop)); th.start()
        t = time.perf_counter()
        model.predict(seq=seq, num_recycles=3, num_models=1, verbose=False)
        aux = model.aux
        lg = {k: float(np.asarray(aux["log"][k])) for k in ("plddt", "ptm", "i_ptm", "i_pae") if k in aux["log"]}
        dt = time.perf_counter() - t
        stop.set(); th.join()
        print(json.dumps(dict(pdb=pdb, tokens=len(chain_seq(pdb, "A")) + len(seq), binder=len(seq), rep=rep,
                              kind="cold" if rep == 0 else "warm", s=round(dt, 4), sm_median=float(np.median(sm)) if sm else None,
                              **lg)), flush=True)
