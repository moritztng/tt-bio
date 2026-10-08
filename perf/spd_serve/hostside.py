"""Host-side cost of one Protenix-v2 fold in the serving worker, CPU only (no device is opened).

    python hostside.py INPUT.yaml MSA_DIR OUT_DIR [--share 32] [--reps 3] [--samples 5] [--msa_db DB]

Times what a JapanFold chip worker does around the device fold, on the thread share the
agent gives each worker (host_thread_cap_env(share)): read + featurise the target with its
cached MSAs (_protenix_inputs), then rank and write every sample's structure plus the PAE
export (_protenix_emit). The coordinates and confidences are placeholders of the real
shapes: the writers' cost depends on the sizes, not the values.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("input"); ap.add_argument("msa_dir"); ap.add_argument("out")
ap.add_argument("--share", type=int, default=32); ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--samples", type=int, default=5); ap.add_argument("--msa_db", default=os.path.expanduser("~/japanfold/msa/db")); ap.add_argument("--fmt", default="cif")
a = ap.parse_args()

from tt_bio import runtime  # noqa: E402
os.environ.update(runtime.host_thread_cap_env(a.share, None))
import torch  # noqa: E402
from tt_bio import worker as W  # noqa: E402

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
# The worker config exactly as `tt-bio predict` builds it for this input (JapanFold's command line).
from tt_bio import main as M  # noqa: E402
captured = {}


class _Stop(Exception):
    pass


def _grab(*args, **_kw):
    captured["payload"] = args[1] if isinstance(args[0], str) else args[0]
    raise _Stop


M._dispatch_run = M._dispatch_to_controller = _grab
M._local_workers = lambda *_a, **_k: [object()]   # enumerating chips is not this tool's business
M._cap_worker_threads = lambda *_a, **_k: None
argv = ["predict", a.input, "--model", "protenix-v2", "--diffusion_samples", str(a.samples),
        "--accelerator", "tenstorrent", "--output_format", a.fmt, "--write_pae",
        "--msa_db_path", a.msa_db, "--msa_dir", a.msa_dir, "--out_dir", str(out / "cli")]
try:
    M.cli.main(argv, standalone_mode=False)
except _Stop:
    pass
cfg = {**captured["payload"]["config"], "struct_dir": str(out)}
W._ensure_local_artifacts(cfg)   # what the worker does with a leased run's config
state = W._WorkerState.__new__(W._WorkerState)
rows = []
for rep in range(a.reps):
    t0 = time.perf_counter()
    feats, chains, specs = state._protenix_inputs(Path(a.input), cfg)
    t1 = time.perf_counter()
    n_tok, n_atom = int(feats["restype"].shape[0]), int(feats["ref_pos"].shape[0])
    g = torch.Generator().manual_seed(rep)
    coords = [torch.randn(n_atom, 3, generator=g) * 20 for _ in range(a.samples)]
    confs = [{"plddt": 0.8, "ptm": 0.7, "iptm": 0.6, "plddt_atom": torch.rand(n_atom, generator=g),
              "pae": torch.rand(n_tok, n_tok, generator=g) * 30, "pde": torch.rand(n_tok, n_tok, generator=g) * 30,
              "chain_pair_iptm": torch.rand(len(chains), len(chains), generator=g)}
             for _ in range(a.samples)]
    t2 = time.perf_counter()
    state._protenix_emit(Path(a.input), cfg, feats, chains, specs, coords, confs)
    t3 = time.perf_counter()
    rows.append({"rep": rep, "featurise_s": round(t1 - t0, 3), "emit_s": round(t3 - t2, 3),
                 "tokens": n_tok, "atoms": n_atom, "msa_depth": int(feats["msa"].shape[0]),
                 "threads": os.environ.get("OMP_NUM_THREADS")})
    print(json.dumps(rows[-1]), flush=True)
