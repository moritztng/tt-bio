"""Controlled warm wall-clock benchmark: OpenDDE fold trace OFF vs ON, same
process / same seed / same resident cache, on the 7ROA target at production
defaults (10 cycles / 200 steps). Mirrors scripts/opendde_fusion_scout.py.

Reports device-synced total fold time and diffusion-only time for each path so
the trace-replay win is honest (diffusion-only isolates the dispatch collapse;
total is the user-facing number).

Run:
    TT_VISIBLE_DEVICES=1 TT_MESH_GRAPH_DESC_PATH=/tmp/single_bh_mesh_graph_descriptor.textproto \
      TT_BIO_TRACE_REGION_SIZE=1073741824 TT_LOGGER_LEVEL=FATAL \
      PYTHONPATH=$PWD /home/ttuser/tt-bio-dev/env/bin/python3 \
      -m perf.opendde_trace_step_parity.bench
"""
import os
os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")

import json
import time

import torch
import ttnn

import tt_bio.protenix as _P
from tt_bio.tenstorrent import get_device
from tt_bio.opendde import OpenDDE, load_opendde_checkpoint
from tt_bio.protenix_data import build_complex_features

torch.set_grad_enabled(False)

SEQ_7ROA = ("QLEDSEVEAVAKGLEEMYANGVTEDNFKNYVKNNFAQQEISSVEEELNVNISDSCVANKIKDEFFAMISISAIVKAAQKKA"
            "WKELAVTVLRFAKANGLKTNAIIVAGQLALWAVQCG")


def _sync(dev):
    ttnn.synchronize_device(dev)


def main():
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
    sd = load_opendde_checkpoint()
    model = OpenDDE(sd, ckc, dev)
    feats = build_complex_features([(SEQ_7ROA, None, "protein")])
    n_step = int(os.environ.get("OPENDDE_NSTEP", "200"))
    n_cycles = int(os.environ.get("OPENDDE_NCYCLES", "10"))
    seed = int(os.environ.get("OPENDDE_SEED", "0"))

    # diffusion-only timing: wrap edm_sample forwarding every kwarg.
    diff_t = {"s": 0.0}
    orig_sample = _P.edm_sample

    def timed_sample(*a, **k):
        _sync(dev)
        s = time.perf_counter()
        r = orig_sample(*a, **k)
        _sync(dev)
        diff_t["s"] += time.perf_counter() - s
        return r

    _P.edm_sample = timed_sample

    def measure(trace):
        diff_t["s"] = 0.0
        _sync(dev)
        s0 = time.perf_counter()
        coords = model.fold(feats, n_step=n_step, n_cycles=n_cycles, seed=seed, trace=trace)
        _sync(dev)
        total = time.perf_counter() - s0
        return {"total_s": total, "diffusion_s": diff_t["s"],
                "finite": bool(torch.isfinite(coords).all().item())}

    # warm both paths (compile + capture the trace once for trace=True; the
    # captured graph is keyed on N and reused across all subsequent traced folds).
    model.fold(feats, n_step=2, n_cycles=1, seed=seed, trace=False)
    model.fold(feats, n_step=2, n_cycles=1, seed=seed, trace=True)

    K = int(os.environ.get("OPENDDE_BENCH_K", "5"))
    off, on = [], []
    for k in range(K):
        r = measure(False); off.append(r)
        print(f"[{k}] off  total={r['total_s']:.3f} diff={r['diffusion_s']:.3f}", flush=True)
        r = measure(True); on.append(r)
        print(f"[{k}] on   total={r['total_s']:.3f} diff={r['diffusion_s']:.3f}", flush=True)

    import statistics as st
    def med(xs): return st.median(xs)
    def mean(xs): return st.mean(xs)
    off_tot = [r["total_s"] for r in off]; on_tot = [r["total_s"] for r in on]
    off_diff = [r["diffusion_s"] for r in off]; on_diff = [r["diffusion_s"] for r in on]
    d_diff = med(off_diff) - med(on_diff)
    d_tot = med(off_tot) - med(on_tot)
    summary = {
        "target": "7ROA", "n_step": n_step, "n_cycles": n_cycles, "seed": seed, "K": K,
        "diffusion_off_med_s": round(med(off_diff), 4), "diffusion_on_med_s": round(med(on_diff), 4),
        "diffusion_off_mean_s": round(mean(off_diff), 4), "diffusion_on_mean_s": round(mean(on_diff), 4),
        "diffusion_win_pct_median": round(100.0 * d_diff / med(off_diff), 2) if med(off_diff) else None,
        "total_off_med_s": round(med(off_tot), 4), "total_on_med_s": round(med(on_tot), 4),
        "total_win_pct_median": round(100.0 * d_tot / med(off_tot), 2) if med(off_tot) else None,
        "off_diff_samples": [round(x, 3) for x in off_diff],
        "on_diff_samples": [round(x, 3) for x in on_diff],
        "off_total_samples": [round(x, 3) for x in off_tot],
        "on_total_samples": [round(x, 3) for x in on_tot],
        "all_finite": all(r["finite"] for r in off + on),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
