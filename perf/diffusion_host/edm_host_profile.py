#!/usr/bin/env python3
"""Direct per-phase host-time attribution for the EDM diffusion sampler (PERF WAR D5).

W10 reported "11-15 % of the diffusion stage is host sampler arithmetic" by SUBTRACTION:
diffusion-stage wall minus 200 x the median of five synced steady-state denoise steps. Subtraction
attributes every non-steady-state cost in the stage to the host, including the first denoise step
(which builds `_atom_cond` and the DiT per-block pair biases, and misses the ttnn program cache for
~1500 ops). This script measures the host phases directly instead, with a `time.perf_counter()`
around each statement of the sampler loop.

`edm_sample_timed` below is a statement-for-statement copy of `tt_bio.protenix.edm_sample` with
timers added and nothing else. `--selftest` proves that: it runs the real sampler and this copy
against a stub denoise, same seed, and asserts `torch.equal` on the returned coordinates for the
M=1, M>1 and member_seeds paths. Run it before trusting any number this script prints.

    # bit-exactness of the instrumented copy (no device)
    python3 perf/diffusion_host/edm_host_profile.py --selftest

    # real 298 aa fold, host phases attributed
    TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_HOLDER=worker:perfwar-diffusion-host-sampler PYTHONPATH=$PWD \
      python3 perf/diffusion_host/edm_host_profile.py --model protenix-v2 \
        --target examples/prot300.yaml --msa-a3m scripts/gpu_vs_tt/fixtures/prot300.a3m \
        --out perf/diffusion_host/hostphase_protenix-v2_298aa.json
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PC = time.perf_counter

PHASES = ["init", "sched", "aug", "center", "rotate", "rng", "xnoisy", "denoise",
          "hooks", "update", "progress", "dump"]


def _new_acc():
    a = {p: 0.0 for p in PHASES}
    a["loop_wall"] = 0.0
    a["call_wall"] = 0.0
    a["steps"] = 0
    a["step_walls"] = []
    a["denoise_walls"] = []
    return a


def edm_sample_timed(diffusion_module, cond, n_atoms, *, multiplicity=1, max_parallel_samples=None,
                     member_seeds=None, n_step=200, gamma0=0.8, gamma_min=1.0,
                     noise_scale=1.003, step_scale=1.5, sigma_data=16.0, s_max=160.0, s_min=4e-4,
                     rho=7.0, seed=None, trace=False, progress_fn=None, dump_fn=None, _acc=None):
    """Statement-for-statement copy of tt_bio.protenix.edm_sample with per-phase host timers."""
    import torch
    from tt_bio.boltz2 import compute_random_augmentation
    from tt_bio.protenix import dram_peak, trunk_tap_host
    acc = _acc if _acc is not None else _new_acc()
    t_call0 = PC()
    t0 = PC()
    M = max(1, int(multiplicity))
    if max_parallel_samples is None or max_parallel_samples > M:
        max_parallel_samples = M
    max_parallel_samples = max(1, int(max_parallel_samples))
    _denoise = (diffusion_module.denoise_traced if trace and M == 1 else diffusion_module.denoise)
    if seed is not None:
        torch.manual_seed(seed)
    inv_rho = 1.0 / rho
    i = torch.arange(n_step, dtype=torch.float64)
    sig = sigma_data * (s_max ** inv_rho + (i / n_step) * (s_min ** inv_rho - s_max ** inv_rho)) ** rho
    sigmas = torch.cat([sig, torch.zeros(1, dtype=torch.float64)]).float()
    gammas = torch.where(sigmas > gamma_min, torch.tensor(gamma0), torch.tensor(0.0))
    shape = (M, n_atoms, 3)
    import os as _os
    _sds = _os.environ.get("TT_BIO_SHARED_DRAW_SEED")
    if _sds:
        torch.manual_seed(int(_sds))
    _per_member = None
    if member_seeds is not None:
        assert len(member_seeds) == M, f"member_seeds {len(member_seeds)} != multiplicity {M}"
        _states = []
        for _sd in member_seeds:
            torch.manual_seed(int(_sds) if _sds else (0 if _sd is None else int(_sd)))
            _states.append(torch.get_rng_state())

        def _per_member(fn):
            outs = []
            for _b in range(M):
                torch.set_rng_state(_states[_b])
                outs.append(fn())
                _states[_b] = torch.get_rng_state()
            return outs
        x = sigmas[0] * torch.cat(_per_member(lambda: torch.randn((1, n_atoms, 3))), 0)
    else:
        x = sigmas[0] * torch.randn(shape)
    if dump_fn is not None:
        for _m in range(M):
            dump_fn(-1, x[_m:_m + 1].detach().cpu())
    sample_ids = torch.arange(M)
    n_chunks = max(1, (M + max_parallel_samples - 1) // max_parallel_samples)
    chunks = [c for c in sample_ids.chunk(n_chunks) if c.numel() > 0]
    acc["init"] += PC() - t0
    t_loop0 = PC()
    for k in range(n_step):
        t_step0 = PC()
        t = PC()
        if progress_fn:
            progress_fn("diffusion", step=k, total=n_step)
        acc["progress"] += PC() - t
        t = PC()
        sigma_tm, sigma_t, gamma = sigmas[k].item(), sigmas[k + 1].item(), gammas[k + 1].item()
        acc["sched"] += PC() - t
        t = PC()
        if _per_member is not None:
            _aug = _per_member(lambda: compute_random_augmentation(1, device=x.device, dtype=x.dtype))
            R = torch.cat([a[0] for a in _aug], 0)
            tr = torch.cat([a[1] for a in _aug], 0)
        else:
            R, tr = compute_random_augmentation(M, device=x.device, dtype=x.dtype)
        acc["aug"] += PC() - t
        t = PC()
        x = x - x.mean(dim=-2, keepdim=True)
        acc["center"] += PC() - t
        t = PC()
        x = torch.einsum("bmd,bds->bms", x, R) + tr
        acc["rotate"] += PC() - t
        t = PC()
        t_hat = sigma_tm * (1 + gamma)
        noise_var = noise_scale ** 2 * (t_hat ** 2 - sigma_tm ** 2)
        acc["sched"] += PC() - t
        t = PC()
        if noise_var <= 0:
            eps = torch.zeros(shape)
        elif _per_member is not None:
            eps = (noise_var ** 0.5) * torch.cat(
                _per_member(lambda: torch.randn((1, n_atoms, 3))), 0)
        else:
            eps = (noise_var ** 0.5) * torch.randn(shape)
        acc["rng"] += PC() - t
        t = PC()
        x_noisy = x + eps
        denoised = torch.zeros_like(x_noisy)
        acc["xnoisy"] += PC() - t
        t = PC()
        for _chunk in chunks:
            denoised[_chunk] = _denoise(x_noisy[_chunk], torch.tensor([t_hat], dtype=torch.float32), cond)
        _dn = PC() - t
        acc["denoise"] += _dn
        acc["denoise_walls"].append(_dn)
        t = PC()
        dram_peak(f"edm step {k}")
        trunk_tap_host(f"edm_denoised[step{k}]", denoised)
        acc["hooks"] += PC() - t
        t = PC()
        d = (x_noisy - denoised) / t_hat
        x = x_noisy + step_scale * (sigma_t - t_hat) * d
        acc["update"] += PC() - t
        t = PC()
        trunk_tap_host(f"edm_x[step{k}]", x)
        acc["hooks"] += PC() - t
        t = PC()
        if dump_fn is not None:
            for _m in range(M):
                dump_fn(k, x[_m:_m + 1].detach().cpu())
        acc["dump"] += PC() - t
        acc["steps"] += 1
        acc["step_walls"].append(PC() - t_step0)
    acc["loop_wall"] += PC() - t_loop0
    acc["call_wall"] += PC() - t_call0
    return x


# --------------------------------------------------------------------------------------------
class _StubDenoise:
    """Deterministic host-only stand-in for DiffusionModule.denoise (selftest only)."""

    def denoise(self, x_noisy, t_hat, cond):
        import torch
        return torch.tanh(x_noisy * 0.5) * float(t_hat.item()) * 0.001 + x_noisy * 0.25


def selftest():
    import torch
    from tt_bio.protenix import edm_sample
    ok = True
    for label, kw in [("M=1", dict(multiplicity=1)),
                      ("M=4", dict(multiplicity=4)),
                      ("M=4,chunk2", dict(multiplicity=4, max_parallel_samples=2)),
                      ("M=3,member_seeds", dict(multiplicity=3, member_seeds=[1, 2, 3]))]:
        a = edm_sample(_StubDenoise(), None, 64, n_step=25, seed=0, **kw)
        b = edm_sample_timed(_StubDenoise(), None, 64, n_step=25, seed=0, **kw)
        same = torch.equal(a, b)
        ok &= same
        print(f"selftest {label:18s} torch.equal={same}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", choices=["protenix-v2", "opendde"])
    ap.add_argument("--target", type=Path)
    ap.add_argument("--msa-a3m", type=Path)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())

    import ttnn
    spec = importlib.util.spec_from_file_location(
        "tt_baseline", REPO / "scripts" / "gpu_vs_tt" / "tt_baseline.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)

    import tt_bio.protenix as P
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    accs = []
    stage = []

    def edm(*a, **kw):
        acc = _new_acc()
        ttnn.synchronize_device(dev)
        t0 = PC()
        out = edm_sample_timed(*a, _acc=acc, **kw)
        ttnn.synchronize_device(dev)
        acc["stage_wall_synced"] = PC() - t0
        accs.append(acc)
        stage.append(acc["stage_wall_synced"])
        return out

    orig = P.edm_sample
    P.edm_sample = edm
    msa_dir = Path("~/.cache/tt-bio-gpu-vs-tt/msa").expanduser()
    one_fold, meta, state = tb.build_fold(args.model, msa_dir, args.target, args.msa_a3m)
    cold_s, _ = one_fold()
    warm_s, _ = one_fold()
    P.edm_sample = orig

    def summarize(acc, tag):
        n = acc["steps"]
        sw = sorted(acc["denoise_walls"])
        host = sum(acc[p] for p in PHASES if p != "denoise")
        row = {"tag": tag, "steps": n,
               "stage_wall_s": acc["stage_wall_synced"],
               "call_wall_s": acc["call_wall"], "loop_wall_s": acc["loop_wall"],
               "host_total_s": host,
               "denoise_total_s": acc["denoise"],
               "denoise_step0_s": acc["denoise_walls"][0] if sw else 0.0,
               "denoise_median_s": sw[len(sw) // 2] if sw else 0.0,
               "denoise_min_s": sw[0] if sw else 0.0,
               "denoise_steady_sum_s": sum(acc["denoise_walls"][1:]),
               "phases_s": {p: acc[p] for p in PHASES},
               "residual_s": acc["stage_wall_synced"] - acc["call_wall"],
               "unattributed_in_loop_s": acc["loop_wall"] - sum(acc[p] for p in PHASES if p != "init"),
               }
        return row

    rows = [summarize(a, f"fold{i}") for i, a in enumerate(accs)]
    out = {"model": args.model, "host": "qb1", "card": 3,
           "cold_fold_s": cold_s, "warm_fold_s": warm_s,
           "ttnn": tb._ttnn_version(), "rows": rows}
    for r in rows:
        n = r["steps"]
        print(f"\n[{r['tag']}] stage {r['stage_wall_s']:.3f} s over {n} steps")
        print(f"  denoise total {r['denoise_total_s']:.3f} s "
              f"(step0 {r['denoise_step0_s']*1e3:.1f} ms, median {r['denoise_median_s']*1e3:.3f} ms)")
        print(f"  HOST total    {r['host_total_s']:.4f} s = "
              f"{100*r['host_total_s']/r['stage_wall_s']:.2f}% of the diffusion stage")
        for p in PHASES:
            v = r["phases_s"][p]
            if v > 0:
                print(f"    {p:9s} {v*1e3:9.2f} ms  {v/n*1e6:8.1f} us/step  "
                      f"{100*v/r['stage_wall_s']:5.2f}% of stage")
        print(f"  unattributed in loop {r['unattributed_in_loop_s']*1e3:.2f} ms, "
              f"outside call {r['residual_s']*1e3:.2f} ms")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
    state.reset()


if __name__ == "__main__":
    main()
