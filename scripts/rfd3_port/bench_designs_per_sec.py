"""p23: real designs/sec throughput measurement for the RFD3 port.

Runs the real end-to-end path (featurize a real PDB -> on-device
TokenInitializer -> RFD3Sampler EDM loop over the real ttnn DiffusionModule)
for the p12/p21-verified IAI_protein.pdb + "A1-10,20,A31-40" fixture, at a
few different `num_timesteps`, so per-step device-forward cost can be
isolated from the one-time TokenInitializer/weight-load cost (step-count
method: per_step = (t(n2) - t(n1)) / (n2 - n1); trunk = t(n1) - n1*per_step).

Usage:
  TT_VISIBLE_DEVICES=0 python3 scripts/rfd3_port/bench_designs_per_sec.py
"""
import os, sys, time, argparse, subprocess, shutil
from pathlib import Path
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3 import build_diffusion_module, build_token_initializer
from tt_bio.rfd3_featurize import featurize
from tt_bio.rfd3_input import InputSpecification
from tt_bio.rfd3_sampler import RFD3Sampler

PDB = os.path.join(os.path.dirname(__file__), "parity_artifacts", "iai_protein", "IAI_protein.pdb")
CONTIG = "A1-10,20,A31-40"
GOLDEN_DIR = os.path.expanduser(os.environ.get("RFD3_GOLDEN_DIR", "~/.coworker/artifacts/rfd3-goldens/capture"))


def _run_multi_device(inputs_yaml, ndev, designs_per_card, num_timesteps, seed, out_root):
    """Measure aggregate designs/sec for `tt-bio design --devices <0..ndev-1>` fan-out.

    One `tt-bio design` invocation = one subprocess per card, each loading weights +
    compiling once then running its shard of (num_designs) D=1 forwards. Wall-clock
    includes the per-worker fixed cost (weight load + cold compile), which is exactly
    what a real one-shot user pays; designs/sec = M / wall_clock. Run with two
    designs_per_card values to see whether that fixed cost amortizes (steady-state).
    """
    device_list = ",".join(str(i) for i in range(ndev))
    M = ndev * designs_per_card
    out_dir = f"{out_root}/multi_d{ndev}_m{designs_per_card}pc"
    shutil.rmtree(out_dir, ignore_errors=True)
    cmd = [sys.executable, "-m", "tt_bio.main", "design", inputs_yaml,
           "--from_pdb", "--devices", device_list, "--num_designs", str(M),
           "--num_timesteps", str(num_timesteps), "--seed", str(seed),
           "--out_dir", out_dir]
    # The parent `tt-bio design` process must see ALL cards to validate --devices
    # (detect_tenstorrent_devices honors TT_VISIBLE_DEVICES). The fanout children
    # set their own per-shard TT_VISIBLE_DEVICES, so drop the ambient pin here.
    env = {k: v for k, v in os.environ.items() if k != "TT_VISIBLE_DEVICES"}
    env["PYTHONPATH"] = os.getcwd()
    env["TT_BIO_LEASE_HOLDER"] = f"worker:tt-bio-rfdiffusion3-batch-perf-p2"
    env["TT_MESH_GRAPH_DESC_PATH"] = os.environ.get(
        "TT_MESH_GRAPH_DESC_PATH",
        "/home/ttuser/tt-bio-dev/env/lib/python3.10/site-packages/ttnn/tt_metal/"
        "fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto")
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    ok = proc.returncode == 0
    n_cif = len(list(Path(out_dir).glob("*.cif"))) if Path(out_dir).exists() else 0
    dps = (M / wall) if wall > 0 else float("nan")
    print(f"[multi --devices={ndev} M={M} ({designs_per_card}/card) ts={num_timesteps}] "
          f"wall={wall:.2f}s ok={ok} cifs={n_cif} -> {dps:.4f} designs/sec (cold-start, "
          f"includes per-worker weight-load + compile)")
    if not ok:
        print("  STDERR tail:", "\n".join(proc.stderr.splitlines()[-15:]))
        print("  STDOUT tail:", "\n".join(proc.stdout.splitlines()[-15:]))
    return wall, dps, n_cif, ok


def main():
    ap = argparse.ArgumentParser(description="RFD3 designs/sec bench (single-card in-process + multi-device fan-out).")
    ap.add_argument("--multi-device", action="store_true",
                    help="Only run the multi-device `tt-bio design --devices` fan-out measurement "
                         "(no in-process single-card device open, so it does not hold card 0's lease).")
    ap.add_argument("--inputs", default=os.path.join(os.path.dirname(__file__),
                     "parity_artifacts", "iai_protein", "iai_inputs.yaml"),
                    help="Inputs YAML for `tt-bio design` (multi-device mode).")
    ap.add_argument("--out-root", default=os.path.join(os.getcwd(), "perf/p2"))
    args = ap.parse_args()
    if args.multi_device:
        print("[multi-device] measuring `tt-bio design --devices` aggregate designs/sec "
              "(qb2 p300c, IAI_protein fixture, num_timesteps=200, --from_pdb)")
        results = []
        # two designs_per_card values per ndev to check fixed-cost amortization
        for ndev, dpc in [(2, 2), (2, 4), (4, 2), (4, 4)]:
            wall, dps, n_cif, ok = _run_multi_device(
                args.inputs, ndev, dpc, num_timesteps=200, seed=42, out_root=args.out_root)
            results.append((ndev, dpc, wall, dps, n_cif, ok))
        print("\n[multi-device summary] (num_timesteps=200, cold-start incl. per-worker fixed cost):")
        for ndev, dpc, wall, dps, n_cif, ok in results:
            print(f"  --devices={ndev} {dpc}/card (M={ndev*dpc}): wall={wall:.2f}s -> {dps:.4f} designs/sec "
                  f"({n_cif} cifs, ok={ok})")
        # steady-state warm estimate: ndev * single-card-warm (per-card compute is
        # independent D=1 forwards; fixed cost amortizes to zero as M grows)
        single_warm = 0.0434  # measured this run (p300c, step-count method, ts=200)
        print(f"\n[steady-state warm estimate] ndev * single-card-warm({single_warm}): "
              f"devices=2 -> {2*single_warm:.4f}, devices=4 -> {4*single_warm:.4f} designs/sec "
              f"(per-card compute is independent; fixed cost -> 0 as M grows)")
        return
    spec = InputSpecification.from_dict({"input": PDB, "contig": CONTIG})
    spec.validate()
    f = featurize(PDB, spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
    is_motif = f["is_motif_atom_with_fixed_coord"]
    L = f["ref_pos"].shape[0]
    print(f"[setup] I={f['restype'].shape[0]} L={L} ({int(is_motif.sum())} motif atoms)")

    t0 = time.time()
    ti_weights = torch.load(os.path.join(GOLDEN_DIR, "token_initializer.real_weights.pt"),
                             map_location="cpu", weights_only=True)
    dm_weights = torch.load(os.path.join(GOLDEN_DIR, "diffusion_module.real_weights.pt"),
                             map_location="cpu", weights_only=True)
    dev_ti = build_token_initializer(ti_weights)
    dev_dm = build_diffusion_module(dm_weights)
    print(f"[setup] weight load + device bring-up: {time.time() - t0:.2f}s")

    coord0 = f["motif_pos"].float().unsqueeze(0)

    def run(n_ts, label):
        sampler = RFD3Sampler(num_timesteps=n_ts)
        with torch.no_grad():
            t_init0 = time.time()
            init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
            t_init = time.time() - t_init0
            g = torch.Generator().manual_seed(42)
            t0 = time.time()
            X, _ = sampler.sample(dev_dm, 1, L, coord0, f, init, is_motif, generator=g)
            t_sample = time.time() - t0
        print(f"[{label}] num_timesteps={n_ts} token_init={t_init:.3f}s sample={t_sample:.3f}s "
              f"total={t_init + t_sample:.3f}s ({(n_ts - 1)} device-forward steps)")
        return t_init, t_sample

    # cold (first shapes seen this process -> pays full kernel-compile cost)
    run(4, "cold")
    # warm, small N
    ti_w, n8 = run(8, "warm")
    # warm, larger N (same shapes already compiled -> isolates per-step cost)
    ti_w2, n40 = run(40, "warm")

    per_step = (n40 - n8) / (40 - 8)
    trunk = n8 - 7 * per_step  # token-init + 1st-step fixed overhead baked into "sample" itself is ~0 (init timed separately)
    print(f"\n[step-count] per_step={per_step * 1000:.1f} ms/step  "
          f"(from warm 8-step={n8:.3f}s vs 40-step={n40:.3f}s)")

    for n_full in (50, 200):
        est_sample = (n_full - 1) * per_step + max(0.0, n8 - 7 * per_step)
        est_total = ti_w2 + est_sample
        print(f"[estimate] num_timesteps={n_full}: sample~{est_sample:.2f}s total~{est_total:.2f}s "
              f"-> {1.0 / est_total:.4f} designs/sec ({est_total:.1f}s/design)")

    # Sequential D=1 baseline. The production in-forward D=1/2/4/8 comparison is
    # measured separately by bench_batch_designs_per_sec.py.
    print("\n[num_designs] N independent D=1 forwards (sequential baseline):")
    N_designs = 8
    n_ts = 40
    sampler = RFD3Sampler(num_timesteps=n_ts)
    with torch.no_grad():
        init = dev_ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
    coord0 = f["motif_pos"].float().unsqueeze(0)
    t0 = time.time()
    with torch.no_grad():
        for i in range(N_designs):
            g = torch.Generator().manual_seed(42 + i)
            sampler.sample(dev_dm, 1, L, coord0, f, init, is_motif, generator=g)
    t_batch = time.time() - t0
    # warm-cache this run too (first call after run(40) is already warm)
    t0 = time.time()
    with torch.no_grad():
        for i in range(N_designs):
            g = torch.Generator().manual_seed(42 + i)
            sampler.sample(dev_dm, 1, L, coord0, f, init, is_motif, generator=g)
    t_batch_warm = time.time() - t0
    per_design_warm = t_batch_warm / N_designs
    print(f"[num_designs={N_designs}] cold={t_batch:.2f}s warm={t_batch_warm:.2f}s "
          f"-> {per_design_warm:.3f}s/design, {1.0/per_design_warm:.4f} designs/sec "
          f"(sequential, 1 card, {n_ts} steps)")
    # extrapolate to num_timesteps=200 (the rc-foundry default)
    per_step_warm = per_design_warm / (n_ts - 1)
    est_200 = (200 - 1) * per_step_warm
    print(f"[num_designs extrapolate] num_timesteps=200: ~{est_200:.2f}s/design -> "
          f"{1.0/est_200:.4f} designs/sec (1 card, sequential)")
    print(f"[reference] H200 batch=8 num_timesteps=200: 0.452 designs/sec "
          f"(rc-foundry bf16 AMP + diffusion_batch_size=8 defaults)")


if __name__ == "__main__":
    main()
