#!/usr/bin/env python3
"""Time the RF3 ttnn port per phase, on the inputs the H200 reference used.

The bar this feeds is `perf/rf3/gpu_reference.json`: `tt_device_s <= 4 x h200_device_s`
at matched length AND matched diffusion batch. Two rules carried over from that harness:

  - featurisation is common cost. The H200 reference excluded it after two rented boxes
    disagreed by 2.3x on it, so it is reported here and left out of the compared number.
  - checkpoint load is outside every reported number, same as the reference.

What is NOT excluded is the port's own host-in-the-loop work, the chirality gradients and
the EDM arithmetic between denoiser calls. That is the port's cost, not shared prep, so
it sits inside `infer_s`.

Phase boundaries carry an explicit device sync, because ttnn dispatch is asynchronous and
an unsynced boundary bills the previous phase's tail to the next one. `--breakdown` adds
syncs *inside* the denoiser, which over-syncs and inflates the total; use it for
attribution only and read the headline off a run without it
(`tt-bio-isolated-op-timing-oversync-inflates-cost`).
"""
from __future__ import annotations

import argparse
import contextlib
import enum
import json
import os
import statistics
import sys
import time
from pathlib import Path

# The upstream RF3 reference targets 3.11+ and uses enum.StrEnum; the tt-bio env is 3.10.
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

LADDER = (128, 256, 512, 768, 1024)


def net_config(ckpt_path: str) -> dict:
    """Block counts off the checkpoint, without instantiating the torch reference."""
    from omegaconf import OmegaConf
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.to_container(ck["train_cfg"].model.net, resolve=True)
    cfg.pop("_target_", None)
    del ck
    return cfg


class Timer:
    """Wall-clock accumulator with a device sync at every boundary."""

    def __init__(self, device):
        import ttnn
        self._ttnn, self._device = ttnn, device
        self.t: dict[str, float] = {}
        self.n: dict[str, int] = {}

    def sync(self):
        self._ttnn.synchronize_device(self._device)

    @contextlib.contextmanager
    def span(self, name):
        self.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0
            self.n[name] = self.n.get(name, 0) + 1

    def wrap(self, obj, attr, name):
        """Time every call to `obj.attr` under `name`. Used by --breakdown."""
        fn = getattr(obj, attr)

        def timed(*a, **kw):
            with self.span(name):
                return fn(*a, **kw)
        setattr(obj, attr, timed)


def instrument_denoiser(tt, host, tm: Timer):
    """Split one denoiser call into conditioning / encoder / DiT / decoder / host."""
    dm = tt.diffusion_module
    tm.wrap(dm.conditioning, "pair", "dn.cond_pair")
    tm.wrap(dm.conditioning, "single", "dn.cond_single")
    tm.wrap(dm, "encoder", "dn.encoder")
    tm.wrap(dm, "transformer", "dn.dit")
    tm.wrap(dm, "decoder", "dn.decoder")
    tm.wrap(dm, "process_s", "dn.process_s")
    tm.wrap(host, "step_inputs", "dn.step_inputs_host")


def one_fold(tt, f, rep_atom_idxs, device, tm: Timer, *, n_recycles: int,
             diffusion_batch_size: int, want_confidence: bool, breakdown: bool):
    """`RF3.predict` unrolled so each phase can be timed on its own."""
    import ttnn
    from tt_bio.rf3.host import HostInputs
    from tt_bio.rf3.sampler import Draws

    with tm.span("upload"):
        host = HostInputs.build(f, device)
    if breakdown:
        instrument_denoiser(tt, host, tm)

    with tm.span("feature_init"):
        s_inputs, s_init, z_init = tt.feature_initializer(
            host.single_in, host.pair_in, host.pair_v, host.keys_indexing,
            host.atom_to_token_mean, host.window_mask, host.n_atom_padded,
            host.token_feats, host.relpos_feat, host.bond_feat)
    with tm.span("recycles"):
        s = ttnn.mul(s_init, 0.0)
        z = ttnn.mul(z_init, 0.0)
        template_channels = tt.recycler.template_embedder.embed_template_feats(
            host.template_feats)
        for i in range(n_recycles):
            s, z = tt.recycler(host, template_channels,
                               host.msa_stack[i % len(host.msa_stack)],
                               s_inputs, s_init, z_init, s, z)
    with tm.span("distogram"):
        distogram = torch.Tensor(ttnn.to_torch(tt.distogram_head(z))).float()

    calls = [0]

    def denoise(x_noisy, t):
        calls[0] += 1
        return tt.diffusion_module(host, x_noisy, t, s_inputs, s, z)

    coord = torch.zeros(diffusion_batch_size, host.n_atom, 3)
    with tm.span("diffusion"):
        x_pred, _ = tt.sampler.sample(denoise, coord, diffusion_batch_size,
                                      draws=Draws())

    out = {"coords": x_pred, "denoiser_calls": calls[0], "n_atom": host.n_atom,
           "n_atom_padded": host.n_atom_padded, "n_token": host.n_token}
    if want_confidence and rep_atom_idxs is not None:
        with tm.span("confidence"):
            conf = tt.confidence(s_inputs, s, z, x_pred, rep_atom_idxs)
            out["plddt_logit_mean"] = float(conf["plddt_logits"].mean())
    return out


PHASES = ("upload", "feature_init", "recycles", "distogram", "diffusion", "confidence")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=128)
    ap.add_argument("--input", default=None, help="override the ladder input path")
    ap.add_argument("--ckpt", default="/home/ttuser/rf3_perf_work/rf3_latest.ckpt")
    ap.add_argument("--n_recycles", type=int, default=10)
    ap.add_argument("--num_steps", type=int, default=50)
    ap.add_argument("--diffusion_batch_size", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2,
                    help="rep 0 is discarded as cold, as the H200 harness does")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-confidence", action="store_true")
    ap.add_argument("--breakdown", action="store_true",
                    help="sync inside the denoiser for attribution; inflates the total")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    inp = args.input or str(REPO / f"perf/rf3/inputs/rf3_{args.aa}.json")

    from tt_bio.rf3.featurize import featurize
    t0 = time.perf_counter()
    fo = featurize(inp, n_recycles=args.n_recycles,
                   diffusion_batch_size=args.diffusion_batch_size, seed=args.seed)[0]
    featurize_s = time.perf_counter() - t0
    f = fo["feats"]
    rep_atom_idxs = fo.get("ground_truth", {}).get("rep_atom_idxs")

    import ttnn
    from tt_bio.rf3 import model as rf3_model
    from tt_bio.tenstorrent import get_device

    cfg = net_config(args.ckpt)
    device = get_device()
    kcfg = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    t0 = time.perf_counter()
    tt = rf3_model.load(
        args.ckpt, kcfg,
        n_pairformer_blocks=cfg["recycler"]["n_pairformer_blocks"],
        n_msa_blocks=cfg["recycler"]["msa_module"]["n_block"],
        n_dit_blocks=cfg["diffusion_module"]["diffusion_transformer"]["n_block"],
        num_timesteps=args.num_steps,
        with_confidence=(not args.no_confidence) and "confidence_head" in cfg)
    load_s = time.perf_counter() - t0

    reps = []
    for rep in range(args.reps):
        tm = Timer(device)
        t0 = time.perf_counter()
        out = one_fold(tt, f, rep_atom_idxs, device, tm,
                       n_recycles=args.n_recycles,
                       diffusion_batch_size=args.diffusion_batch_size,
                       want_confidence=not args.no_confidence,
                       breakdown=args.breakdown)
        wall = time.perf_counter() - t0
        rec = dict(tm.t)
        rec["_counts"] = dict(tm.n)
        rec["infer_s"] = wall
        rec["cold"] = rep == 0
        rec["finite"] = bool(torch.isfinite(out["coords"]).all())
        rec["coord_rms"] = float(out["coords"].pow(2).mean().sqrt())
        rec["denoiser_calls"] = out["denoiser_calls"]
        if "plddt_logit_mean" in out:
            rec["plddt_logit_mean"] = out["plddt_logit_mean"]
        reps.append(rec)
        print(f"[rep {rep}{' cold' if rep == 0 else ''}] "
              + "  ".join(f"{k}={rec[k]:.3f}" for k in PHASES if k in rec)
              + f"  infer={rec['infer_s']:.3f}  finite={rec['finite']}", flush=True)
        if args.breakdown:
            for k in sorted(k for k in rec if k.startswith("dn.")):
                print(f"    {k:24s} {rec[k]:8.3f} s over {rec['_counts'][k]:4d} calls"
                      f"  ({rec[k] / max(rec['_counts'][k], 1) * 1e3:7.2f} ms/call)",
                      flush=True)

    warm = [r for r in reps if not r["cold"]] or reps
    keys = [k for k in warm[0] if isinstance(warm[0][k], float)]
    med = {k: statistics.median([r[k] for r in warm]) for k in keys}

    rungs = json.loads((REPO / "perf/rf3/gpu_reference.json").read_text())["rungs"]
    match = [r for r in rungs if r["rung_aa"] == args.aa
             and r["batch"] == args.diffusion_batch_size]
    target = match[0] if match else None

    report = {
        "tag": args.tag, "aa": args.aa, "input": inp,
        "n_atom": out["n_atom"], "n_atom_padded": out["n_atom_padded"],
        "n_token": out["n_token"],
        "n_recycles": args.n_recycles, "num_steps": args.num_steps,
        "denoiser_calls": out["denoiser_calls"],
        "diffusion_batch_size": args.diffusion_batch_size,
        "math_fidelity": "HiFi4", "fp32_dest_acc_en": True,
        "breakdown_syncs": args.breakdown,
        "featurize_s": featurize_s, "ckpt_load_s": load_s,
        "reps": reps, "median_warm": med,
        "git_head": os.popen(f"git -C {REPO} rev-parse --short HEAD").read().strip(),
    }
    if target:
        report["h200_device_s"] = target["h200_device_s"]
        report["tt_target_device_s"] = target["tt_target_device_s"]
        report["ratio_vs_target"] = med["infer_s"] / target["tt_target_device_s"]
        report["gap_vs_h200"] = med["infer_s"] / target["h200_device_s"]
        print(f"\n{args.aa} aa b{args.diffusion_batch_size}: "
              f"tt {med['infer_s']:.2f} s  target {target['tt_target_device_s']:.2f} s  "
              f"=> {report['ratio_vs_target']:.3f}x of target, "
              f"{report['gap_vs_h200']:.2f}x H200 "
              f"({'PASS' if report['ratio_vs_target'] <= 1 else 'OVER'})")
    print(json.dumps({k: v for k, v in report.items() if k != "reps"}, indent=2,
                     default=str))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
