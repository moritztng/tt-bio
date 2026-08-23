#!/usr/bin/env python3
"""Boltz-2's 64-block trunk pairformer against its fp32 torch golden, at both fused-SDPA configs.

The question is whether the fused triangle-attention kernel's op default (HiFi2, math_approx on,
no fp32_dest_acc) costs the TRUNK anything, or whether 64 residual+LayerNorm blocks absorb it.
perf/fused_sdpa/errstruct.py answers it per call in fp64 -- HiFi4 / approx off / fp32_dest_acc is
1.98x lower total error on 18/18 real captured calls -- and a per-call win that the trunk absorbs
is not a reason to change a shipped default. So this scores the whole 64-block stack.

Three legs per input, in order: `base` (op default), `hifi` (the lever), `control` (op default
again). The device path is deterministic at fixed input, so `control` must equal `base` BIT for
BIT; that is the harness's proof it is pairing arms correctly, not a noise estimate. Any margin
between `base` and `hifi` is therefore real, which makes the acceptance bar a materiality bar
(>= 10% relative) rather than a significance one.

Two inputs, because real-vs-synthetic decided a 73x-vs-1.55x reading once already in this lineage:

    synthetic   the scales tests/test_tenstorrent.py::test_pairformer uses, seeded
    real        the (s, z) a real fold's trunk is handed, captured by perf/fused_sdpa/b2_capture.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
torch.set_grad_enabled(False)


def metrics(got, ref):
    g, r = got.double(), ref.double()
    e = g - r
    return {
        "median_rel": float((e.abs() / r.abs().clamp_min(1e-30)).median()),
        "rel_rms": float(e.pow(2).sum().sqrt() / r.pow(2).sum().sqrt()),
        "max_abs": float(e.abs().max()),
    }


def set_hifi(tt, on: bool) -> int:
    n = 0
    for b in tt.module.blocks:
        for a in (b.triangle_attention_start, b.triangle_attention_end):
            a.sdpa_hifi = on
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", default="320,512")
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--capture", default="/home/ttuser/b2cap_trunk_{n}.pt",
                    help="real captured trunk input, {n} substituted with the sequence length")
    ap.add_argument("--seed", type=int, default=893)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import PairformerModule, WeightScope
    from tt_bio.reference import PairformerModule as PairformerModuleTorch
    assert Path(T.__file__).resolve().is_relative_to(REPO), f"wrong tree: {T.__file__}"

    cache = Path(os.environ.get("BOLTZ_CACHE", "~/.boltz")).expanduser()
    state = torch.load(cache / "boltz2_conf.ckpt", map_location="cpu", mmap=True,
                       weights_only=False)["state_dict"]

    # The construction-site mechanism, checked on its own: no weights, no device, just that the
    # env override reaches the kwarg. The legs below flip the attribute directly so all three share
    # one set of loaded weights, which is what makes the A/A bit-exact rather than approximate.
    probe = {}
    for env, want in (("", False), ("boltz2.trunk", True), ("-boltz2.trunk", True),
                      ("all", True)):
        os.environ["TT_BIO_TRIATT_SDPA_HIFI_AB"] = env
        got = T.triatt_sdpa_hifi_site("boltz2.trunk")
        probe[env or "(unset)"] = got
    os.environ["TT_BIO_TRIATT_SDPA_HIFI_AB"] = ""
    assert probe["(unset)"] is False and probe["boltz2.trunk"] is True \
        and probe["-boltz2.trunk"] is False and probe["all"] is True, probe
    print(f"site-flag probe ok: {probe}", flush=True)

    report = {"blocks": args.blocks, "seed": args.seed, "cases": []}

    for n in [int(x) for x in args.seq_lens.split(",") if x.strip()]:
        for kind in ("synthetic", "real"):
            if kind == "synthetic":
                torch.manual_seed(args.seed)
                s = 8 * torch.randn(1, n, 384)
                z = 26 * torch.randn(1, n, n, 128)
            else:
                p = Path(args.capture.format(n=n))
                if not p.exists():
                    print(f"skip real n={n}: no capture at {p}", flush=True)
                    continue
                blob = torch.load(p, weights_only=False)
                s, z = blob["s"].float(), blob["z"].float()
                assert tuple(s.shape) == (1, n, 384), s.shape
            mask, pair_mask = torch.ones(1, n), torch.ones(1, n, n)

            tt = PairformerModule(args.blocks, 32, 4, 24, 16, transform_s=True)
            ref = PairformerModuleTorch(384, 128, args.blocks, v2=True).eval()
            sd = WeightScope.wrap(state).child("pairformer_module").as_dict()
            tt.load_state_dict(sd, strict=False)
            ref.load_state_dict(sd, strict=False)

            print(f"\n=== n={n} {kind}: golden ===", flush=True)
            s_ref, z_ref = ref(s.clone(), z.clone(), mask, pair_mask)
            del ref

            legs, outs = {}, {}
            for leg, on in (("base", False), ("hifi", True), ("control", False)):
                if tt.module is not None:
                    flipped = set_hifi(tt, on)
                else:
                    flipped = None
                before = list(T.SDPA_HIFI_CALLS)
                s_tt, z_tt = tt(s.clone(), z.clone(), mask, pair_mask)
                if flipped is None:
                    flipped = set_hifi(tt, on)      # created on the first forward
                hifi_calls = T.SDPA_HIFI_CALLS[0] - before[0]
                legs[leg] = {"hifi_calls": hifi_calls, "sites_flipped": flipped,
                             "s": metrics(s_tt, s_ref), "z": metrics(z_tt, z_ref)}
                outs[leg] = (s_tt.clone(), z_tt.clone())
                m = legs[leg]
                print(f"  {leg:8s} hifi_calls={hifi_calls:4d}  "
                      f"s med {m['s']['median_rel']:.5e} rms {m['s']['rel_rms']:.5e}  "
                      f"z med {m['z']['median_rel']:.5e} rms {m['z']['rel_rms']:.5e}", flush=True)

            aa_s = float((outs["base"][0] - outs["control"][0]).abs().max())
            aa_z = float((outs["base"][1] - outs["control"][1]).abs().max())
            ab_z = float((outs["base"][1] - outs["hifi"][1]).abs().max())
            case = {"n": n, "input": kind, "legs": legs,
                    "aa_max_abs": {"s": aa_s, "z": aa_z}, "base_vs_hifi_max_abs_z": ab_z}
            for t in ("s", "z"):
                b, h = legs["base"][t]["median_rel"], legs["hifi"][t]["median_rel"]
                case[f"delta_{t}_median_rel"] = (h - b) / b
                b, h = legs["base"][t]["rel_rms"], legs["hifi"][t]["rel_rms"]
                case[f"delta_{t}_rel_rms"] = (h - b) / b
            print(f"  A/A max|base-control|  s {aa_s:.3e}  z {aa_z:.3e}   "
                  f"base-vs-hifi max|dz| {ab_z:.3e}", flush=True)
            print(f"  delta z median_rel {case['delta_z_median_rel']:+.2%}  "
                  f"z rel_rms {case['delta_z_rel_rms']:+.2%}  "
                  f"s median_rel {case['delta_s_median_rel']:+.2%}  "
                  f"s rel_rms {case['delta_s_rel_rms']:+.2%}", flush=True)
            report["cases"].append(case)
            del tt, outs, s_ref, z_ref

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print("\nwrote " + args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
