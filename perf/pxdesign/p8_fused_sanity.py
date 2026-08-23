"""Is the fused-HiFi trunk output finite, and how far is it from the incumbent's?

NOT the accuracy verdict -- that is `tap_gate.py --device` plus `filter_flip_rate.py` over the
50-design population and it belongs to `pxdesign-af2ig-port-p18`. This is the floor underneath the
2.844x: an arm that returns NaN, or that has drifted by an order of magnitude, is not a speedup at
all, and RF3 shipped exactly that mistake once (all-fp32 CBs, pcc 0.893, `triatt_sdpa.py:93-99`).

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:pxdesign-perf-p8 \
        PYTHONPATH=. python3 perf/pxdesign/p8_fused_sanity.py --tokens 848
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "af2_port"))

NUM_MSA_ROWS = 2


def run(model, tokens: int, fused: bool):
    from tt_bio import tenstorrent as TT
    from tt_bio.af2_reference import C_EXTRA, C_M, C_Z

    TT._TRIATT_FUSED_HIFI = fused
    for key in TT.TRIATT_FUSED_HIFI_STATS:
        TT.TRIATT_FUSED_HIFI_STATS[key] = 0

    g = torch.Generator().manual_seed(0)
    d = model.trunk_dtype
    msa = torch.randn(NUM_MSA_ROWS, tokens, C_M, generator=g).to(d)
    pair = torch.randn(tokens, tokens, C_Z, generator=g).to(d)
    extra = torch.randn(1, tokens, C_EXTRA, generator=g).to(d)
    extra_mask = torch.zeros(1, tokens, dtype=d)
    msa_mask = torch.ones(NUM_MSA_ROWS, tokens, dtype=d)
    pair_mask = torch.ones(tokens, tokens, dtype=d)

    z = model.extra_msa_stack(extra, pair, extra_mask, pair_mask)
    m, z = model.evoformer_stack(msa, z, msa_mask, pair_mask)
    return (m.float(), z.float(), dict(TT.TRIATT_FUSED_HIFI_STATS))


def stats(name, t):
    return {"tensor": name, "finite": bool(torch.isfinite(t).all()),
            "nan": int(torch.isnan(t).sum()), "inf": int(torch.isinf(t).sum()),
            "absmax": float(t.abs().max()), "rms": float(t.pow(2).mean().sqrt())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=848)
    ap.add_argument("--params", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--aa", action="store_true",
                    help="run the INCUMBENT twice: the numeric A/A control, without which "
                         "the fused arm's delta cannot be told from this arm's own "
                         "run-to-run spread")
    args = ap.parse_args()

    from tt_bio.af2 import load_af2_device_model
    from tt_bio.af2_weights import load_af2_state_dict
    from tap_gate import DEFAULT_PARAMS

    model = load_af2_device_model(load_af2_state_dict(args.params or DEFAULT_PARAMS),
                                  template=False)
    model.eval()
    model.set_rne_residual(True)

    with torch.no_grad():
        m_i, z_i, st_i = run(model, args.tokens, False)
        m_f, z_f, st_f = run(model, args.tokens, not args.aa)

    out = {"mode": "af2ig_fused_hifi_sanity",
           "arm_b": "incumbent (A/A)" if args.aa else "fused", "tokens": args.tokens,
           "incumbent_stats": st_i, "fused_stats": st_f,
           "incumbent": [stats("m", m_i), stats("z", z_i)],
           "fused": [stats("m", m_f), stats("z", z_f)]}
    for name, a, b in (("m", m_i, m_f), ("z", z_i, z_f)):
        out[f"{name}_rel_rms"] = float((a - b).pow(2).mean().sqrt() / a.pow(2).mean().sqrt())
        out[f"{name}_max_abs_delta"] = float((a - b).abs().max())
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
