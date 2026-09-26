#!/usr/bin/env python3
"""Is `AF2DeviceModel.template_cached` correct on a multimer_v3 fold with recycles?

`_template_key` hashes every `template_*` feature and deliberately omits `pair`, on the ground
that `AF2Template.forward` is constant in `pair`. That argument is written for the monomer:
`TemplateEmbedding.forward` attends the pair as the query over `num_templ` keys, and with one
template the softmax weight is exactly 1.0, so the query drops out. `af2_reference.py:1106`
picks `TemplateEmbeddingMultimer` for multimer instead, and that class has no pointwise
attention at all -- its `_features` ends with `(self.query_norm(pair), self.pair_embedding[8])`,
so `pair` is summed into the template act and run through the whole triangle-attention pair
stack. Nothing collapses and nothing drops out.

Because `pair = pair + template_embedding(pair, ...)` fires once per recycling pass and `pair`
is different in each, a cache keyed without `pair` serves recycle 1's answer to recycles 2..4.

The probe runs ONE fold with the cache ON and, on every call the cache serves, recomputes the
embedding from that pass's own `pair` and reports the distance between the two. One run, both
arms, no cross-run alignment to get wrong: the cached value and the correct value are produced
from the same tensors microseconds apart. The `off` arm runs the same fold with the cache off
so the trunk outputs of the two arms can be compared end to end.

    TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-p10-tmplemb \
        PYTHONPATH=. python3 perf/bcx_p10_tmplemb/cachecheck.py --out out/cache
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PDB = ROOT / "scripts/af2_port/parity_artifacts/designpop_bg119/binder_complex.pdb"
DEFAULT_PARAMS = "/home/ttuser/bcx_e2e/af2_params/params_model_1_multimer_v3.npz"


def to_torch(v):
    a = np.asarray(v)
    if a.dtype == np.bool_:
        return torch.from_numpy(a)
    if a.dtype.kind in "iu":
        return torch.from_numpy(a.astype(np.int64))
    return torch.from_numpy(a.astype(np.float32))


def distance(a: torch.Tensor, b: torch.Tensor) -> dict:
    """`a` against `b`, in float64 so the report is not itself an approximation."""
    x, y = a.detach().double().flatten(), b.detach().double().flatten()
    d = x - y
    ny = float(y.norm())
    return {"rel_l2": float(d.norm()) / max(ny, 1e-300),
            "max_abs": float(d.abs().max()),
            "cos": float(torch.dot(x, y)) / max(float(x.norm()) * ny, 1e-300),
            "norm_ref": ny}


def build_feats(pdb: str):
    from tt_bio import af2_data
    from tt_bio.af2_data import complex_features, initial_recycle_state, parse_pdb_chain

    chain = parse_pdb_chain(pdb, "B")
    keep = chain.mask[:, 0] == 1
    restypes = af2_data._rc.restypes
    binder_seq = "".join(restypes[a] if a < len(restypes) else "A"
                         for a in chain.aatype[keep])
    feats_np = complex_features(pdb, binder_seq, "A", "B")
    prev_np = initial_recycle_state(feats_np)
    return ({k: to_torch(v) for k, v in feats_np.items()},
            {k: to_torch(v) for k, v in prev_np.items()}, binder_seq)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", default=str(DEFAULT_PDB))
    ap.add_argument("--params", default=DEFAULT_PARAMS)
    ap.add_argument("--monomer", action="store_true",
                    help="the control: the variant the cache was written for")
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tt_bio.af2 import load_af2_device_model
    from tt_bio.af2_weights import load_af2_state_dict

    multimer = not args.monomer
    feats, prev0, binder_seq = build_feats(args.pdb)
    state = load_af2_state_dict(args.params, multimer=multimer)
    model = load_af2_device_model(state, template=True, multimer=multimer,
                                  structure=False, trunk_dtype=torch.bfloat16).eval()

    # Every call records what the cache did and, when it served, what the correct answer was.
    calls: list[dict] = []
    original = type(model).template_embedding

    def probed(self, pair, feats_, mask_2d, multichain_mask):
        key = self._template_key(feats_, mask_2d, multichain_mask)
        served = self._template_cache is not None and self._template_cache[0] == key
        t0 = time.perf_counter()
        out = original(self, pair, feats_, mask_2d, multichain_mask)
        row = {"call": len(calls), "served_from_cache": bool(served),
               "n": int(pair.shape[0]), "seconds": round(time.perf_counter() - t0, 4),
               "pair_norm": float(pair.detach().double().norm())}
        if served:
            saved, self._template_cache = self._template_cache, None
            try:
                fresh = original(self, pair, feats_, mask_2d, multichain_mask)
            finally:
                self._template_cache = saved
            row["cached_vs_correct"] = distance(out, fresh)
        calls.append(row)
        return out

    type(model).template_embedding = probed
    try:
        arms = {}
        for arm in ("on", "off"):
            model.template_cached = (arm == "on")
            model._template_cache = None
            calls.clear()
            prev = dict(prev0)
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(args.recycles + 1):
                    out = model(feats, prev)
                    prev = {"prev_msa_first_row": out["msa_first_row"],
                            "prev_pair": out["pair"],
                            "prev_pos": prev["prev_pos"]}
            arms[arm] = {"calls": list(calls),
                         "fold_seconds": round(time.perf_counter() - t0, 3),
                         "pair": out["pair"].detach().clone(),
                         "single": out["single"].detach().clone()}
            print(json.dumps({"arm": arm, "fold_s": arms[arm]["fold_seconds"],
                              "calls": arms[arm]["calls"]}, indent=1), flush=True)
    finally:
        type(model).template_embedding = original

    report = {
        "pdb": args.pdb, "params": args.params, "multimer": multimer,
        "binder_residues": len(binder_seq), "tokens": int(feats["seq_mask"].shape[0]),
        "recycles": args.recycles,
        "arms": {a: {k: v for k, v in d.items() if k not in ("pair", "single")}
                 for a, d in arms.items()},
        "trunk_on_vs_off": {
            "pair": distance(arms["on"]["pair"], arms["off"]["pair"]),
            "single": distance(arms["on"]["single"], arms["off"]["single"]),
        },
        "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "cachecheck.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
