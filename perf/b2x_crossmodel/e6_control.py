"""Does the E6 gated channel move still agree with the sequence it replaces?

`reblock_permute_gated` is `chunk` + two gated multiplies + the (0,3,1,2) channel move in one
kernel, and it is only ever legal because it is bit-exact against that sequence. This scores
exactly that, with no new flag involved: arm E6 (`TT_BIO_REBLOCK_PERMUTE_GATED` on, the shipped
default) against arm REF (the same call with `set_enabled_gated(False)`, which sends
`eligible_gated` to False and runs the unfused chunk+gate+move).

Any model that builds its trimuls with `gated_move=True` (opendde, esmfold2) and calls them
without a pair mask takes the E6 path on today's main, so a disagreement here is a live defect,
not a property of the flag under test.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ttnn
from tt_bio import tenstorrent as T
from tt_bio import reblock_permute as RB
import trimul_ab as H


def sha(t):
    return hashlib.sha256(t.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="protenix-v2", choices=sorted(H.LOADERS))
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--gated-move", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tri, ship_gm, label = H.LOADERS[a.model]()
    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    c_z = tri["start"]["norm_in.weight"].shape[0]
    torch.manual_seed(0)
    z = ttnn.from_torch(torch.randn(1, a.n, a.n, c_z), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16)
    print(f"# {a.model} {label} c_z={c_z} N={a.n} gated_move={bool(a.gated_move)} "
          f"(ships {ship_gm}) mask=None", flush=True)
    res = {}
    for which in ("start", "end"):
        tm = T.TriangleMultiplication(which == "end", tri[which], ckc,
                                      gated_move=bool(a.gated_move))
        row = {}
        for arm, e6 in (("REF", False), ("E6", True), ("E6b", True)):
            prev = RB.set_enabled_gated(e6)
            fired0 = RB.STATS_GATED[0]
            try:
                out = tm(z, None)
                ttnn.synchronize_device(dev)
                h = torch.Tensor(ttnn.to_torch(out)).to(torch.bfloat16)
                ttnn.deallocate(out)
                row[arm] = {"sha": sha(h), "shape": list(h.shape),
                            "e6_moves": RB.STATS_GATED[0] - fired0}
            except RuntimeError as e:
                row[arm] = {"sha": "THREW", "e6_moves": RB.STATS_GATED[0] - fired0,
                            "error": str(e).split("backtrace")[0].strip()[:300]}
            RB.set_enabled_gated(prev)
            print(f"  {which:5s} {arm:4s} e6_moves={row[arm]['e6_moves']:3d} "
                  f"sha={row[arm]['sha']} {row[arm].get('error','')}", flush=True)
        row["E6_matches_REF"] = row["E6"]["sha"] == row["REF"]["sha"]
        row["E6_stable"] = row["E6"]["sha"] == row["E6b"]["sha"]
        res[which] = row
    print(json.dumps({k: {"E6_matches_REF": v["E6_matches_REF"], "E6_stable": v["E6_stable"],
                          "e6_moves": v["E6"]["e6_moves"]} for k, v in res.items()}, indent=2))
    if a.out:
        json.dump({"model": a.model, "n": a.n, "gated_move": bool(a.gated_move),
                   "rejects": {f"{k[0]}{list(k[1])}": v for k, v in RB.REJECTS.items()},
                   "arms": res}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
