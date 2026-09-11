"""Cross-model A/B for TT_BIO_TRIMUL_MASK_AFTER_MOVE on the shared TriangleMultiplication.

One process, one device, arms interleaved: A (flag off, today's default), B (flag on), A2 (flag
off again, the A/A floor). Every arm runs the SAME `TriangleMultiplication` instance on the SAME
input buffer, so a difference is the flag and nothing else -- except on a card that miscomputes,
which is what A2 is for (pc card 0, `pc-card0-512aa-fold-nondeterminism`).

Weights are each model's real checkpoint, remapped by that model's own remap, so the widths, the
biases and the `gated_move` default are the ones the model ships. Masks are [1, N, N] float, the
rank the models build.

    python3 perf/b2x_crossmodel/trimul_ab.py --model protenix-v2 --n 256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio import reblock_permute as RB  # noqa: E402

BOLTZ = os.path.expanduser(os.environ.get("BOLTZ_CACHE", "~/.boltz"))


# ------------------------------------------------------------------ weight sources

def _tri_pair(sd, prefix):
    """Pull the two trimuls under `prefix` out of a remapped block state dict."""
    out = {}
    for which, keys in (("start", ("tri_mul_out", "triangle_multiplication_start")),
                        ("end", ("tri_mul_in", "triangle_multiplication_end"))):
        for key in keys:
            p = f"{prefix}{key}."
            w = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
            if w:
                out[which] = w
                break
        else:
            raise AssertionError(f"no weights under {prefix}{keys}")
    return out


def load_protenix_v2():
    ck = torch.load(f"{BOLTZ}/protenix-v2.pt", map_location="cpu", weights_only=True)
    ck = ck.get("model", ck)
    pfx = "module.pairformer_stack.blocks.0."
    blk = {k[len(pfx):]: v for k, v in ck.items() if k.startswith(pfx)}
    assert blk, "protenix-v2 block 0 not found"
    from tt_bio.protenix_weights import remap_pairformer_block
    return _tri_pair(remap_pairformer_block(blk), ""), False, "protenix-v2 pairformer block 0"


def load_openfold3():
    from tt_bio.openfold3_weights import remap_pairformer_block, _sub
    sd = torch.load(f"{BOLTZ}/of3-p2-155k.pt", map_location="cpu", weights_only=False)
    blk = _sub(sd, "pairformer_stack.blocks.0")
    return _tri_pair(remap_pairformer_block(blk), ""), False, "openfold3 pairformer block 0"


def load_opendde():
    from tt_bio.opendde import load_opendde_checkpoint, route_opendde_weights
    sd = load_opendde_checkpoint()
    routed = route_opendde_weights(sd)
    # The refiner IS a Pairformer and is built with gated_move=True, as is the shared
    # Protenix-family trunk at c_z=384. Score the refiner block 0.
    return _tri_pair(routed["refiner"], "layers.0."), True, "opendde refiner block 0 (c_z=384)"


def load_af2():
    from tt_bio.af2_weights import load_af2_state_dict
    sd = load_af2_state_dict(f"{BOLTZ}/af2/params/params_model_1_ptm.npz")
    out = {}
    for which, key in (("start", "tri_mul_out"), ("end", "tri_mul_in")):
        p = f"evoformer.0.{key}."
        out[which] = {k[len(p):]: v for k, v in sd.items() if k.startswith(p)}
        assert out[which], f"no AF2 weights under {p}"
    # af2.py builds these with gated_move=False ("the fused chunk+gate forward move takes no
    # bias"), and AF2 is the only checkpoint whose trimul carries biases.
    return out, False, "af2-ig evoformer block 0 (biased trimul)"


LOADERS = {
    "protenix-v2": load_protenix_v2,
    "openfold3": load_openfold3,
    "opendde": load_opendde,
    "af2": load_af2,
}


# ------------------------------------------------------------------ the A/B

def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.int16).numpy().tobytes()).hexdigest()[:16]


def run(model: str, N: int, reps: int, mask_kind: str, seed: int, loaded=None):
    tri, gated_move, label = loaded or LOADERS[model]()
    dev = T.get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    c_z = tri["start"]["norm_in.weight"].shape[0]
    hidden = tri["start"]["g_in.weight"].shape[0] // 2
    print(f"# {model}: {label}  c_z={c_z} hidden={hidden} N={N} "
          f"gated_move={gated_move} mask={mask_kind}", flush=True)

    torch.manual_seed(seed)
    z = torch.randn(1, N, N, c_z, dtype=torch.float32)
    z_tt = ttnn.from_torch(z, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    if mask_kind == "none":
        mask_tt = None
    else:
        m = torch.ones(1, N, N, dtype=torch.float32)
        if mask_kind == "ragged":
            # a real fold's pair mask: the tile-padding tail is zero
            keep = N - 24
            m[:, keep:, :] = 0.0
            m[:, :, keep:] = 0.0
        mask_tt = ttnn.from_torch(m, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    results = {}
    for which in ("start", "end"):
        tm = T.TriangleMultiplication(which == "end", tri[which], ckc, gated_move=gated_move)
        per_arm = {}
        for arm, on in (("A", False), ("B", True), ("A2", False), ("B2", True)):
            prev = T.set_trimul_mask_after_move(on)
            shas, dts = [], []
            fired0 = RB.STATS_GATED[0]
            for _ in range(reps):
                t0 = time.perf_counter()
                out = tm(z_tt, mask_tt)
                ttnn.synchronize_device(dev)
                dts.append(time.perf_counter() - t0)
                h = torch.Tensor(ttnn.to_torch(out)).to(torch.bfloat16)
                ttnn.deallocate(out)
                shas.append(_sha(h))
            fired = (RB.STATS_GATED[0] - fired0) / reps
            T.set_trimul_mask_after_move(prev)
            per_arm[arm] = {"sha": shas, "ms": round(1e3 * min(dts), 3),
                            "e6_moves_per_call": fired}
            print(f"  {which:5s} arm {arm:2s} on={int(on)}  E6/call={fired:g}  "
                  f"sha={shas}  {per_arm[arm]['ms']} ms", flush=True)
        results[which] = per_arm
    ttnn.deallocate(z_tt)
    if mask_tt is not None:
        ttnn.deallocate(mask_tt)

    verdict = {}
    for which, per_arm in results.items():
        a, b, a2, b2 = (set(per_arm[k]["sha"]) for k in ("A", "B", "A2", "B2"))
        floor_clean = len(a | a2) == 1
        arm_b_stable = len(b | b2) == 1
        verdict[which] = {
            "B_fired_e6_per_call": per_arm["B"]["e6_moves_per_call"],
            "A_fired_e6_per_call": per_arm["A"]["e6_moves_per_call"],
            "A_A2_floor_bit_exact": floor_clean,
            "B_B2_stable": arm_b_stable,
            "A_eq_B": (a | a2) == (b | b2) and floor_clean and arm_b_stable,
            "A_shas": sorted(a | a2), "B_shas": sorted(b | b2),
        }
    return {"model": model, "label": label, "c_z": c_z, "hidden": hidden, "N": N,
            "gated_move": gated_move, "mask": mask_kind, "reps": reps,
            "gated_stats": list(RB.STATS_GATED),
            "rejects": {f"{k[0]}{list(k[1])}": v for k, v in RB.REJECTS.items()},
            "arms": results, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(LOADERS))
    ap.add_argument("--n", type=int, nargs="+", default=[256])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mask", nargs="+", default=["ones"],
                    choices=["none", "ones", "ragged"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    loaded = LOADERS[a.model]()
    runs = []
    for N in a.n:
        for mk in a.mask:
            r = run(a.model, N, a.reps, mk, a.seed, loaded=loaded)
            fired = any(v["B_fired_e6_per_call"] for v in r["verdict"].values())
            ok = all(v["A_eq_B"] for v in r["verdict"].values())
            r["vacuous"] = not fired
            print(f"== {a.model} N={N} mask={mk}: "
                  f"{'BIT-EXACT' if ok else 'DIFFERS'}"
                  f"{'' if fired else '  (VACUOUS: zero E6 moves in the flag-on arm)'}",
                  flush=True)
            if not fired:
                print("   rejects: " + json.dumps(r["rejects"]), flush=True)
            runs.append(r)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(runs, f, indent=2)
        print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
