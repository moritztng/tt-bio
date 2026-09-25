#!/usr/bin/env python3
"""The VJP, per-parameter worst case, with the machine's own run-to-run floor beside it.

`graddigest.py` asked the identity question first and its A/A control refused it: two runs of
the SAME arm, same process design, same card, same clock, produced different whole-set digests
(`out/grads_a1.json` 14ec02c0... against `out/grads_a2.json` f514c6af...). So this backward is
not bit-reproducible run to run, a cross-arm digest match was never available, and a digest
MISmatch would have proved nothing. That is the whole reason the control is run first.

What is available is the comparison the floor makes readable. Three arms in ONE process, so
the weights, the fixture, the device and the clock are shared and the only difference is the
arm:

    A1  zeros=host     the reference, kept
    A2  zeros=host     the A/A FLOOR -- the same arm twice, so its spread is the machine's
    B1  zeros=device   the fix, kept
    B2  zeros=device   the B/B FLOOR

BOTH floors, because the first reading of this comparison had only A/A and it was misread.
A/A left 1,388 of 1,648 gradients bit-identical and A/B only 196, which looks like the fix
moving four times as many tensors as a rerun does. It is not a claim either floor can support
on its own: if B/B also leaves ~200 bit-identical, then what the arms differ in is WHICH
tensors the backward's own nondeterminism lands on, not how far the fix moves them. A
one-sided floor cannot tell those apart.

Per parameter, against A1, in float64: relative L2 ``||x-y|| / ||x||`` and cosine. Reported as
a worst case LOCATED BY PARAMETER PATH, per A46 clause 1, never as a mean -- a mean over 1,648
tensors hides the one that moved. The verdict is a comparison, not a threshold: the fix is
inside the machine's own noise if its worst case does not exceed A/A's.

`perf/hallgrad/gradcheck.py` is the absolute bar against a float64 reference validated by
central finite differences, and `perf/of3t_orchestrator/bwd/batch_gate.sh` runs it over the
batch. This is the complementary reading: that one says the backward is right, this one says
the change did not move it.

    gradcompare.py --out perf/of3t_zerosfill/out/gradcompare.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.clocksample import during                                      # noqa: E402
from perf.of3t_perf import step as S                                     # noqa: E402

ARMS = [("A1", "host"), ("A2", "host"), ("B1", "device"), ("B2", "device")]


def run_arm(zeros, held, dev, params, S_, ag, TT, ttnn, torch, cycles):
    """One taped forward + backward. Returns {param path: float32 CPU gradient}."""
    for t in params.values():
        t._grad = None
    ag.DEVICE_ZEROS = (zeros == "device")
    snap_args, snap_kwargs = held["trunk_snap"]
    args_ = S_._rehydrate(snap_args, dev)
    kwargs_ = {k: v for k, v in S_._rehydrate(snap_kwargs, dev).items()
               if k != "progress_fn"}
    trunk = held["trunk"][0]
    trunk.num_cycles = cycles
    gc.collect()
    t0 = time.perf_counter()
    with ag.tape():
        _s, z = trunk(*args_, **kwargs_)
        ttnn.synchronize_device(dev)
    fwd = time.perf_counter() - t0
    if not isinstance(z, ag.Tensor):
        raise SystemExit("the trunk output is not taped; nothing to differentiate")
    nodes = len(ag._reverse_topo([z]))
    zr = z.value
    seed = ttnn.from_torch(torch.ones(tuple(int(d) for d in zr.shape)),
                           layout=ttnn.TILE_LAYOUT, device=dev, dtype=zr.dtype)
    rs = TT.recompute_scope()
    rs.__enter__()
    t0 = time.perf_counter()
    try:
        ag.backward([z], [seed])
        ttnn.synchronize_device(dev)
    finally:
        rs.__exit__(None, None, None)
    bwd = time.perf_counter() - t0
    g = {n: ttnn.to_torch(t.grad).to(torch.float32)
         for n, t in params.items() if getattr(t, "grad", None) is not None}
    ag.release_pins()
    gc.collect()
    return g, {"forward_s": round(fwd, 2), "backward_s": round(bwd, 2),
               "tape_nodes": nodes, "params_with_grad": len(g)}


def compare(ref, other, torch):
    """Per-parameter relative L2 and cosine, worst case located by path."""
    rows, shape_mismatch, missing = [], [], []
    for n, x in ref.items():
        y = other.get(n)
        if y is None:
            missing.append(n)
            continue
        if tuple(x.shape) != tuple(y.shape):
            shape_mismatch.append(n)
            continue
        xf, yf = x.to(torch.float64).flatten(), y.to(torch.float64).flatten()
        nx = float(xf.norm())
        rel = float((xf - yf).norm()) / nx if nx else 0.0
        den = nx * float(yf.norm())
        cos = float((xf * yf).sum()) / den if den else 1.0
        rows.append((rel, cos, n, nx))
    rows.sort(key=lambda r: -r[0])
    rel = [r[0] for r in rows]

    def pct(q):
        return round(rel[min(len(rel) - 1, int(round((1 - q) * (len(rel) - 1))))], 9)

    # A worst case over 1,648 tensors is one tensor, and three of them are not reproducible
    # run to run here (`*.msa_module.blocks.{0,1,2}.pwa.z_norm_bias`, rel L2 0.58-1.57 between
    # two runs of the IDENTICAL arm, on gradients of L2 ~3e4, so not a divide-by-small).
    # Report the distribution beside the worst case, and the worst case with those three
    # excluded, so the reading does not rest on the one tensor that moves anyway.
    stable = [r for r in rows if not r[2].endswith(".pwa.z_norm_bias")]
    return {
        "n_compared": len(rows), "missing_in_other": missing,
        "shape_mismatch": shape_mismatch,
        "worst_rel_l2": round(rows[0][0], 9) if rows else None,
        "worst_rel_l2_param": rows[0][2] if rows else None,
        "min_cosine": round(min(r[1] for r in rows), 9) if rows else None,
        "min_cosine_param": min(rows, key=lambda r: r[1])[2] if rows else None,
        "rel_l2_p50": pct(0.50), "rel_l2_p90": pct(0.90), "rel_l2_p99": pct(0.99),
        "n_over_1e-2": sum(1 for x in rel if x > 1e-2),
        "worst_rel_l2_excl_pwa_z_norm_bias":
            round(stable[0][0], 9) if stable else None,
        "worst_param_excl_pwa_z_norm_bias": stable[0][2] if stable else None,
        "min_cosine_excl_pwa_z_norm_bias":
            round(min(r[1] for r in stable), 9) if stable else None,
        "top10_by_rel_l2": [{"param": r[2], "rel_l2": round(r[0], 9),
                             "cosine": round(r[1], 9), "ref_l2": round(r[3], 6)}
                            for r in rows[:10]],
        "n_bit_identical": sum(1 for r in rows if r[0] == 0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--out", type=Path,
                    default=REPO / "perf/of3t_zerosfill/out/gradcompare.json")
    a = ap.parse_args()

    out = {"doc": __doc__.splitlines()[0], "argv": sys.argv[1:], "arms": {}, "env": {
        "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(),
        "board": "pc card 0 -- Blackhole p150a, custom 130-core firmware"},
        "config": {"crop": a.tokens, "cycles": a.cycles,
                   "arms": [{"tag": t, "zeros": z} for t, z in ARMS]}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump = lambda: a.out.write_text(json.dumps(out, indent=1, default=str))  # noqa: E731
    dump()

    with during() as clk:
        try:
            import torch
            import ttnn
            from tt_bio import autograd as ag
            from tt_bio import taped_ttnn as TT
            from tt_bio.tenstorrent import get_device

            held, _meta = S.capture(a.tokens, out)
            dev = get_device()
            out["env"]["arch"] = str(dev.arch())
            params = S.declare_weights(held["trunk"][0], out)

            exact_ctx = ag.exact_training(False)
            exact_ctx.__enter__()
            out["env"]["exact_training_ops"] = list(ag.exact_training_ops())

            # A1 and B1 are kept; A2 and B2 are compared and freed. Peak is two float32
            # copies of 165.2M gradient elements, about 1.3 GB, on a 30 GB host with no swap.
            kept, pairs = {}, [("A1", "A2", "A/A FLOOR"), ("B1", "B2", "B/B FLOOR"),
                               ("A1", "B1", "A/B FIX")]
            for tag, zeros in ARMS:
                g, meta = run_arm(zeros, held, dev, params, S, ag, TT, ttnn, torch, a.cycles)
                meta["zeros"] = zeros
                meta["device_zeros"] = bool(ag.DEVICE_ZEROS)
                out["arms"][tag] = meta
                print(f"[{tag}] zeros={zeros} fwd {meta['forward_s']} s "
                      f"bwd {meta['backward_s']} s nodes {meta['tape_nodes']} "
                      f"grads {meta['params_with_grad']}", flush=True)
                kept[tag] = g
                for x, y, lab in pairs:
                    key = f"{x}_vs_{y}"
                    if y != tag or key in out or x not in kept:
                        continue
                    c = out[key] = compare(kept[x], kept[y], torch)
                    c["label"] = lab
                    print(f"   {lab}: worst rel L2 {c['worst_rel_l2']} at "
                          f"{c['worst_rel_l2_param']}, min cosine {c['min_cosine']}, "
                          f"{c['n_bit_identical']}/{c['n_compared']} bit-identical", flush=True)
                    print(f"      excl pwa.z_norm_bias: worst "
                          f"{c['worst_rel_l2_excl_pwa_z_norm_bias']} at "
                          f"{c['worst_param_excl_pwa_z_norm_bias']} | p50 {c['rel_l2_p50']} "
                          f"p90 {c['rel_l2_p90']} p99 {c['rel_l2_p99']} | "
                          f"{c['n_over_1e-2']} over 1e-2", flush=True)
                if tag in ("A2", "B2"):
                    del kept[tag]
                    gc.collect()
                dump()

            aa = out.get("A1_vs_A2", {}).get("worst_rel_l2")
            bb = out.get("B1_vs_B2", {}).get("worst_rel_l2")
            ab = out.get("A1_vs_B1", {}).get("worst_rel_l2")
            if None not in (aa, bb, ab):
                floor = max(aa, bb)
                out["verdict"] = {
                    "aa_floor_worst_rel_l2": aa, "bb_floor_worst_rel_l2": bb,
                    "ab_fix_worst_rel_l2": ab, "floor_used": floor,
                    "ab_over_floor": round(ab / floor, 4) if floor else None,
                    "inside_the_floor": bool(ab <= floor),
                    "bit_identical": {k: out[k]["n_bit_identical"] for k in
                                      ("A1_vs_A2", "B1_vs_B2", "A1_vs_B1")},
                    "worst_excl_pwa_z_norm_bias": {
                        k: out[k]["worst_rel_l2_excl_pwa_z_norm_bias"] for k in
                        ("A1_vs_A2", "B1_vs_B2", "A1_vs_B1")},
                    "rel_l2_p90": {k: out[k]["rel_l2_p90"] for k in
                                   ("A1_vs_A2", "B1_vs_B2", "A1_vs_B1")},
                    "note": "A/A and B/B are each the same arm twice, so between them they "
                            "are the machine's own run-to-run spread on this backward. The "
                            "fix is inside that or it is not; there is no threshold to move "
                            "here. Read the bit-identical counts together: if B/B leaves as "
                            "few bit-identical as A/B does, the arms differ in WHICH tensors "
                            "the nondeterminism lands on, not in how far the fix moves them."}
            exact_ctx.__exit__(None, None, None)
        except Exception:                                                # noqa: BLE001
            out["error"] = traceback.format_exc()[-6000:]
    out["env"]["aiclk_during"] = clk.summary()
    out["env"]["aiclk_line"] = clk.line(0)
    dump()
    print()
    print(json.dumps(out.get("verdict", {"error": out.get("error", "")[-400:]}), indent=1))
    print(out["env"].get("aiclk_line"))
    print(f"-> {a.out}")
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
