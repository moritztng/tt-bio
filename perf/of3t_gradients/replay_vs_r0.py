#!/usr/bin/env python3
"""The banked r = 0 CPU replay of their own step, against BOTH references.

D18 said BUNDLE-MIN was taped in train mode, so its published gradient is one dropout draw.
`of3t-reference` has since republished at r = 0 (`grads_f64_r0.pt`). This row's CPU replay at
r = 0 was already banked, so the dropout story can be tested from the other side for the cost
of two loads: the SAME replay gradients against the WITHDRAWN train-mode reference and against
the REPUBLISHED r = 0 one. If dropout is the whole gap, the second reads far smaller.

Upstream against upstream. Nothing on a card, nothing of ours, so a disagreement here prices
the reference and not our stack.
"""
import hashlib, json, sys, time
from pathlib import Path
import torch

BUNDLE = Path("/home/ttuser/of3t/bundle_min")
CAP = Path("/home/ttuser/of3t_gradients/cap")
BLOCKS = [0, 23, 47]
REFS = {
    "withdrawn_train_mode": ("grads_f64_recycles0.pt",
                             "1a6af8bba8c5fa33dd5366f15e87fffc22595a351c3c77b476a56fab8b4b4b74"),
    "republished_r0":       ("grads_f64_r0.pt",
                             "89457d8977327699c84fc90741a013bc369f835dea6008676492c786fb87f113"),
}


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def rel(a, b):
    return float(torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-300))


def main():
    t0 = time.time()
    rep = {"instrument": "banked r = 0 CPU replay vs both references, upstream against upstream",
           "blocks": BLOCKS, "refs": {}}
    ours = {}
    for i in BLOCKS:
        blob = torch.load(CAP / f"block{i}_boundary.pt", map_location="cpu", weights_only=False)
        for k, g in blob["grad"].items():
            if g is not None:
                ours[f"pairformer_stack.blocks.{i}.{k}"] = g.double()
        del blob
    rep["n_banked_tensors"] = len(ours)
    print(f"[{time.time()-t0:.0f}s] {len(ours)} banked replay gradients", flush=True)

    for label, (fn, declared) in REFS.items():
        p = BUNDLE / fn
        got = sha256(p)
        print(f"[{time.time()-t0:.0f}s] {fn} sha256 {got[:16]}... match={got == declared}",
              flush=True)
        ref = torch.load(p, map_location="cpu", weights_only=False)
        per = {}
        for n, g in ours.items():
            r = ref.get(n)
            per[n] = None if r is None else rel(g, r.double())
        vals = sorted(v for v in per.values() if v is not None)
        worst_n = max((n for n, v in per.items() if v is not None), key=lambda n: per[n])
        by_block = {}
        for i in BLOCKS:
            pre = f"pairformer_stack.blocks.{i}."
            vb = sorted(v for n, v in per.items() if v is not None and n.startswith(pre))
            wn = max((n for n, v in per.items() if v is not None and n.startswith(pre)),
                     key=lambda n: per[n])
            by_block[str(i)] = {"n": len(vb), "median": vb[len(vb) // 2], "worst": vb[-1],
                                "worst_tensor": wn, "over_bar_5e-2": sum(1 for v in vb if v > 5e-2)}
        rep["refs"][label] = {
            "file": fn, "sha256": got, "sha256_declared": declared, "sha256_match": got == declared,
            "n_compared": len(vals), "n_absent_either_side": sum(1 for v in per.values() if v is None),
            "median": vals[len(vals) // 2], "worst": vals[-1], "worst_tensor": worst_n,
            "over_bar_5e-2": sum(1 for v in vals if v > 5e-2),
            "by_block": by_block, "per_tensor": per,
        }
        print(f"[{time.time()-t0:.0f}s] {label}: median {vals[len(vals)//2]:.4e} "
              f"worst {vals[-1]:.4e} on {worst_n}, over bar {rep['refs'][label]['over_bar_5e-2']}"
              f"/{len(vals)}", flush=True)
        del ref
    # ---- A15: the compared set's share of the squared gradient norm, never the count alone ---
    ref = torch.load(BUNDLE / REFS["republished_r0"][0], map_location="cpu", weights_only=False)
    tot = sum(float(torch.linalg.vector_norm(g.double())) ** 2 for g in ref.values()
              if g is not None)
    got = sum(float(torch.linalg.vector_norm(ref[n].double())) ** 2 for n in ours
              if ref.get(n) is not None)
    by_sec = {}
    for n, g in ref.items():
        if g is None:
            continue
        by_sec[n.split(".")[0]] = by_sec.get(n.split(".")[0], 0.0) + \
            float(torch.linalg.vector_norm(g.double())) ** 2
    del ref
    rep["a15_reach"] = {
        "n_compared": len(ours), "n_reference_tensors": 4147,
        "squared_norm_total": tot, "squared_norm_compared": got,
        "share_of_squared_norm_pct": 100.0 * got / tot,
        "scope": "pairformer blocks 0, 23 and 47 only -- three of forty-eight",
        "by_section_pct": {k: 100.0 * v / tot for k, v in sorted(by_sec.items())},
    }
    print(f"[{time.time()-t0:.0f}s] A15: {len(ours)} tensors holding "
          f"{rep['a15_reach']['share_of_squared_norm_pct']:.4f} % of the squared gradient norm",
          flush=True)

    # ---- SS3e: the bar must be shown to be capable of failing --------------------------------
    # One tensor's replay gradient scaled by 1.01, against the republished reference, on the
    # arm that is otherwise the tightest available. The control has to move THAT tensor and
    # leave every other one where it was; a control that shifts the whole set is measuring the
    # harness rather than the check (memory `negative-control-must-break-what-check-reads`).
    base = rep["refs"]["republished_r0"]["per_tensor"]
    victim = min((n for n, v in base.items() if v is not None and v < 5e-2),
                 key=lambda n: base[n])
    ref = torch.load(BUNDLE / REFS["republished_r0"][0], map_location="cpu", weights_only=False)
    moved, unmoved = {}, 0
    for n, g in ours.items():
        r = ref.get(n)
        if r is None:
            continue
        v = rel(g * 1.01 if n == victim else g, r.double())
        if base[n] is None or abs(v - base[n]) > 1e-12:
            moved[n] = {"before": base[n], "after": v}
        else:
            unmoved += 1
    del ref
    rep["negative_control"] = {
        "question": "which check fails if our model is replaced by zeros?",
        "answer": "every one. A zeroed gradient gives rel L2 = ||0 - g_ref|| / ||g_ref|| = "
                  "exactly 1.0 on all 171 tensors, 171/171 over the 5.0e-02 bar, median 1.0 -- "
                  "which is why the median is reported beside the count: the withdrawn "
                  "reference's own median of 1.096 is within noise of the zero-model answer, "
                  "and only the republished reference's 6.22e-02 is a number a zero model "
                  "could not have produced.",
        "perturbation": "x1.01 on one tensor's replay gradient",
        "victim": victim, "victim_before": base[victim],
        "victim_after": moved.get(victim, {}).get("after"),
        "n_moved": len(moved), "n_unmoved": unmoved,
        "rejected": (len(moved) == 1 and victim in moved
                     and moved[victim]["after"] > base[victim]),
        "other_movers": {k: v for k, v in moved.items() if k != victim},
    }
    nc = rep["negative_control"]
    print(f"[{time.time()-t0:.0f}s] control: {victim} {nc['victim_before']:.4e} -> "
          f"{nc['victim_after']:.4e}, moved {nc['n_moved']}, unmoved {nc['n_unmoved']}, "
          f"rejected={nc['rejected']}", flush=True)

    # zeros arm, run rather than asserted
    zero_over = zero_vals = None
    ref = torch.load(BUNDLE / REFS["republished_r0"][0], map_location="cpu", weights_only=False)
    zs = [rel(torch.zeros_like(g), ref[n].double()) for n, g in ours.items() if ref.get(n) is not None]
    del ref
    zs.sort()
    rep["negative_control"]["zero_model_measured"] = {
        "n": len(zs), "median": zs[len(zs) // 2], "worst": zs[-1], "best": zs[0],
        "over_bar_5e-2": sum(1 for v in zs if v > 5e-2)}
    print(f"[{time.time()-t0:.0f}s] zero model: median {zs[len(zs)//2]:.6f}, "
          f"over bar {rep['negative_control']['zero_model_measured']['over_bar_5e-2']}/{len(zs)}",
          flush=True)

    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
    print(f"[{time.time()-t0:.0f}s] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
