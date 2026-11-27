#!/usr/bin/env python3
"""Upstream's own CPU-autocast bf16 run vs its fp32 run on one loss-parity dump, per candidate:
the trunk output s (input of insertion_q_proj, Frobenius rel over valid tokens, insertion rows vs
rest) and every insertion loss term. Answers whether upstream's bf16 floor for a term is a property
of a near-fp32 trunk or a draw on a sensitive term. Usage: autocast_s_probe.py --ref DIR [--batch batch08]
"""
import argparse, os, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from loss_parity_reference import build_model, load_batch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--batch", default="batch08")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    _, model = build_model(Path(os.environ["RFD4_REPO"]).expanduser(), torch.float32)
    got = {}
    model.nn.insertion_q_proj.register_forward_pre_hook(lambda m, x: got.__setitem__("s", x[0].detach().float().clone()))
    batch, meta, _, _ = load_batch(a.ref / f"{a.batch}.npz", torch.float32)
    S, L = {}, {}
    for arm in ("fp32", "bf16"):
        torch.set_rng_state(meta["rng_state"])
        with torch.no_grad():
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=arm == "bf16"):
                out = model.call_nn(dict(batch), n_recycle=meta["depth"])
            out = {k: ({m: v.float() for m, v in d.items()} if isinstance(d, dict) else d.float()) for k, d in out.items()}
            S[arm], L[arm] = got["s"], model.fm.compute_loss(batch=batch, nn_out=out)
    m, ins = batch["padding_mask"].bool(), batch["insertion_source_idx"] >= 0
    for c in range(S["fp32"].shape[0]):
        for lab, sel in (("all", m[c]), ("insertion", m[c] & ins[c]), ("rest", m[c] & ~ins[c])):
            if sel.any():
                r = S["fp32"][c][sel]
                print(f"{a.batch} cand{c} {lab:9s} n={int(sel.sum()):4d} s rel {float((S['bf16'][c][sel] - r).norm() / r.norm()):.3e}")
    for k in (k for k in L["fp32"] if "insertion" in k):
        x, y = L["fp32"][k].flatten().tolist(), L["bf16"][k].flatten().tolist()
        print(k, "rel per cand", [f"{abs(p - q) / abs(p):.3e}" if p else "-" for p, q in zip(x, y)])
    print("winners", list(meta["winners"]), "depth", meta["depth"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
