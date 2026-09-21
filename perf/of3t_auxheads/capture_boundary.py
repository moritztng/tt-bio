#!/usr/bin/env python3
"""Capture the 0.4.3 reference boundary for aux_heads, msa_module and input_embedder.

One float64 CPU run of upstream 0.4.3 on BUNDLE-MIN-043s own batch and replayed draws,
hooked at three module boundaries. For each it records, in float64:

  * the INPUTS the module was handed (args and kwargs, detached clones),
  * the OUTPUTS it produced -- the forward reference of PROTOCOL A18s first clause,
  * the COTANGENT dL/d(output) on every floating output tensor, which is what seeds our
    side s backward so the gradient comparison is taken at the same boundary,
  * the modules own parameter gradients from this very run.

The last item is the self-check that makes the boundary provably the reference s own: this
run must reproduce grads_f64_043.pt bit for bit on the captured sections (PROTOCOL A13
reported 4,170 of 4,170 bit-identical across two fresh processes, so anything else here
means the capture run is not the reference run).

It imports bundle_min.py rather than reimplementing the step: the draw replay, the dropout
disable, the no_autocast float64 context and the key gate are all load-bearing and a second
copy of them would be a second thing to keep right.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "of3t_reference"))
import bundle_min as BM  # noqa: E402

SECTIONS = ["aux_heads", "msa_module", "input_embedder"]


def _snap(x):
    """Detached float64-preserving clone of anything a hook is handed."""
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: _snap(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_snap(v) for v in x)
    return x


class BoundaryRecorder:
    """Forward pre/post hooks plus per-output cotangent hooks on one nn.Module."""

    def __init__(self, name, module):
        self.name = name
        self.module = module
        self.n_calls = 0
        self.inputs = None
        self.outputs = None
        self.cotangents = {}
        self.grad_enabled = None
        self._h = [
            module.register_forward_pre_hook(self._pre, with_kwargs=True),
            module.register_forward_hook(self._post, with_kwargs=True),
        ]

    def _pre(self, mod, args, kwargs):
        self.n_calls += 1
        # Keep the LAST grad-enabled call. Their trunk runs num_recycles + 1 passes and tapes
        # only the final one (model.py:230), so an earlier no-grad call is a different
        # function and must not become the boundary (PROTOCOL 3c-bis).
        if torch.is_grad_enabled():
            self.grad_enabled = True
            self.inputs = {"args": _snap(args), "kwargs": _snap(kwargs)}
        return None

    def _post(self, mod, args, kwargs, output):
        if not torch.is_grad_enabled():
            return None
        self.outputs = _snap(output)
        for key, t in self._walk(output):
            if torch.is_tensor(t) and t.requires_grad and t.is_floating_point():
                t.register_hook(self._make_cot(key))
        return None

    @staticmethod
    def _walk(out, prefix=""):
        if torch.is_tensor(out):
            yield (prefix or "out", out)
        elif isinstance(out, dict):
            for k, v in out.items():
                yield from BoundaryRecorder._walk(v, f"{prefix}{k}." if prefix else f"{k}")
        elif isinstance(out, (list, tuple)):
            for i, v in enumerate(out):
                yield from BoundaryRecorder._walk(v, f"{prefix}{i}" if not prefix else f"{prefix}{i}")

    def _make_cot(self, key):
        def hook(g):
            # Accumulate: a tensor read on several paths receives several cotangents, and the
            # one the boundary replay needs is their sum.
            prev = self.cotangents.get(key)
            self.cotangents[key] = g.detach().clone() if prev is None else prev + g.detach()
            return None
        return hook

    def close(self):
        for h in self._h:
            h.remove()


def rel_l2(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    return float(torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True, type=Path)
    ap.add_argument("--batch-sha256")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--replay-draws", required=True, type=Path)
    ap.add_argument("--reference-grads", type=Path,
                    help="grads_f64_043.pt, for the bit-identity self-check")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20260919)
    ap.add_argument("--num-recycles", type=int, default=0)
    ap.add_argument("--sections", default=",".join(SECTIONS))
    a = ap.parse_args()

    sections = [s for s in a.sections.split(",") if s]
    a.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    deterministic = BM.pin_deterministic_kernels(True)
    got = BM.sha256_file(a.batch)
    if a.batch_sha256 and got != a.batch_sha256:
        raise SystemExit(f"batch sha256 {got} != expected {a.batch_sha256}")

    raw_batch = torch.load(a.batch, weights_only=False)
    cfg, model, loss_fn, dropout = BM.build(torch.float64, a.seed, "cpu", a.num_recycles)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    sd = {k: (v.to(torch.float64) if torch.is_tensor(v) and v.is_floating_point() else v)
          for k, v in sd.items()}
    inc = model.load_state_dict(sd, strict=False)
    if inc.unexpected_keys:
        raise SystemExit(f"KEY GATE FAILED: {len(inc.unexpected_keys)} unexpected tensors; "
                         f"this is not the revision the checkpoint belongs to.")
    print(f"checkpoint loaded: {len(sd)} tensors, {len(inc.missing_keys)} missing, 0 unexpected",
          flush=True)

    batch = BM.move(raw_batch, "cpu", torch.float64)
    pinned = BM.rng_state(model)
    replay = torch.load(a.replay_draws, map_location="cpu", weights_only=False)

    recs = {}
    for s in sections:
        recs[s] = BoundaryRecorder(s, getattr(model, s))

    BM.set_rng_state(pinned, model)
    rec = BM.DrawRecorder(replay)
    t0 = time.time()
    loss, breakdown, out = BM.forward_loss(model, loss_fn, batch, rec, disable_autocast=True)
    t_fwd = time.time() - t0
    print(f"forward {t_fwd:.0f}s  loss {float(loss):.15f}", flush=True)

    t0 = time.time()
    loss.backward()
    t_bwd = time.time() - t0
    print(f"backward {t_bwd:.0f}s", flush=True)
    for r in recs.values():
        r.close()

    # ---- the boundary artifacts ------------------------------------------------------------
    report = {
        "instrument": "0.4.3 boundary capture for aux_heads / msa_module / input_embedder",
        "loss": float(loss),
        "loss_terms": {k: (float(v) if torch.is_tensor(v) and v.numel() == 1 else str(v)) for k, v in breakdown.items()},
        "num_recycles_pinned": a.num_recycles,
        "dropout": dropout,
        "deterministic_kernels": deterministic,
        "replayed_draws": {"file": a.replay_draws.name,
                           "sha256": BM.sha256_file(a.replay_draws),
                           "n_mismatch": len(rec.mismatch)},
        "batch": {"file": a.batch.name, "sha256": got},
        "checkpoint": {"file": a.checkpoint.name, "sha256": BM.sha256_file(a.checkpoint)},
        "timings_s": {"forward": t_fwd, "backward": t_bwd},
        "sections": {},
    }

    ref = None
    if a.reference_grads:
        print("loading published reference for the bit-identity self-check...", flush=True)
        ref = torch.load(a.reference_grads, map_location="cpu", weights_only=False)

    named = dict(model.named_parameters())
    for s in sections:
        r = recs[s]
        pg, presence = {}, {}
        for n, p in named.items():
            if not n.startswith(s + "."):
                continue
            presence[n] = p.grad is not None
            pg[n] = p.grad.detach().to(torch.float64).clone() if p.grad is not None else None
        blob = {
            "inputs": r.inputs,
            "outputs": r.outputs,
            "cotangents": r.cotangents,
            "param_grads": pg,
            "param_presence": presence,
        }
        torch.save(blob, a.out / f"boundary_{s}.pt")
        ent = {
            "n_forward_calls": r.n_calls,
            "n_params": len(pg),
            "n_with_gradient": sum(1 for v in pg.values() if v is not None),
            "cotangent_keys": sorted(r.cotangents.keys()),
            "output_shapes": {k: list(v.shape) for k, v in r._walk(r.outputs)
                              if torch.is_tensor(v)},
            "file": f"boundary_{s}.pt",
            "sha256": BM.sha256_file(a.out / f"boundary_{s}.pt"),
        }
        if ref is not None:
            n_id = n_cmp = 0
            worst, worst_name = 0.0, None
            for n, g in pg.items():
                if g is None or n not in ref or ref[n] is None:
                    continue
                n_cmp += 1
                rg = ref[n].double()
                if torch.equal(g, rg):
                    n_id += 1
                else:
                    d = rel_l2(g, rg)
                    if d > worst:
                        worst, worst_name = d, n
            ent["self_check_vs_published_reference"] = {
                "n_compared": n_cmp, "n_bit_identical": n_id,
                "worst_rel_l2": worst, "worst_tensor": worst_name,
                "reproduced": n_id == n_cmp and n_cmp == ent["n_with_gradient"],
            }
        report["sections"][s] = ent
        print(f"{s}: {ent['n_params']} params, {ent['n_with_gradient']} with gradient, "
              f"cotangents {ent['cotangent_keys']}", flush=True)
        if ref is not None:
            print(f"   self-check {ent['self_check_vs_published_reference']}", flush=True)

    report["total_s"] = time.time() - t_start
    (a.out / "capture_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps(report["sections"], indent=1, default=str)[:2000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
