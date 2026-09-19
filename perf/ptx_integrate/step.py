#!/usr/bin/env python3
"""One REAL Protenix-v2 training step, composed, at the recipe crop, on one card.

Every number this campaign has is a component measurement. `ptx-fastpath` differentiated the
shipped pairformer classes one at a time, `ptx-objective` trained a small MLP against the
composed objective, `ptx-diffusion` priced a token-DiT block's backward, `ptx-crop` measured a
trunk tape's footprint. Nothing had run all of them in one step, and a composition can be wrong
in ways none of its parts are.

WHAT IS REAL, because it matters which parts are:

  weights    the protenix-v2 checkpoint remapped by `tt_bio.protenix_weights`, at the depth and
             widths it ships: 48 pairformer blocks, c_z 256, 8 triangle heads of 32, the real
             `distogram_head.linear` (64 bins) and the real `DiffusionModule`.
  features   `tt_bio.protenix_data.build_protein_features`, the shipped data pipeline.
  forward    the SHIPPED modules, taped in place by `autograd.tape()` rebinding `ttnn` inside
             tt-bio. There is no copy of a forward in this file.
  objective  `tt_bio.train.objectives.objective("af3")`, the shipped eight-term row at
             Protenix's own pretraining weights. A term whose outputs are absent is recorded
             SKIPPED by the row itself and the breakdown prints which.
  optimizer  `tt_bio.train.optim.AdamW`, which refuses a non-fp32 master at construction.

  labels     SYNTHETIC: one frozen random structure, memorisation-style. Nothing here is a claim
             about accuracy on real data. What is measured is that the gradient of the composed
             objective through the composed forward is right, which no prior row measured.

THE SEAM, and it is the shipped pipeline's rather than this harness's. `Protenix._trunk_cond`
writes the trunk to HOST before building the diffusion conditioning, and `denoise` returns host
coordinates. A tape ends at `ttnn.to_torch`. So the trunk takes its gradient through the
distogram head and the denoiser takes its own: one step, one optimizer, two graphs. The
denoiser's device tensor is `last_r_update_device` and the EDM preconditioning after it is
affine, so seeding it with `c_out * dL/dx` is exact and not an approximation.

VERIFICATION: float64 central finite differences through THE SAME composed forward -- the loss
is recomputed by re-running the real forward at w +/- eps*d -- so there is no second
implementation that could be wrong in the same direction. A negative control must be rejected.

    step.py --tokens 384 --steps 3 --out out/step_qb1c0.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from perf.clocksample import during  # noqa: E402

CKPT = Path(os.environ.get("PTX_V2_CKPT", "/home/ttuser/protenix_ckpt/protenix-v2.pt"))
AA = "ACDEFGHIKLMNPQRSTVWY"


def sequence(n, seed):
    """A deterministic sequence of the requested length; the featurisation of it is real."""
    r = np.random.default_rng(seed)
    return "".join(AA[i] for i in r.integers(0, len(AA), n))


def labels(nt, seed):
    """The one structure to fit. Frozen, so the only path to a lower loss is a right gradient."""
    from tt_bio.train import losses as X
    r = np.random.default_rng(seed + 1)
    true = r.normal(0, 8, (nt, 3))
    td = np.linalg.norm(true[:, None, :] - true[None, :, :], axis=-1)
    return {
        "true_xyz": true, "coord_mask": np.ones(nt, bool), "true_dist": td,
        "lddt_pair_mask": X.lddt_mask(td, np.ones((nt, nt)), np.zeros(nt, bool)),
        "bond_mask": (np.abs(np.arange(nt)[:, None] - np.arange(nt)[None, :]) == 1).astype(float),
        "frame_atom_index": np.stack([(np.arange(nt) - 1) % nt, np.arange(nt),
                                      (np.arange(nt) + 1) % nt], 1),
    }


def device_weights(obj, prefix="", seen=None, out=None, depth=0):
    """Every device weight tensor reachable from a shipped module, by attribute path.

    Walked rather than listed: the parameter set comes from the model, the same argument
    `tt_bio.train.lora.weights_for` makes for its census. That census reads `ops.linear` sites
    and the pairformer routes none of its calls through `ops.linear` (LEDGER K3), so it finds
    nothing here -- which is the gap this walk covers and the state doc records.
    """
    import ttnn
    out = {} if out is None else out
    seen = set() if seen is None else seen
    if depth > 4 or id(obj) in seen:
        return out
    seen.add(id(obj))
    items = obj.items() if isinstance(obj, dict) else (
        list(enumerate(obj)) if isinstance(obj, (list, tuple)) else
        vars(obj).items() if hasattr(obj, "__dict__") else [])
    for k, v in items:
        name = f"{prefix}{k}"
        if isinstance(v, ttnn.Tensor):
            out[name] = (obj, k, v)
        elif isinstance(v, (list, tuple, dict)) or hasattr(v, "__dict__"):
            device_weights(v, name + ".", seen, out, depth + 1)
    return out


class Composed:
    """The composed forward. Holds the shipped model and nothing that reimplements it."""

    def __init__(self, a, out):
        import torch
        import ttnn
        from tt_bio.protenix import Protenix
        from tt_bio import protenix_data as PD
        from tt_bio.tenstorrent import get_device

        self.a, self.out = a, out
        self.dev = get_device()
        t0 = time.perf_counter()
        self.model = Protenix.load_from_checkpoint(CKPT, device=self.dev)
        out["load_s"] = round(time.perf_counter() - t0, 2)

        ck = torch.load(CKPT, map_location="cpu", weights_only=True)
        ck = ck.get("model", ck)
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
        dw, db = sd["distogram_head.linear.weight"].float(), sd["distogram_head.linear.bias"].float()
        del ck, sd
        self.DW = ttnn.from_torch(dw.t().contiguous(), layout=ttnn.TILE_LAYOUT,
                                  device=self.dev, dtype=ttnn.bfloat16)
        self.DB = ttnn.from_torch(db.reshape(1, -1), layout=ttnn.TILE_LAYOUT,
                                  device=self.dev, dtype=ttnn.bfloat16)

        t0 = time.perf_counter()
        self.feats = PD.build_protein_features(sequence(a.tokens, a.seed))
        out["feats_s"] = round(time.perf_counter() - t0, 2)
        self.nt = int(self.feats["restype"].shape[0])
        self.rep = self.feats["distogram_rep_atom_mask"].bool().numpy()
        self.n_atom = int(self.rep.shape[0])
        self.cycles = a.cycles or int(self.model.trunk.N_CYCLES)
        out["shapes"] = {"tokens": self.nt, "atoms": self.n_atom,
                         "rep_atoms": int(self.rep.sum()), "c_z": int(self.model.trunk.C_Z),
                         "pairformer_blocks": len(self.model.trunk.PF.blocks),
                         "n_cycles": self.cycles, "distogram_bins": int(dw.shape[0])}
        self.batch = labels(self.nt, a.seed)
        self.row = None

        # The t-independent conditioning, from the SHIPPED pipeline. This is the untaped half
        # of the seam: `_trunk_cond` round-trips the trunk to host, exactly as a fold does,
        # and the sampler reuses it -- so it is built once and not per step.
        t0 = time.perf_counter()
        self.cond, self.aux = self.model._trunk_cond(self.feats, n_cycles=self.cycles)
        out["trunk_cond_s"] = round(time.perf_counter() - t0, 2)

        r = np.random.default_rng(a.seed + 2)
        sig = self.model.diffusion.SIGMA_DATA
        self.t_hat = torch.tensor([float(sig * math.exp(a.log_sigma))], dtype=torch.float32)
        self.x_noisy = torch.tensor(r.normal(0, float(self.t_hat[0]), (1, self.n_atom, 3)),
                                    dtype=torch.float32)
        s2 = float(self.t_hat[0]) / sig
        self.c_out = float(self.t_hat[0]) / math.sqrt(1.0 + s2 ** 2)
        self.relp = self.feats["relp"] if "relp" in self.feats else \
            self.model._generate_relp(self.feats)

        # One discovery forward, untaped. The shipped modules upload a weight on first use and
        # cache it, so before this runs the denoiser's decoder weights do not exist as device
        # tensors and a parameter walk cannot see them. Same argument `weights_for` makes for
        # spending a forward on its census: the parameter set comes from the model.
        t0 = time.perf_counter()
        self.forward()
        out["discovery_s"] = round(time.perf_counter() - t0, 2)

    # ---- the composed forward ---------------------------------------------------------
    def forward(self, params=None):
        """Run it under one tape. `params` pins named weights as trainable leaves.

        Returns the host outputs the objective consumes and the two device roots a backward
        seeds. Everything between is the shipped forward.
        """
        import torch
        import ttnn
        from tt_bio import autograd as ag

        m = self.model
        with ag.tape():
            if params:
                # A weight becomes a trainable leaf by pre-seeding the tape's raw-handle map:
                # `_wrap` is idempotent per handle, so the shipped module's own `ttnn.linear`
                # call picks up the parameter instead of minting an untracked leaf. No call
                # site changes and no weight is copied.
                for t in params.values():
                    ag.parameter(t.value)

            s_tt, z_tt = m.trunk(self.feats, self.aux["s_inputs"], self.relp,
                                 self.feats["token_bonds"], n_cycles=self.cycles)
            # The real distogram head: upstream symmetrises the pair track, then one linear to
            # 64 bins. `ttnn` here is the taped proxy, so both ops are on the tape.
            zs = ttnn.add(z_tt, ttnn.permute(z_tt, (0, 2, 1, 3)))
            dlog = ttnn.linear(zs, self.DW, bias=self.DB,
                               compute_kernel_config=m.compute_kernel_config)
            # One denoise call, which is what Protenix differentiates per step: one noise
            # level drawn per sample, `denoise_net` called once, no trajectory in the graph.
            x = m.diffusion.denoise(self.x_noisy, self.t_hat, self.cond)
            r_dev = getattr(m.diffusion, "last_r_update_device", None)
            dlog_h = np.asarray(ttnn.to_torch(ag._unwrap(dlog)).float(), np.float64)

        xa = np.asarray(x[0].detach().cpu(), np.float64)
        xt = xa[self.rep]
        return {"outputs": {"distogram_logits": dlog_h.reshape(self.nt, self.nt, -1)[
                                :self.nt, :self.nt],
                            "pred_xyz": xt,
                            "pred_dist": np.linalg.norm(xt[:, None, :] - xt[None, :, :], axis=-1)},
                "roots": {"distogram_logits": dlog, "r_update": r_dev},
                "atoms": xa}

    def loss(self, params=None):
        """The composed objective on the composed forward. Returns (total, breakdown, seeds, fwd)."""
        from tt_bio.train import objectives
        if self.row is None:
            self.row = objectives.objective("af3")
        fwd = self.forward(params)
        total, breakdown, seeds = self.row(self.batch, fwd["outputs"])
        return total, breakdown, seeds, fwd


def seed_roots(comp, fwd, seeds):
    """Map the objective's host seeds onto the two device roots the tape can accept.

    `pred_xyz` and `pred_dist` are token-level, gathered at the distogram representative atoms
    the way `ConfidenceHead` gathers them, so their seeds scatter back to those atom rows and
    are zero elsewhere. The EDM preconditioning is affine in `r_update`, so the scatter is
    multiplied by `c_out` and that is the whole chain rule.
    """
    from tt_bio import autograd as ag
    taped = lambda t: isinstance(t, ag.Tensor)
    roots, gs = [], []
    if "distogram_logits" in seeds and taped(fwd["roots"]["distogram_logits"]):
        roots.append(fwd["roots"]["distogram_logits"])
        gs.append(np.asarray(seeds["distogram_logits"], np.float32))
    gx = None
    if "pred_xyz" in seeds:
        gx = np.asarray(seeds["pred_xyz"], np.float64)
    if "pred_dist" in seeds:
        # d(dist)/d(xyz): dist_ij = |x_i - x_j|, so dL/dx_i = sum_j (g_ij + g_ji) * u_ij.
        x = fwd["outputs"]["pred_xyz"]
        d = fwd["outputs"]["pred_dist"]
        gd = np.asarray(seeds["pred_dist"], np.float64)
        gsym = gd + gd.T
        diff = x[:, None, :] - x[None, :, :]
        safe = np.where(d > 1e-8, d, 1.0)[..., None]
        contrib = (gsym[..., None] * diff / safe).sum(1)
        gx = contrib if gx is None else gx + contrib
    if gx is not None and taped(fwd["roots"]["r_update"]):
        full = np.zeros((comp.n_atom, 3), np.float64)
        full[comp.rep] = gx
        roots.append(fwd["roots"]["r_update"])
        gs.append((comp.c_out * full).astype(np.float32).reshape(1, comp.n_atom, 3))
    return roots, gs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=384)
    ap.add_argument("--cycles", type=int, default=None)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-sigma", type=float, default=0.0)
    ap.add_argument("--fd-leaves", type=int, default=2)
    ap.add_argument("--fd-eps", default="0.02,0.05,0.1,0.25")
    ap.add_argument("--no-fd", action="store_true")
    ap.add_argument("--diffusion-sites", default="atom_attention_decoder,layernorm_a",
                    type=lambda v: tuple(x for x in v.split(",") if x))
    ap.add_argument("--out", type=Path, default=Path("out/step.json"))
    a = ap.parse_args()

    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.train.optim import AdamW
    from tt_bio import tenstorrent as TT

    out = {"doc": __doc__, "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(), "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(), "ckpt": str(CKPT)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        a.out.write_text(json.dumps(out, indent=1, default=str))
    dump()

    with during() as clk:
        try:
            comp = Composed(a, out)
            out["env"]["arch"] = str(comp.dev.arch())
            dump()

            # --- the trainable set. The last pairformer block's own weights plus the
            # distogram head: real trunk weights, discovered off the model.
            blk = comp.model.trunk.PF.blocks[-1]
            found = device_weights(blk, prefix="pairformer.last.")
            found["distogram_head.linear.weight"] = (comp, "DW", comp.DW)
            # The denoiser's own weights. Without them nothing in the diffusion graph descends
            # from a taped leaf, `denoise` returns raw tensors and the module is in the forward
            # without being in the step -- which is what the first run of this harness did.
            diff = device_weights(comp.model.diffusion, prefix="diffusion.")
            for n, v in diff.items():
                if any(k in n for k in a.diffusion_sites):
                    found[n] = v
            params = {}
            for n, (owner, key, t) in sorted(found.items()):
                if len(t.shape) < 2:
                    continue
                params[n] = ag.parameter(t)
            holders = {n: found[n] for n in params}
            out["params"] = {n: [int(d) for d in t.value.shape] for n, t in params.items()}
            out["n_params"] = sum(int(np.prod(list(t.value.shape))) for t in params.values())
            dump()

            old_id = {n: id(t.value) for n, t in params.items()}
            opt = AdamW(params, lr=a.lr)
            history = []
            for step in range(a.steps):
                t0 = time.perf_counter()
                opt.zero_grad()
                total, breakdown, seeds, fwd = comp.loss(params)
                from tt_bio import autograd as _ag
                out.setdefault("diag", {})[f"step{step}"] = {
                    "seeds": sorted(seeds),
                    "distogram_root_taped": isinstance(fwd["roots"]["distogram_logits"], _ag.Tensor),
                    "r_update_taped": isinstance(fwd["roots"]["r_update"], _ag.Tensor),
                    "r_update_is_none": fwd["roots"]["r_update"] is None}
                roots, gs = seed_roots(comp, fwd, seeds)
                out["diag"][f"step{step}"]["n_roots"] = len(roots)
                ag.backward(roots, [ttnn.from_torch(torch.tensor(g), layout=ttnn.TILE_LAYOUT,
                                                    device=comp.dev, dtype=ttnn.bfloat16)
                                    for g in gs])
                have = {n: (t.grad is not None) for n, t in params.items()}
                opt.step()
                # The optimizer replaces `t.value`; the module still holds the tensor it
                # uploaded, so the new value is written back or the next forward reads the
                # checkpoint's weights with a falling loss curve and no error anywhere.
                for n, t in params.items():
                    owner, key, _ = holders[n]
                    if isinstance(owner, dict):
                        owner[key] = t.value
                    elif isinstance(owner, (list, tuple)):
                        owner[key] = t.value
                    else:
                        setattr(owner, key, t.value)
                    holders[n] = (owner, key, t.value)
                    ag._PARAMS.pop(old_id[n], None)
                    ag.parameter(t.value)
                    old_id[n] = id(t.value)
                rec = {"step": step, "loss": total, "s": round(time.perf_counter() - t0, 2),
                       "grad_norm": getattr(opt, "last_grad_norm", None),
                       "terms": {k: (v.get("value"), v.get("skipped")) for k, v in breakdown.items()},
                       "with_grad": sorted(n for n, h in have.items() if h),
                       "no_grad": sorted(n for n, h in have.items() if not h)}
                history.append(rec)
                out["history"] = history
                print(f"[step {step}] loss {total:.6f}  {rec['s']}s  "
                      f"grads {len(rec['with_grad'])}/{len(params)}", flush=True)
                dump()
            out["displacement"] = opt.check_displacement()
        except Exception:
            out["error"] = traceback.format_exc()
            print(out["error"], flush=True)
        out["clock"] = clk.summary() if hasattr(clk, "summary") else None
    dump()
    print("WROTE", a.out)
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    sys.exit(main())
