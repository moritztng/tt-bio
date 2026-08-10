#!/usr/bin/env python3
"""X3 deliverables 1-3: live-fold stage walls on qb1 card 2, alternating arms in ONE session.

Arms alternate fold by fold (base, arm, base, arm, ...) so a drift in host load hits both equally.
`trunk_msa` and `trunk_template` are timed with `ttnn.synchronize_device` on both sides. The OPM
untilize is counted and timed in the live fold by wrapping `ttnn.to_layout` only for the calls that
untilize a 9536-wide tensor, so nothing else in the fold is perturbed.

  --arm c3      the PWA per-head weight-slice cache  (stage: trunk_msa)
  --arm c5      the template z-projection hoist      (stage: trunk_template)
  --arm both    both
"""
from __future__ import annotations
import argparse, importlib.util, json, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch, ttnn
import tt_bio.protenix as P
import tt_bio.tenstorrent as T

STAGE = {}          # name -> list of seconds, one per call
OPM = {"n": 0, "us": []}
CAPTURE = {}        # (arm, stage) -> torch tensor of the first call of the last fold
ARM = ["base"]
GRAB = [False]


def _sync():
    ttnn.synchronize_device(T._device or T.get_device())


def timed_stage(name, fn):
    def wrapper(self, *a, **k):
        _sync()
        t0 = time.perf_counter()
        out = fn(self, *a, **k)
        _sync()
        STAGE.setdefault(name, []).append(time.perf_counter() - t0)
        if GRAB[0] and (ARM[0], name) not in CAPTURE:
            CAPTURE[(ARM[0], name)] = ttnn.to_torch(out).clone()
        return out
    return wrapper


# ---- the OPM untilize, counted and timed inside the live fold ------------------------------
_orig_to_layout = ttnn.to_layout


def to_layout_probe(tensor, layout, *a, **k):
    wide = (layout == ttnn.ROW_MAJOR_LAYOUT and len(tensor.shape) == 2
            and tensor.shape[-1] >= 4096 and tensor.layout == ttnn.TILE_LAYOUT)
    if not wide:
        return _orig_to_layout(tensor, layout, *a, **k)
    _sync()
    t0 = time.perf_counter()
    out = _orig_to_layout(tensor, layout, *a, **k)
    _sync()
    OPM["n"] += 1
    OPM["us"].append((time.perf_counter() - t0) * 1e6)
    OPM.setdefault("shape", str(tuple(tensor.shape)))
    return out


# ---- baseline arms: undo the production change in-session ----------------------------------
_prod_pwa_call = T.PairWeightedAveraging.__call__
_prod_template = P.Trunk._template


def base_pwa_call(self, m, z, attn_mask=None):
    """Re-cut all 32 per-head weight views on every call, which is what shipped before C3."""
    hd, nh = self.head_dim, self.n_heads
    saved = (self.z_head, self.m_head, self.g_head, self.o_head)
    self.z_head = [self.z_weight[:, i:i + 1] for i in range(nh)]
    self.m_head = [self.m_weight[:, i * hd:(i + 1) * hd] for i in range(nh)]
    self.g_head = [self.g_weight[:, i * hd:(i + 1) * hd] for i in range(nh)]
    self.o_head = [self.o_weight[i * hd:(i + 1) * hd, :] for i in range(nh)]
    fresh = (self.z_head, self.m_head, self.g_head, self.o_head)
    try:
        return _prod_pwa_call(self, m, z, attn_mask)
    finally:
        for lst in fresh:
            for t in lst:
                ttnn.deallocate(t)
        self.z_head, self.m_head, self.g_head, self.o_head = saved


def base_template(self, z3, tpl_a, N, nt):
    """The pre-C5 body: the z projection re-evaluated inside the template loop."""
    zn = self._ln(z3, "template_embedder.layernorm_z.weight", "template_embedder.layernorm_z.bias")
    u = None
    for t in range(nt):
        v = ttnn.add(tpl_a[t], self._lin(zn, "template_embedder.linear_no_bias_z.weight"))
        for pl in self.TPL:
            v = pl(None, v)[1]
        v = self._ln(v, "template_embedder.layernorm_v.weight", "template_embedder.layernorm_v.bias")
        u = v if u is None else ttnn.add(u, v)
    u = ttnn.multiply(u, 1.0 / (1e-7 + nt))
    return self._lin(ttnn.relu(u), "template_embedder.linear_no_bias_u.weight")


def set_arm(arm, which):
    ARM[0] = arm
    on = arm == "arm"
    T.PairWeightedAveraging.__call__ = (
        _prod_pwa_call if (on and which in ("c3", "both")) else base_pwa_call)
    P.Trunk._template = timed_stage(
        "trunk_template", _prod_template if (on and which in ("c5", "both")) else base_template)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["c3", "c5", "both"])
    ap.add_argument("--folds", type=int, default=3, help="folds PER ARM after the cold pair")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    P.Trunk._msa = timed_stage("trunk_msa", P.Trunk._msa)
    ttnn.to_layout = to_layout_probe

    spec = importlib.util.spec_from_file_location(
        "tt_baseline", REPO / "scripts" / "gpu_vs_tt" / "tt_baseline.py")
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    msa_dir = Path("~/.cache/tt-bio-gpu-vs-tt/msa").expanduser()
    one_fold, meta, _state = tb.build_fold(
        "protenix-v2", msa_dir, REPO / "examples/prot300.yaml",
        REPO / "scripts/gpu_vs_tt/fixtures/prot300.a3m")
    print(f"loaded, n_msa={meta['n_msa']}", flush=True)

    out = {"card": "qb1 card 2", "which": args.which, "n_msa": meta["n_msa"], "arms": {}}
    # cold pair, discarded
    for arm in ("base", "arm"):
        set_arm(arm, args.which)
        STAGE.clear(); OPM["n"] = 0; OPM["us"].clear()
        one_fold()
    print("cold pair done", flush=True)

    per = {"base": {}, "arm": {}}
    for rep in range(args.folds):
        for arm in ("base", "arm"):
            set_arm(arm, args.which)
            STAGE.clear(); OPM["n"] = 0; OPM["us"].clear()
            GRAB[0] = (rep == args.folds - 1)
            wall, _m = one_fold()
            GRAB[0] = False
            for k, v in STAGE.items():
                per[arm].setdefault(k, []).append(sum(v))
                per[arm].setdefault(k + "_calls", []).append(len(v))
            per[arm].setdefault("fold_s", []).append(wall)
            per[arm].setdefault("opm_untilize_calls", []).append(OPM["n"])
            if OPM["us"]:
                per[arm].setdefault("opm_untilize_us", []).append(st.median(OPM["us"]))
            print(f"  rep {rep} {arm:4s} fold {wall:7.3f}s  "
                  + "  ".join(f"{k} {sum(v)*1000:8.2f}ms x{len(v)}" for k, v in STAGE.items())
                  + f"   opm_untilize n={OPM['n']}"
                  + (f" med {st.median(OPM['us']):.1f}us" if OPM["us"] else ""), flush=True)

    for arm in ("base", "arm"):
        out["arms"][arm] = {k: (round(st.median(v), 4) if isinstance(v[0], float) else v)
                            for k, v in per[arm].items()}
        out["arms"][arm + "_raw"] = per[arm]
    out["opm_shape"] = OPM.get("shape")

    # parity: the stage output, base vs arm, at the same fold index
    par = {}
    for stage in ("trunk_msa", "trunk_template"):
        b, a = CAPTURE.get(("base", stage)), CAPTURE.get(("arm", stage))
        if b is None or a is None:
            continue
        eq = bool(torch.equal(b, a))
        d = (b.float() - a.float()).abs()
        par[stage] = {"torch_equal": eq, "max_abs": float(d.max()),
                      "mean_abs": float(d.mean()), "shape": list(b.shape)}
        print(f"  PARITY {stage}: torch.equal={eq} max_abs={float(d.max()):.6g}", flush=True)
    out["parity"] = par
    args.out.write_text(json.dumps(out, indent=1, default=str))

    for stage in ("trunk_msa", "trunk_template"):
        if stage in out["arms"]["base"]:
            bs, as_ = out["arms"]["base"][stage], out["arms"]["arm"][stage]
            print(f"{stage:16s} base {bs*1000:8.3f} ms/stage -> arm {as_*1000:8.3f}  "
                  f"saved {(bs-as_)*1000:7.3f} ms/stage = {(bs-as_)*10*1000:7.2f} ms/fold "
                  f"({100*(bs-as_)/bs:5.2f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
