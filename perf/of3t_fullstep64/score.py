#!/usr/bin/env python3
"""of3t-fullstep64: ON, OFF and upstream-bf16 against the float64 full step, on the bijection.

    score.py --f64 grads_f64.pt --bf16 grads_bf16.pt --bijection BIJECTION.json \
             --arm ON1=grad_ON1.pt --arm ON2=grad_ON2.pt --arm OFF=grad_OFF.pt --out SCORE.json

Our arms are walk-path dumps; each is carried into upstream names through the bijection (by-value
bands sliced and un-transposed, derived APB leaves inverted), then scored in float64 exactly as
PREREGISTERED.md fixes:

  global rel  ||g_a - g_64|| / ||g_64|| over the scored set concatenated into ONE vector, with
              r, cos and the A43 identity residual |rel^2 - (1 + r^2 - 2 r cos)|;
  per head    trunk / distogram / confidence / diffusion (upstream prefix);
  bf16 mass   F64 squared mass in tensors where the arm's per-tensor rel <= bf16's;
  per section top-level module (aux_heads split by head), to localise a defect common to arms;
  unread      F64 squared mass the bijection does not score, beside every figure.

Device dumps are flat (`trainfwd_run.grad_snapshot` saves `.reshape(-1)`), so each one is put back
into its walked parameter's shape from DEVICE_SHAPES.json before any band is sliced; a flat vector
narrowed on axis 0 or transposed with `.t()` gives a wrong tensor without raising.

A tensor is in the reference set when its float64 gradient exists and is non-zero. An upstream
tensor reached by more than one device tensor gets the SUM of their gradients (the same weight
uploaded twice is one parameter used twice); the count is reported.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bijmap import apb_inverse  # noqa: E402

PREFIX = (("aux_heads.distogram.", "distogram"), ("aux_heads.", "confidence"),
          ("sample_diffusion.", "diffusion"), ("diffusion_module.", "diffusion"), ("", "trunk"))


def head_of(n):
    return next(h for p, h in PREFIX if n.startswith(p))


def section_of(n):
    """Top-level module; aux_heads split by head."""
    parts = n.split(".")
    return ".".join(parts[:2]) if parts[0] == "aux_heads" else parts[0]


def load_device(path, shapes):
    """A flat device dump, each tensor restored to its parameter's shape (row-major)."""
    out = {}
    for k, v in torch.load(path, weights_only=False).items():
        if v is None:
            continue
        shape = shapes[k]
        if v.numel() != torch.Size(shape).numel():
            raise SystemExit(f"{path}: {k} has {v.numel()} elements, parameter shape {shape}")
        out[k] = v.reshape(shape)
    return out


def to_upstream(dev_grads, bij):
    out, multi = {}, 0
    for key, pls in bij["placements"].items():
        parts = []
        for pl in pls:
            g = dev_grads.get(pl["device_path"])
            if g is None:
                continue
            band = g.to(torch.float64).narrow(pl["axis"], pl["start"], pl["length"])
            if pl["layout"].endswith("transposed"):
                band = band.T
            parts.append(band.contiguous())
        if parts:
            multi += len(parts) > 1
            out[key] = sum(parts[1:], parts[0])
    for key, dsc in bij["derived"].items():
        g = dev_grads.get(dsc["device_path"])
        if g is None:
            continue
        g = g.to(torch.float64)
        if dsc["rule"] == "apb_inverse":
            out[key] = apb_inverse(g, dsc["leaf"], dsc["H"], dsc["d"], dsc["D"], dsc["c_s"])
        else:
            out[key] = (g * dsc["scale"]).t().contiguous()
    return out, multi


def triple(pairs):
    """Concatenated definition: one vector pair, so the identity holds exactly."""
    rr = aa = ra = dd = 0.0
    for r, a in pairs:
        rr += float((r * r).sum())
        aa += float((a * a).sum())
        ra += float((r * a).sum())
        dd += float(((a - r) ** 2).sum())
    if rr == 0:
        return None
    rel, r_, cos = (dd / rr) ** 0.5, (aa / rr) ** 0.5, (ra / (aa * rr) ** 0.5 if aa else 0.0)
    return {"rel": rel, "r": r_, "cos": cos,
            "identity_residual": abs(rel ** 2 - (1 + r_ ** 2 - 2 * r_ * cos)),
            "ref_sq": rr, "arm_sq": aa}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f64", required=True)
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--bijection", required=True)
    ap.add_argument("--shapes", default=str(HERE / "DEVICE_SHAPES.json"))
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ref = {k: v.to(torch.float64) for k, v in torch.load(a.f64, weights_only=False).items()
           if v is not None and bool(v.any())}
    bf = {k: v.to(torch.float64) for k, v in torch.load(a.bf16, weights_only=False).items()
          if v is not None}
    bij = json.loads(Path(a.bijection).read_text())
    shapes = json.loads(Path(a.shapes).read_text())["shapes"]
    bad = [(pl["device_path"], pl["device_shape"]) for pls in bij["placements"].values()
           for pl in pls if list(pl["device_shape"]) != shapes[pl["device_path"]]]
    if bad:
        raise SystemExit(f"bijection device_shape disagrees with {a.shapes}: {bad[:5]}")
    mass = {k: float((v * v).sum()) for k, v in ref.items()}
    total = sum(mass.values())
    heads = sorted({head_of(k) for k in ref} | {"diffusion"})
    head_total = {h: sum(m for k, m in mass.items() if head_of(k) == h) for h in heads}

    arms = {}
    dev_raw = {}
    for spec in a.arm:
        name, path = spec.split("=", 1)
        dev_raw[name] = load_device(path, shapes)
        arms[name], multi = to_upstream(dev_raw[name], bij)
        arms[name + "__multi"] = multi
    names = [s.split("=", 1)[0] for s in a.arm]
    placed = set(bij["placements"]) | set(bij["derived"])
    scored = set(ref) & placed
    carried_nothing = {}
    for n in names:   # placed but no gradient on the device tensor: the arm carried zero
        miss = sorted(scored - set(arms[n]))
        carried_nothing[n] = {"n": len(miss), "mass_fraction": sum(mass[k] for k in miss) / total,
                              "sample": miss[:10]}
        for k in miss:
            arms[n][k] = torch.zeros_like(ref[k])
    unread = sorted(set(ref) - scored)
    unread_mass = sum(mass[k] for k in unread)

    def block(arm, keys):
        return triple([(ref[k], arm[k]) for k in keys])

    rep = {"reference": {"file": a.f64, "n_nonzero": len(ref), "squared_norm": total,
                         "by_head_squared_norm": head_total},
           "scored": {"n": len(scored), "unread_n": len(unread),
                      "unread_mass_fraction": unread_mass / total,
                      "unread_by_head_mass_fraction": {
                          h: (sum(mass[k] for k in unread if head_of(k) == h) / head_total[h]
                              if head_total[h] else None) for h in heads},
                      "unread_worst": sorted(unread, key=lambda k: -mass[k])[:25]},
           "arms": {}}
    for k in set(ref) - set(bf):
        bf[k] = torch.zeros_like(ref[k])
    bf_rel = {k: (float(((bf[k] - ref[k]) ** 2).sum()) / mass[k]) ** 0.5 if k in bf else 1.0
              for k in ref}
    for name, arm in [(n, arms[n]) for n in names] + [("BF16", bf)]:
        per = {k: (float(((arm[k] - ref[k]) ** 2).sum()) / mass[k]) ** 0.5 for k in scored}
        e = {"global": block(arm, sorted(scored)),
             "by_head": {h: block(arm, sorted(k for k in scored if head_of(k) == h))
                         for h in heads},
             "by_section": {sec: block(arm, sorted(k for k in scored if section_of(k) == sec))
                            for sec in sorted({section_of(k) for k in scored})},
             "mass_at_or_better_than_bf16": (sum(mass[k] for k in scored if per[k] <= bf_rel[k])
                                             / sum(mass[k] for k in scored)),
             "worst": [{"tensor": k, "rel": per[k], "mass_fraction": mass[k] / total}
                       for k in sorted(scored, key=lambda k: -per[k])[:10]],
             "worst_by_error_mass": [
                 {"tensor": k, "rel": per[k], "error_sq_share": per[k] ** 2 * mass[k]}
                 for k in sorted(scored, key=lambda k: -(per[k] ** 2 * mass[k]))[:10]]}
        if name != "BF16":
            e["multi_placed"] = arms[name + "__multi"]
            e["placed_but_carried_nothing"] = carried_nothing[name]
        rep["arms"][name] = e
    if "BF16" in rep["arms"]:
        rep["arms"]["BF16"]["global_all_reference"] = block(bf, sorted(set(ref) & set(bf)))

    # A/A, in the walk space the arms were written in.
    if {"ON1", "ON2"} <= set(dev_raw):
        a1, a2 = dev_raw["ON1"], dev_raw["ON2"]
        same = sum(1 for k in a1 if k in a2 and torch.equal(a1[k], a2[k]))
        rep["aa_ON1_vs_ON2"] = {"n": len(a1), "bit_identical": same,
                                "keysets_equal": set(a1) == set(a2)}
    on = "ON1" if "ON1" in rep["arms"] else None
    if on and "OFF" in rep["arms"]:
        cmp = {}
        for scope in ["global"] + heads:
            x = rep["arms"][on]["global"] if scope == "global" else rep["arms"][on]["by_head"][scope]
            y = rep["arms"]["OFF"]["global"] if scope == "global" else rep["arms"]["OFF"]["by_head"][scope]
            if not x or not y:
                cmp[scope] = None
                continue
            tie = abs(x["rel"] - y["rel"]) <= 0.05 * min(x["rel"], y["rel"])
            cmp[scope] = {"rel_ON": x["rel"], "rel_OFF": y["rel"],
                          "nearer": "TIE" if tie else ("ON" if x["rel"] < y["rel"] else "OFF")}
        rep["on_vs_off"] = cmp
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({"scored": rep["scored"]["n"], "unread_mass": rep["scored"]["unread_mass_fraction"],
                      "global": {n: rep["arms"][n]["global"]["rel"] for n in rep["arms"]},
                      "on_vs_off": rep.get("on_vs_off"), "aa": rep.get("aa_ON1_vs_ON2")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
