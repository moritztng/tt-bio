#!/usr/bin/env python3
"""of3t-frameself step 1: is the reference trunk gradient a*g_s + b*g_z?

The self-test settled H-A: the REAL `model.pairformer_stack`, differentiated in the capture's
own process on the captured pair, reads 0.7945281613194322 against `grads_f64_043.pt` -- the
same number `ref_grad.py`'s reconstructed loop reads to 1e-15. So the replay is exonerated and
the captured pair is insufficient.

The stack's pair path is closed: the z-only arm's dL/ds_in is exactly 0.0, so s_in cannot reach
z_out. The trunk gradient is therefore exactly linear in the two cotangents,
g(cot_s, cot_z) = g_s + g_z, and a cotangent that is wrong only by a scale on ONE side shows up
as the reference lying in the span of the two banked arms. That is what this measures.

  residual(a,b) ~ 1e-12   the true cotangent is (a*cot_s, b*cot_z) and the defect is two
                          scalars, one per output;
  residual large          the true cotangent differs in DIRECTION, not only in scale, and the
                          span of these two arms does not contain the reference.

Reference is `grads_f64_043.pt`, the full-model float64 backward on batch_step003 at
num_recycles 0 (R133: named on every line). The arms are INJECTED and are the numerator.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, socket, statistics
from pathlib import Path
import torch

PRE = "pairformer_stack."

def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def grads(p):
    d = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    return d["grads"] if isinstance(d, dict) and "grads" in d else d

def solve2(ss, sz, zz, sr, zr):
    det = ss * zz - sz * sz
    if det == 0.0 or ss <= 0.0 or zz <= 0.0:
        return None, None, None
    a = (zz * sr - sz * zr) / det
    b = (ss * zr - sz * sr) / det
    return a, b, det

class Acc:
    __slots__ = ("ss","sz","zz","sr","zr","rr","cc","ce","n","worst","worst_name")
    def __init__(self):
        self.ss=self.sz=self.zz=self.sr=self.zr=self.rr=self.cc=self.ce=0.0
        self.n=0; self.worst=-1.0; self.worst_name=None
    def add(self, s, z, r, c, name):
        self.n += 1
        self.ss += float(torch.dot(s,s)); self.zz += float(torch.dot(z,z))
        self.sz += float(torch.dot(s,z)); self.sr += float(torch.dot(s,r))
        self.zr += float(torch.dot(z,r)); self.rr += float(torch.dot(r,r))
        self.cc += float(torch.dot(c,c))
        d = s + z - c
        self.ce += float(torch.dot(d,d))
    def out(self):
        rr = self.rr
        a, b, det = solve2(self.ss, self.sz, self.zz, self.sr, self.zr)
        res2 = None
        if a is not None:
            res2 = max(rr - (a*self.sr + b*self.zr), 0.0)
        # a pinned to 1: the s side is already exact (ds_in is bit-identical), fit only the z scale
        b1 = (self.zr - self.sz) / self.zz if self.zz > 0 else None
        res_b1 = None
        if b1 is not None:
            res_b1 = max(rr - 2*(self.sr + b1*self.zr) + (self.ss + 2*b1*self.sz + b1*b1*self.zz), 0.0)
        # one shared scalar on the sum (the step-0 reading, recomputed from the same Gram)
        aa = self.ss + 2*self.sz + self.zz
        ar = self.sr + self.zr
        a0 = ar/aa if aa > 0 else None
        res_a0 = max(rr - ar*ar/aa, 0.0) if aa > 0 else None
        f = lambda x: math.sqrt(x/rr) if (x is not None and rr > 0) else None
        return {
            "n_tensors": self.n,
            "ref_squared_norm": rr,
            "s_arm_squared_norm": self.ss, "z_arm_squared_norm": self.zz,
            "ctrl_squared_norm": self.cc,
            "sum_identity_rel_l2_gs_plus_gz_vs_gctrl": math.sqrt(self.ce/self.cc) if self.cc>0 else None,
            "cos_gs_gz": self.sz/math.sqrt(self.ss*self.zz) if self.ss>0 and self.zz>0 else None,
            "one_shared_scalar": {"a": a0, "one_over_a": (1/a0 if a0 else None),
                                  "residual_frac_of_ref": f(res_a0)},
            "two_scalars": {"a_s": a, "b_z": b, "one_over_a_s": (1/a if a else None),
                            "one_over_b_z": (1/b if b else None),
                            "residual_frac_of_ref": f(res2)},
            "a_pinned_to_1": {"b_z": b1, "one_over_b_z": (1/b1 if b1 else None),
                              "residual_frac_of_ref": f(res_b1)},
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--sonly", required=True, type=Path)
    ap.add_argument("--zonly", required=True, type=Path)
    ap.add_argument("--ctrl", required=True, type=Path)
    ap.add_argument("--blocks", type=int, default=48)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    ref = grads(a.ref); gs = grads(a.sonly); gz = grads(a.zonly); gc = grads(a.ctrl)
    names = sorted(k for k in ref if k.startswith(PRE) and ref[k] is not None)
    print(f"{len(names)} trunk tensors", flush=True)

    pooled = Acc()
    per_block = {str(i): Acc() for i in range(a.blocks)}
    # path classification, measured not asserted: which cotangent actually reaches this tensor
    cls = {"s_only": Acc(), "z_only": Acc(), "mixed": Acc()}
    cls_names = {"s_only": [], "z_only": [], "mixed": []}
    percls_worst = {}
    for n in names:
        r = ref[n].to(torch.float64).reshape(-1)
        s = gs[n].to(torch.float64).reshape(-1)
        z = gz[n].to(torch.float64).reshape(-1)
        c = gc[n].to(torch.float64).reshape(-1)
        sn = float(torch.dot(s,s)); zn = float(torch.dot(z,z))
        k = "s_only" if zn == 0.0 else ("z_only" if sn == 0.0 else "mixed")
        cls[k].add(s,z,r,c,n); cls_names[k].append(n)
        pooled.add(s,z,r,c,n)
        parts = n.split(".")
        if len(parts) > 2 and parts[1] == "blocks":
            per_block[parts[2]].add(s,z,r,c,n)
    out = {
        "what": __doc__.strip().splitlines()[0],
        "host": socket.gethostname(), "row": "of3t-frameself", "defect": "D242",
        "device_involved": False,
        "why_no_aiclk": "CPU only, no Tenstorrent device is opened on either host",
        "reference": {"path": str(a.ref), "sha256": sha256_file(a.ref),
                      "is": "the full-model float64 backward on batch_step003, num_recycles 0",
                      "injected": False},
        "arms": {k: {"path": str(v), "sha256": sha256_file(v), "bytes": os.path.getsize(v),
                     "injected": True}
                 for k, v in (("s_only", a.sonly), ("z_only", a.zonly), ("ctrl_both", a.ctrl))},
        "pooled": pooled.out(),
        "by_path_class": {k: dict(v.out(), n_names=len(cls_names[k])) for k, v in cls.items()},
        "path_class_examples": {k: v[:4] for k, v in cls_names.items()},
        "per_block": {k: v.out() for k, v in per_block.items() if v.n},
    }
    pb = out["per_block"]
    def summ(xs):
        xs = [x for x in xs if x is not None]
        return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs),
                "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
                "min": min(xs), "max": max(xs)}
    out["per_block_summary"] = {
        "two_scalars_a_s": summ([v["two_scalars"]["a_s"] for v in pb.values()]),
        "two_scalars_b_z": summ([v["two_scalars"]["b_z"] for v in pb.values()]),
        "two_scalars_residual": summ([v["two_scalars"]["residual_frac_of_ref"] for v in pb.values()]),
        "one_shared_scalar_residual": summ([v["one_shared_scalar"]["residual_frac_of_ref"] for v in pb.values()]),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["pooled"], indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ("n_tensors","one_shared_scalar","two_scalars","a_pinned_to_1")}
                      for k, v in out["by_path_class"].items()}, indent=1))
    print("wrote " + str(a.out))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
