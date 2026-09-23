#!/usr/bin/env python3
"""of3t-cotcoh pass 2: the D240 census, two break levers, the family B capture fix, and the
trunk scored in process so no 1.3 GB gradient dump lands on a disk with 1.4 GB free.

Four things, and no arithmetic changed unless a lever asks for it.

  1. CAPTURE, fixed. `capcot.py` kept the LAST nonzero fire at the pair-track site. That site
     fires the taped LayerNorm backward 12 times per block with 2 of them on the real 56x56
     block, so the kept tensor was one contribution of two -- at block 47 its norm was 0.4717x
     the reference's at cosine 0.9730. dW is linear in the cotangent and both fires share the
     same xhat, so the effective cotangent is their SUM and that is what this captures.

  2. D240 CENSUS. `tt_bio/autograd.py::Tensor.add_grad` stores the first contribution in the
     closure's own dtype and promotes to fp32 only when a second arrives, so cotangent dtype is
     keyed on graph fan-out. This counts, per block and per tensor: how many contributions
     landed, the dtype the first arrived in, the dtype `self.grad` ended in, and the dtype of
     the VALUE -- because `autograd.backward` casts the gradient to `t.value.dtype` before it
     calls the closure, which is what decides the dtype the closure actually consumes.

  3. TWO LEVERS.
       promote_first  typecast the FIRST contribution to fp32 as well, which is D240's own
                      one-line fix.
       cot_fp32       drop the down-cast to `t.value.dtype` at the consumption point and hand
                      the closure the accumulated gradient in the dtype it was accumulated in.
     Both are confined to the tape. An inference fold installs no tape, so neither can reach
     inference by construction and neither needs a flag.

  4. TRUNK, scored in process against the float64 reference over the same 2,736 tensors
     `model_scope.py` calls the pairformer section. The `none` arm must reproduce
     0.9349175217825587, which is the composition control: a lever's reading means nothing
     until the arm that changes nothing reproduces the banked number.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "perf/of3t_trunkg043"))
print("SYS_PATH resolved: " + repr(sys.path[:2]), flush=True)

FAM_A = "pre_norm_s_weight"
FAM_B = "transition_z.norm_weight"
BLK = re.compile(r"blocks\.(\d+)\.")
SEC = "pairformer_stack.blocks."
F64REF = "/home/ttuser/of3t-campaign-refs/bundle_min_043/grads_f64_043.pt"
BANKED_TRUNK_VS_F64 = 0.9349175217825587      # of3t-modelframe, MODEL_FRAMEMATCHED, trunk section
BANKED_UPSTREAM_BF16_VS_F64 = 0.3147698293887927


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lever", default="none",
                    choices=("none", "promote_first", "cot_fp32"))
    ap.add_argument("--cot-out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--real-rows", type=int, default=56)
    a, rest = ap.parse_known_args()
    passthrough = [x for x in rest if x != "--"]
    t0 = time.perf_counter()
    R = a.real_rows

    import torch
    import ttnn
    from tt_bio import autograd as ag
    import tt_bio.taped_ttnn as tt
    import tt_bio.tenstorrent as T

    # ---------------- lever 1: promote the FIRST contribution too -------------------------
    LEV = {"lever": a.lever, "promote_first_fired": 0, "cot_fp32_kept": 0,
           "cot_fp32_would_have_downcast": 0}
    _real_add = ag.Tensor.add_grad
    GRADLOG = {}          # id(tensor) -> [n_contributions, first_dtype, final_dtype, value_dtype]

    def add_grad(self, grad):
        if self.requires_grad:
            k = id(self)
            e = GRADLOG.get(k)
            if e is None:
                GRADLOG[k] = [1, str(grad.dtype), None, str(self.value.dtype)]
                if a.lever == "promote_first" and grad.dtype != ttnn.float32:
                    grad = ttnn.typecast(grad, ttnn.float32)
                    LEV["promote_first_fired"] += 1
            else:
                e[0] += 1
        out = _real_add(self, grad)
        if self.requires_grad and self.grad is not None:
            GRADLOG[id(self)][2] = str(self.grad.dtype)
        return out

    ag.Tensor.add_grad = add_grad

    # ---------------- lever 2: do not down-cast the cotangent at consumption ---------------
    if a.lever == "cot_fp32":
        _rt, _un, _re = ag._reverse_topo, ag._unshard, ag._retire

        def backward(roots, seeds=None):
            if isinstance(roots, ag.Tensor):
                roots = [roots]
                seeds = [seeds] if not isinstance(seeds, (list, tuple)) else seeds
            roots = list(roots)
            if seeds is None:
                seeds = [None] * len(roots)
            seeds = list(seeds)
            order = _rt(roots)
            keep = {id(r) for r in roots}
            for r, sd in zip(roots, seeds):
                g = ttnn.ones_like(r.value) if sd is None else sd
                r.grad = g if r.grad is None else ttnn.add(r.grad, g)
            for t in order:
                if t.node is not None:
                    g = t.grad
                    if g is None:
                        continue
                    if g.dtype != t.value.dtype:
                        # THE LEVER. The shipped line is
                        #   g = ttnn.typecast(g, t.value.dtype)
                        # which throws an fp32 accumulated cotangent back to the forward
                        # activation's bf16 at every consumption point. Keep it instead.
                        LEV["cot_fp32_would_have_downcast"] += 1
                        LEV["cot_fp32_kept"] += 1
                    t.node.fn(_un(g))
                    if id(t) not in keep:
                        _re(t)
        ag.backward = backward
        try:
            import tt_bio.train as _tr            # some callers bind the name at import
            if hasattr(_tr, "backward"):
                _tr.backward = backward
        except Exception:
            pass

    # ---------------- keep the harness's gradient dump off the disk, and score it ----------
    OUTPATH = None
    for i, x in enumerate(passthrough):
        if x == "--out":
            OUTPATH = passthrough[i + 1]
    _real_save = torch.save
    SCORE = {}

    def _sq(t):
        return float(torch.linalg.vector_norm(t.to(torch.float64))) ** 2

    def _score(grads):
        ref = torch.load(F64REF, map_location="cpu", weights_only=False)
        if isinstance(ref, dict) and "grads" in ref:
            ref = ref["grads"]
        per, blk = {}, {}
        errs = refs = 0.0
        n = miss = 0
        for k, v in ref.items():
            if not k.startswith(SEC) or v is None:
                continue
            o = grads.get(k)
            if o is None:
                miss += 1
                continue
            r = v.to(torch.float64)
            d = o.to(torch.float64) - r
            e2, r2 = _sq(d), _sq(r)
            per[k] = {"err_sq": e2, "ref_sq": r2, "our_sq": _sq(o)}
            errs += e2
            refs += r2
            n += 1
            b = int(k[len(SEC):].split(".")[0])
            g = blk.setdefault(b, [0.0, 0.0, 0])
            g[0] += e2
            g[1] += r2
            g[2] += 1
        rel = (errs / refs) ** 0.5 if refs > 0 else None
        return {"n_tensors": n, "absent": miss,
                "trunk_mass_weighted_rel_l2_vs_float64": rel,
                "banked_none_arm_value": BANKED_TRUNK_VS_F64,
                "reproduces_banked": (rel == BANKED_TRUNK_VS_F64) if rel is not None else None,
                "rel_difference_vs_banked": (abs(rel - BANKED_TRUNK_VS_F64) /
                                             BANKED_TRUNK_VS_F64) if rel else None,
                "multiple_over_upstream_bf16": (rel / BANKED_UPSTREAM_BF16_VS_F64
                                                if rel else None),
                "upstream_bf16_vs_float64": BANKED_UPSTREAM_BF16_VS_F64,
                "upstream_bf16_provenance": "of3t-modelframe MODEL_FRAMEMATCHED_composed3660"
                                            "_n384.json trunk section, a constant across this "
                                            "row's arms",
                "by_block": {str(b): {"err_sq": v[0], "ref_sq": v[1], "n": v[2],
                                      "rel_l2_vs_float64": (v[0] / v[1]) ** 0.5 if v[1] else None}
                             for b, v in sorted(blk.items())},
                "per_tensor": per}

    def _norms(o):
        if torch.is_tensor(o):
            return {"__sqnorm__": _sq(o), "shape": list(o.shape), "dtype": str(o.dtype)}
        if isinstance(o, dict):
            return {k: _norms(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(_norms(v) for v in o)
        return o

    def _save(obj, f, *ar, **kw):
        if isinstance(f, str) and OUTPATH is not None and \
                os.path.abspath(f) == os.path.abspath(OUTPATH):
            try:
                SCORE.update(_score(obj["grads"]))
                print("TRUNK vs float64: %.16f (banked none-arm %.16f)"
                      % (SCORE["trunk_mass_weighted_rel_l2_vs_float64"],
                         BANKED_TRUNK_VS_F64), flush=True)
            except Exception as e:
                SCORE["ERROR"] = f"{type(e).__name__}: {e}"
                print("SCORE ERROR " + SCORE["ERROR"], flush=True)
            return _real_save({"REDIRECTED": "perf/of3t_cotcoh/arm2.py",
                               "sqnorms": _norms(obj)}, f, *ar, **kw)
        return _real_save(obj, f, *ar, **kw)

    torch.save = _save

    # ---------------- the module handle, so a gamma can be named --------------------------
    MOD = []
    _RealPF = T.Pairformer

    class _PF(_RealPF):
        def __init__(self, *ar, **kw):
            super().__init__(*ar, **kw)
            MOD.append(self)

    T.Pairformer = _PF
    WPATH = {}

    def _name_of(v):
        if not WPATH and MOD:
            from tt_bio.tenstorrent import device_weights
            for p, w in device_weights(MOD[0]).items():
                try:
                    WPATH[w.buffer_address()] = p
                except Exception:
                    pass
        try:
            return WPATH.get(v.buffer_address())
        except Exception:
            return None

    CAP, FIRES = {}, {}
    _shipped_verb = ag._taped_layer_norm

    def _mask(fam, t, C):
        t = t.reshape(-1, C)
        if fam == "A":
            return t[:R].contiguous()
        rows = t.shape[0] // 384
        if rows < R:
            return None
        return t.reshape(rows, 384, C)[:R, :R].reshape(-1, C).contiguous()

    def _verb(shipped, args, kwargs):
        argl = list(args) + [None] * (3 - len(args))
        xw = ag._wrap(argl[0])
        gw = ag._wrap(kwargs.get("weight", argl[1]))
        out = _shipped_verb(shipped, args, kwargs)
        if out is None or getattr(out, "node", None) is None or gw is None:
            return out
        path = _name_of(gw.value)
        fam = None
        if path:
            if FAM_A in path:
                fam = "A"
            elif FAM_B in path:
                fam = "B"
        m = BLK.search(path or "")
        blk = int(m.group(1)) if m else -1
        orig = out.node.fn

        def fn(g, orig=orig, xw=xw, gw=gw, fam=fam, blk=blk, path=path):
            if fam is None or blk < 0:
                return orig(g)
            key = (fam, blk)
            C = int(gw.value.shape[-1])
            gt = ttnn.to_torch(g)
            full = float(torch.linalg.vector_norm(gt.to(torch.float64))) ** 2
            rec = FIRES.setdefault(key, {"fires": 0, "nonzero": 0, "chunk_sqnorms": []})
            rec["fires"] += 1
            rec["chunk_sqnorms"].append(full)
            gm = _mask(fam, gt, C)
            if gm is not None and float(torch.linalg.vector_norm(gm.to(torch.float64))) > 0.0:
                rec["nonzero"] += 1
                xt = ttnn.to_torch(xw.value)
                xm = _mask(fam, xt, C)
                e = CAP.get(key)
                if e is None:
                    CAP[key] = {"g": gm.to(torch.float64).clone(),
                                "x": xm.to(torch.float64).clone(),
                                "g_dtype": str(g.dtype), "x_dtype": str(xw.value.dtype),
                                "gamma_path": path, "n_summed": 1,
                                "x_max_abs_diff_between_fires": 0.0,
                                "outside_mask_sqnorm": full - float(
                                    torch.linalg.vector_norm(gm.to(torch.float64))) ** 2,
                                "per_fire_masked_sqnorm": [
                                    float(torch.linalg.vector_norm(gm.to(torch.float64))) ** 2]}
                else:
                    # dW is linear in g and both fires share the same xhat, so the effective
                    # cotangent is the SUM. The x check is what says they share it.
                    e["x_max_abs_diff_between_fires"] = max(
                        e["x_max_abs_diff_between_fires"],
                        float((e["x"] - xm.to(torch.float64)).abs().max()))
                    e["g"] = e["g"] + gm.to(torch.float64)
                    e["n_summed"] += 1
                    e["per_fire_masked_sqnorm"].append(
                        float(torch.linalg.vector_norm(gm.to(torch.float64))) ** 2)
            return orig(g)

        out.node.fn = fn
        return out

    ag._TAPED["layer_norm"] = _verb
    tt._VERBS["layer_norm"] = _verb

    import dev_grad
    sys.argv = ["dev_grad.py"] + passthrough
    rc = dev_grad.main()

    # ---------------- the D240 census -----------------------------------------------------
    fan = {}
    bf16_consumed = fp32_grad_bf16_value = 0
    for _k, (n, first, final, valdt) in GRADLOG.items():
        fan[n] = fan.get(n, 0) + 1
        consumed = valdt          # autograd.backward casts g to t.value.dtype before the closure
        if "BFLOAT16" in consumed.upper():
            bf16_consumed += 1
        if final and "FLOAT32" in final.upper() and "BFLOAT16" in valdt.upper():
            fp32_grad_bf16_value += 1
    d240 = {
        "what": "D240: add_grad stores the first contribution as-is and promotes to fp32 only "
                "when a second arrives, so cotangent dtype is keyed on graph fan-out",
        "taped_tensors_that_received_a_gradient": len(GRADLOG),
        "fan_in_histogram_contributions_per_tensor": {str(k): v for k, v in sorted(fan.items())},
        "tensors_with_exactly_one_contribution": fan.get(1, 0),
        "n_grad_ended_bf16": sum(1 for v in GRADLOG.values()
                                 if v[2] and "BFLOAT16" in v[2].upper()),
        "n_grad_ended_fp32": sum(1 for v in GRADLOG.values()
                                 if v[2] and "FLOAT32" in v[2].upper()),
        "n_value_bf16_so_cotangent_is_CONSUMED_bf16": bf16_consumed,
        "n_fp32_grad_downcast_to_bf16_at_consumption": fp32_grad_bf16_value,
        "note": "autograd.backward casts the accumulated gradient to t.value.dtype before it "
                "calls the closure, so the dtype the closure CONSUMES is the value's, not the "
                "accumulator's. The last two counts are the ones that decide whether D240 can "
                "carry error.",
        "lever_counters": LEV,
    }

    torch.save = _real_save
    torch.save({"sites": {f"{f}:{b}": v for (f, b), v in CAP.items()},
                "real_rows": R, "lever": a.lever, "argv": passthrough}, a.cot_out)
    rep = {"what": "of3t-cotcoh pass 2 arm: fixed capture, D240 census, lever, trunk in process",
           "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "lever": a.lever,
           "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True).stdout.strip(),
           "argv": passthrough, "real_rows": R, "seconds": round(time.perf_counter() - t0, 1),
           "D240": d240,
           "capture": {f"{f}:{b}": {"gamma_path": v["gamma_path"], "P": int(v["g"].shape[0]),
                                    "C": int(v["g"].shape[1]), "n_summed": v["n_summed"],
                                    "x_max_abs_diff_between_fires":
                                        v["x_max_abs_diff_between_fires"],
                                    "per_fire_masked_sqnorm": v["per_fire_masked_sqnorm"],
                                    "outside_mask_sqnorm": v["outside_mask_sqnorm"],
                                    "fires": FIRES[(f, b)]["fires"],
                                    "nonzero_fires": FIRES[(f, b)]["nonzero"]}
                       for (f, b), v in sorted(CAP.items())},
           "TRUNK": {k: v for k, v in SCORE.items() if k != "per_tensor"},
           "cot_out": a.cot_out,
           "cot_out_bytes": os.path.getsize(a.cot_out) if os.path.exists(a.cot_out) else None}
    json.dump(rep, open(a.report, "w"), indent=1)
    if SCORE.get("per_tensor"):
        json.dump({"host": socket.gethostname(), "lever": a.lever,
                   "per_tensor": SCORE["per_tensor"]},
                  open(a.report.replace(".json", "_PERTENSOR.json"), "w"))
    print(json.dumps({"lever": a.lever, "n_A": sum(1 for k in CAP if k[0] == "A"),
                      "n_B": sum(1 for k in CAP if k[0] == "B"),
                      "trunk": SCORE.get("trunk_mass_weighted_rel_l2_vs_float64"),
                      "seconds": rep["seconds"]}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
