"""Upstream Mol-1's ConfidenceHead on the CPU, on the inputs a TT fold handed its own head, graded against the arms.

    python -I conf_ref.py CKPT DUMP_DIR [--threads N]

DUMP_DIR holds inputs.pt and arms.pt from hook/sitecustomize.py. Runs upstream's head (mol/model/modules/confidence.py,
torch backends) in float32 and float64 on exactly those s_inputs / s_trunk / z_trunk / coords, so trunk and diffusion
differences between devices drop out: what is left is the head itself. Writes DUMP_DIR/ref.json with, per arm, mean
pLDDT per sample, the largest per-atom pLDDT gap to float64, and the relative error of the confidence pairformer's
single output (s_single) against float64.
Run in the upstream venv (molence installed), from outside the clone.
"""
import argparse
import json
from pathlib import Path

import torch


def expected_plddt(logits):
    nb = logits.shape[-1]
    return torch.softmax(logits.double(), -1) @ ((torch.arange(nb, dtype=torch.float64) + 0.5) / nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("dump")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from mol.model.modules.confidence import ConfidenceHead

    d = Path(a.dump)
    inp = torch.load(d / "inputs.pt", weights_only=False)
    arms = torch.load(d / "arms.pt", weights_only=False)
    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    pre = next(p for p in ("module.confidence_head.", "confidence_head.") if any(k.startswith(p) for k in sd))
    w = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    n_ta, c_s, _ = w["plddt_weight"].shape
    head = ConfidenceHead(n_blocks=4, c_s=c_s, c_z=w["linear_no_bias_s1.weight"].shape[0],
                          c_s_inputs=w["linear_no_bias_s1.weight"].shape[1], max_atoms_per_token=n_ta,
                          hidden_scale_up=True)
    head.load_state_dict(w, strict=True)
    head.eval()

    # Upstream casts the distances and the pairformer outputs to float32 by name; under float64 every leaf module
    # takes its inputs at its own weight's dtype, so the float64 run stays float64 past those casts.
    def at_weight_dtype(mod, args, kwargs):
        p = next(mod.parameters(recurse=False), None)
        cast = lambda x: x.to(p.dtype) if torch.is_tensor(x) and x.is_floating_point() else x  # noqa: E731
        return tuple(cast(x) for x in args), {k: cast(v) for k, v in kwargs.items()}

    for m in head.modules():
        if next(m.parameters(recurse=False), None) is not None:
            m.register_forward_pre_hook(at_weight_dtype, with_kwargs=True)

    taps = []
    stack_fwd = head.pairformer_stack.forward

    def tapped(*args, **kw):
        s, z = stack_fwd(*args, **kw)
        taps.append(s.detach().clone())
        return s, z

    head.pairformer_stack.forward = tapped
    feats = dict(inp["feats"])
    ref = {}
    for dt in (torch.float32, torch.float64):
        taps.clear()
        h = head.to(dt)
        with torch.no_grad():
            plddt, _, _, _ = h(input_feature_dict=feats, s_inputs=inp["s_inputs"].to(dt), s_trunk=inp["s_trunk"].to(dt),
                               z_trunk=inp["z_trunk"].to(dt), pair_mask=None, x_pred_coords=inp["coords"].to(dt))
        ref[str(dt).split(".")[-1]] = {"atom": expected_plddt(plddt), "s": torch.stack(taps).double()}
    f64, f32 = ref["float64"], ref["float32"]

    def grade(atom, s):
        out = {"plddt": [round(float(x), 4) for x in atom.mean(-1)],
               "max_atom_gap": round(float((atom - f64["atom"]).abs().max()), 4),
               "mean_atom_gap": round(float((atom - f64["atom"]).abs().mean()), 4)}
        if s is not None and len(s):
            n = f64["s"].shape[-2]  # a TT tap may carry the padded token axis
            s = torch.stack([x.double().reshape(-1, x.shape[-1])[:n] for x in s]) if isinstance(s, list) else s
            out["s_rel_err"] = [round(float((x - y).norm() / y.norm()), 5) for x, y in zip(s, f64["s"])]
        return out

    res = {"levers": inp["levers"], "fold_plddt": inp["fold_plddt"],
           "upstream_float64": grade(f64["atom"], f64["s"]), "upstream_float32": grade(f32["atom"], f32["s"])}
    for name, r in arms.items():
        if name == "fold" or "plddt_atom" not in r:
            res[name] = r if name == "fold" else {"error": r.get("error")}
            continue
        res[name] = grade(r["plddt_atom"].double(), r["s_single"])
    (d / "ref.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if isinstance(v, dict) and "max_atom_gap" in v:
            print(f"{k:28s} plddt {' '.join(f'{x:.3f}' for x in v['plddt'])}  max|d| {v['max_atom_gap']:.3f}"
                  f"  s_rel {v.get('s_rel_err', ['-'])[0]}")


if __name__ == "__main__":
    main()
