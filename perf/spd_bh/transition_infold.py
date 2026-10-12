"""spd-gap-small: which in-fold Transition call does a row-height change move?

    PYTHONPATH=TREE python perf/spd_bh/transition_infold.py --report OUT.json -- <xmodel_ab.py args>

Wraps Transition._transition, then runs perf/of3_4xpd/xmodel_ab.py in this process with the given
arguments (use --ab-attr tt_bio.tenstorrent:_SMALL_GRID_TRANSITION_H512 --repeat 1). Every call is
logged by signature (shape, dtype, add_to_input, buffer type, b8 hidden, row height used) under the
attribute's current value. The first call of each signature whose shipped height differs from the
height with the attribute flipped is re-run at the flipped height on a copy of its input, and both
outputs are compared to each other (torch.equal) and to a float64 host reference (rel_rms), so a
precision move (both equally close to float64) is told apart from a wrong value.
"""
import json, runpy, sys
from pathlib import Path

import torch


def ref64(mod, x, add):
    w = {k: v.double() for k, v in mod.weights.items() if k.split(".")[0] in ("norm", "fc1", "fc2", "fc3")}
    xn = torch.nn.functional.layer_norm(x, x.shape[-1:], w["norm.weight"], w["norm.bias"])
    h = torch.nn.functional.silu(xn @ w["fc1.weight"].t()) * (xn @ w["fc2.weight"].t())
    t = h @ w["fc3.weight"].t()
    return x + t if add else t


def rel(a, b):
    return float(((a - b).norm() / b.norm()).item())


def main():
    i = sys.argv.index("--")
    report = Path(sys.argv[sys.argv.index("--report") + 1])
    rest = sys.argv[i + 1:]
    import ttnn
    from tt_bio import tenstorrent as T

    calls, probes, seen = {}, [], set()
    orig = T.Transition._transition

    def wrapped(self, x, memory_config, add_to_input):
        flag = T._SMALL_GRID_TRANSITION_H512
        sig = (tuple(x.shape), str(x.dtype), bool(add_to_input),
               str(memory_config.buffer_type) if memory_config is not None else "None", bool(self._hidden_b8))
        probe = flag and sig not in seen and len(x.shape) == 4
        if probe:
            seen.add(sig)
            x_in = ttnn.to_torch(x).double()
            x_copy = ttnn.clone(x)
        out = orig(self, x, memory_config, add_to_input)
        h = self._h_last[1] if self._h_last else None
        key = json.dumps([flag, *sig, h])
        calls[key] = calls.get(key, 0) + 1
        if probe:
            T._SMALL_GRID_TRANSITION_H512 = False
            try:
                out_off = orig(self, x_copy, memory_config, add_to_input)
                h_off = self._h_last[1] if self._h_last else None
            finally:
                T._SMALL_GRID_TRANSITION_H512 = True
            a, b = ttnn.to_torch(out).double(), ttnn.to_torch(out_off).double()
            rec = {"sig": sig, "h_on": h, "h_off": h_off, "equal": bool(torch.equal(a, b)),
                   "rel_on_off": rel(a, b), "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
            try:
                r = ref64(self, x_in, add_to_input)
                rec.update(rel_rms_on=rel(a, r), rel_rms_off=rel(b, r))
            except Exception as exc:        # weights not kept on the host: the pair still answers equal / not
                rec["ref_error"] = repr(exc)[:200]
            probes.append(rec)
            if out_off is not x_copy:
                ttnn.deallocate(out_off)
            if x_copy.is_allocated():
                ttnn.deallocate(x_copy)
            report.write_text(json.dumps({"probes": probes, "calls": calls}, indent=1))
        return out

    T.Transition._transition = wrapped
    sys.argv = [str(Path(__file__).resolve().parents[1] / "of3_4xpd" / "xmodel_ab.py"), *rest]
    try:
        runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        report.write_text(json.dumps({"probes": probes, "calls": calls,
                                      "clash": {str(k): v for k, v in T.TRANSITION_H_CLASH.items()}}, indent=1))


if __name__ == "__main__":
    main()
