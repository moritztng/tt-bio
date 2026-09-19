#!/usr/bin/env python3
"""Backward-op inventory at the tt-bio pin. Opens NO device: nanobind overload text is
printed by provoking a TypeError, which needs no device handle.

Run with the tt-bio venv:  /home/moritz/tt-bio/env/bin/python op_inventory.py
"""
import ttnn

MOREH_BACKWARD = [
    "moreh_matmul_backward", "moreh_linear_backward", "moreh_bmm_backward",
    "moreh_dot_backward", "moreh_layer_norm_backward", "moreh_group_norm_backward",
    "moreh_softmax_backward", "moreh_logsoftmax_backward", "moreh_softmin_backward",
    "moreh_sum_backward", "moreh_mean_backward", "moreh_norm_backward",
    "moreh_cumsum_backward", "moreh_nll_loss_backward",
]
# every eltwise op Protenix v2 dispatches, against its *_bw
PROTENIX_ELTWISE = ["add", "mul", "relu", "sigmoid", "sqrt", "pow", "clamp",
                    "addcmul", "addalpha", "embedding", "concat", "div", "sub"]


def sig(name):
    op = getattr(ttnn, name, None)
    if op is None:
        return "ABSENT AT THIS PIN"
    try:
        op(object())
    except TypeError as e:
        # nanobind prints "Called with:" then "Available signatures:"; take the latter.
        txt = str(e)
        tail = txt.split("Available signatures:", 1)
        if len(tail) == 2:
            return "\n  ".join(l.strip() for l in tail[1].strip().splitlines() if l.strip())
        return "(no Available signatures block)"
    except Exception as e:
        return f"(no overload text: {type(e).__name__})"
    return "(no overload text)"


if __name__ == "__main__":
    allm = sorted(n for n in dir(ttnn) if "moreh" in n.lower())
    bw = sorted(n for n in dir(ttnn) if n.endswith("_bw"))
    print(f"moreh ops at this pin: {len(allm)}   eltwise *_bw ops: {len(bw)}")
    print(f"moreh *_backward present: "
          f"{sum(1 for n in MOREH_BACKWARD if hasattr(ttnn, n))}/{len(MOREH_BACKWARD)}\n")
    for n in MOREH_BACKWARD:
        print("=" * 100)
        print(n)
        print("  " + sig(n))
    print("\n" + "=" * 100)
    print("Protenix eltwise coverage (backward of the ops the forward census dispatches):")
    for n in PROTENIX_ELTWISE:
        print(f"  {n:12s} -> {n + '_bw' if hasattr(ttnn, n + '_bw') else 'NONE'}")
    print("\nShape ops (reshape/permute/slice/pad/to_layout/typecast/unsqueeze) have no *_bw and")
    print("need none: the backward of a shape op is the inverse shape op, already in ttnn.")
