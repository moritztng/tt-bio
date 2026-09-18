"""Run the LoRA census and the adapter through the live dispatch, off-card.

The protocol gate (`tests/test_training_hook_protocol.py`) is static: it reads the hook
classes and checks each carries a `__call__` of the right arity. That catches the TypeError
but cannot say the reattached hooks still DO what they did. This runs them.

`ttnn.linear` and `ttnn.layer_norm` are replaced by stand-ins returning a marker, so the
dispatch, both hooks and `_site_name` are the real ones and only the kernel is fake. No
device is opened.

The last section is the negative control for the second defect this row fixed. The hook is
now called from `OpSurface.dispatching`'s wrapper rather than from `ops.linear`, so
`dispatch.py` had to join `_site_name`'s skip list. Dropping it again must collapse every
site onto one `dispatch.py` entry, or the check above is passing for some other reason.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE)]

import ttnn                                    # noqa: E402

import _fake_model as fm                       # noqa: E402
from tt_bio import ops                         # noqa: E402
from tt_bio.train import lora                  # noqa: E402


class FakeT:
    def __init__(self, shape):
        self.shape = shape


def main() -> int:
    ttnn.linear = lambda x, w, **kw: FakeT((x.shape[0], w.shape[-1]))
    ttnn.layer_norm = lambda x, **kw: FakeT(x.shape)

    x, wq, wv, gamma = FakeT((32, 128)), FakeT((128, 64)), FakeT((128, 256)), FakeT((128,))
    fail = []

    # ---- Tier 1: the census records both linear sites and declines every call ------------
    sites = lora.census(fm.forward, x, wq, wv, gamma)
    print("census sites:")
    for _, s in sorted(sites.items()):
        print(f"  {s}")

    if len(sites) != 2:
        fail.append(f"census found {len(sites)} sites, expected 2")
    if any("dispatch.py" in n for n in sites):
        fail.append("a site is named after tt_bio/dispatch.py: _site_name stopped on the "
                    "OpSurface.dispatching wrapper instead of the real caller")
    if not all("_fake_model.py" in n and n.endswith(":block") for n in sites):
        fail.append(f"sites are not named after the caller: {sorted(sites)}")
    shapes = sorted((s.in_features, s.out_features, s.calls, s.has_bias) for s in sites.values())
    if shapes != [(128, 64, 3, False), (128, 256, 3, False)]:
        fail.append(f"wrong shapes or call counts off the census: {shapes}")

    # Declining is what makes a census safe on the real forward, and it must not stay armed.
    if not isinstance(ops.linear(x, wq), FakeT):
        fail.append("the census did not decline: the value is not production's")
    if ops.grad_hook() is not None:
        fail.append("the census did not restore the previous hook")

    # ---- Tier 2: the adapter adapts its site and delegates everything else ---------------
    lora.lora_linear = lambda *a, **k: "ADAPTED"   # the tape wants a device; the routing does not
    seen, installed = [], []

    def base(name, shipped, args, kwargs):
        seen.append(name)
        return None                                 # decline, so production runs

    target = sorted(sites)[0]
    ops.set_grad_hook(base)
    try:
        with lora.attach({target: ("A", "B")}, lora.LoraConfig(rank=8, alpha=16.0)):
            installed.append(ops.grad_hook())
            out = fm.block(x, wq, wv, gamma)
    finally:
        ops.set_grad_hook(None)

    print(f"\nadapted site:      {target}")
    print(f"delegated to base: {seen}")
    if out[0] != "ADAPTED":
        fail.append(f"the targeted site was not adapted: {out[0]!r}")
    if sorted(seen) != ["layer_norm", "linear"]:
        fail.append(f"delegation is wrong: base saw {seen}, expected one linear + one layer_norm")
    adapter = type(installed[0])
    for cls in (lora._Census, adapter):
        stale = [m for m in ("linear", "layer_norm") if hasattr(cls, m)]
        if stale:
            fail.append(f"{cls.__name__} still exposes retired entry points {stale}")

    # ---- negative control: the skip-list entry is load-bearing ---------------------------
    keep = lora._INTERNAL
    lora._INTERNAL = tuple(p for p in keep if not p.endswith("dispatch.py"))
    try:
        got = lora.census(fm.forward, x, wq, wv, gamma)
        note, blamed = f"{len(got)} site(s): {sorted(got)}", any("dispatch.py" in n for n in got)
        # Collapsing is only a collapse if it actually lost a site.
        blamed = blamed and len(got) < 2
    except ValueError as exc:
        # The two linears now share one apparent site, and the census refuses two shapes at
        # one site rather than picking either. That refusal IS the collapse.
        note, blamed = f"ValueError: {exc}", "dispatch.py" in str(exc)
    finally:
        lora._INTERNAL = keep
    print(f"\ncontrol, dispatch.py off the skip list: {note}")
    if not blamed:
        fail.append("control did not collapse onto dispatch.py, so the skip-list entry is "
                    "not what makes the naming right")

    print("\n" + ("FAIL\n  " + "\n  ".join(fail) if fail else
                  "PASS: census naming, decline, restore, adapter routing and delegation all hold"))
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
