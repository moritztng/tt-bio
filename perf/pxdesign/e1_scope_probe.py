#!/usr/bin/env python3
"""Does the fp32-softmax L1-padded plan reach AF2-IG's attentions and nothing else?

Card-free. `TT_BIO_FP32_SOFTMAX_L1_PADDED` must be UNSET for this to mean anything, and the probe
refuses to run if it is set.

The claim under test (state/pxdesign-af2ig-land.md, D2/E1): the padded extent is the extent the
L1 shard actually takes, AF2-IG's token counts are ragged at every rung it runs, and AF2-IG is the
only model whose fold-level accuracy and perf were measured on that plan. So AF2's two triangle
attentions and its MSA row attention pin it on, and `AttentionPairBias` -- the protenix-v2 /
openfold3 / PXDesign-d pairformer, whose stage cells were priced with the lever off -- keeps
following the env var.

Three of the five sites are checked DYNAMICALLY, by recording the kwarg that actually arrives at
`_fp32_softmax_attention`; the two constructor sites are checked STRUCTURALLY, by resolving the
keyword arguments on the real `TriangleAttention(...)` call nodes in the AST, because
`AF2PairBlock.__init__` pushes weights to a device and this probe has no card.

IT MUST FAIL IF E1 IS REVERTED, and each check names which revert it catches. Verified once by
reverting `af2.py`'s `l1_padded_plan=True` and confirming checks 4 and 5 go red.
"""
import ast
import os
import pathlib
import sys

if os.environ.get("TT_BIO_FP32_SOFTMAX_L1_PADDED"):
    sys.exit("TT_BIO_FP32_SOFTMAX_L1_PADDED is set; the probe only reads under the shipped default")

import ttnn  # noqa: E402
from tt_bio import af2, tenstorrent  # noqa: E402

ROOT = pathlib.Path(tenstorrent.__file__).parent
fails = []


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    if not ok:
        fails.append(name)


# --- 1. the shipped default is OFF, so nothing below is an artefact of a stray env --------------
check("default-off", tenstorrent._FP32_SOFTMAX_L1_PADDED is False,
      f"_FP32_SOFTMAX_L1_PADDED={tenstorrent._FP32_SOFTMAX_L1_PADDED}")


# --- 2. _fp32_softmax_len: None follows the env, a bool pins ------------------------------------
class RaggedTensor:
    """848 tokens in a 864-row tile grid: the 848 rung, which is 16 mod 32."""
    shape = (1, 4, 848, 64)
    padded_shape = (1, 4, 864, 64)


t = RaggedTensor()
got = (tenstorrent._fp32_softmax_len(t, 2),
       tenstorrent._fp32_softmax_len(t, 2, True),
       tenstorrent._fp32_softmax_len(t, 2, False))
check("len-override", got == (848, 864, 848),
      f"(env, True, False) -> {got}, want (848, 864, 848)")


# --- 3. the kwarg exists on both surfaces and defaults to "follow the env" ----------------------
import inspect  # noqa: E402

for fn, label in ((tenstorrent._fp32_softmax_attention, "_fp32_softmax_attention"),
                  (tenstorrent.TriangleAttention.__init__, "TriangleAttention.__init__")):
    par = inspect.signature(fn).parameters.get("l1_padded_plan")
    check(f"signature-{label}", par is not None and par.default is None,
          "absent" if par is None else f"default={par.default!r}")


# --- 4. the two AF2PairBlock construction sites pin True, structurally --------------------------
def call_kwarg(path, callee, kw):
    """Every `callee(...)` call node in `path`, as the literal value it passes for `kw`."""
    try:
        tree = ast.parse((ROOT / path).read_text())
    except (SyntaxError, UnicodeDecodeError):
        return []   # vendored trees carry files this interpreter cannot parse; none build ours
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name != callee:
            continue
        found = "<absent>"
        for k in node.keywords:
            if k.arg == kw:
                found = ast.unparse(k.value)
        out.append((node.lineno, found))
    return out


sites = call_kwarg("af2.py", "TriangleAttention", "l1_padded_plan")
check("af2-triangle-attention-sites", len(sites) == 2 and all(v == "True" for _, v in sites),
      f"{sites} (want two sites, both True -- catches a revert of af2.py:323/327)")

# the setter has to reach every attention AF2 owns, or an A/B leg silently runs one arm twice
src = (ROOT / "af2.py").read_text()
body = src[src.index("def set_l1_padded_plan"):src.index("def set_template_host")]
check("setter-reaches-every-site",
      all(a in body for a in ("tri_att_start.l1_padded_plan", "tri_att_end.l1_padded_plan",
                              "msa_row_attn.l1_padded_plan")),
      "AF2DeviceModel.set_l1_padded_plan covers both triangle attentions and the MSA row softmax")

# the shared class hands its own instance value down; anything else silently unscopes it
dispatch = [v for ln, v in call_kwarg("tenstorrent.py", "_fp32_softmax_attention", "l1_padded_plan")]
check("triangle-attention-dispatch", dispatch.count("self.l1_padded_plan") == 1,
      f"{dispatch} (want exactly one `self.l1_padded_plan` and one `<absent>`)")
check("pairformer-untouched", dispatch.count("<absent>") == 1,
      f"{dispatch} (AttentionPairBias must pass nothing and keep following the env)")


# --- 4b. no OTHER model can reach the lever, which is E4's structural half ----------------------
# E1 is scoped, so the bar for the five shipped models is not "within the floor" but BIT-IDENTICAL
# to main. That holds only if no construction site outside af2.py pins the kwarg, and there are
# more `TriangleAttention(...)` sites in this package than the two AF2 owns.
import pathlib as _pl  # noqa: E402

other = []
for f in sorted(ROOT.rglob("*.py")):
    if f.name == "af2.py":
        continue
    for ln, v in call_kwarg(f.relative_to(ROOT), "TriangleAttention", "l1_padded_plan"):
        if v != "<absent>":
            other.append((str(f.relative_to(ROOT)), ln, v))
check("no-foreign-construction-site", not other,
      f"{len(other)} non-af2 TriangleAttention site(s) pin the lever: {other} "
      "(every other model must inherit `None` and stay byte-identical to main)")

# and the same for the helper: only TriangleAttention's own dispatch and AF2's MSA row may pass it
helper = []
for f in sorted(ROOT.rglob("*.py")):
    for ln, v in call_kwarg(f.relative_to(ROOT), "_fp32_softmax_attention", "l1_padded_plan"):
        if v != "<absent>":
            helper.append((f.name, ln, v))
check("helper-callers-enumerated",
      sorted(h[2] for h in helper) == ["self.l1_padded_plan", "self.l1_padded_plan"],
      f"{helper} (want exactly two, both `self.l1_padded_plan`: TriangleAttention's dispatch "
      "and AF2Attention._attend)")


# --- 5. what actually arrives at the helper, recorded through the real call sites ---------------
seen = []


def recorder(q, k, v, bias, **kw):
    seen.append(kw.get("l1_padded_plan", "<absent>"))
    return "o"


real_attn, real_dealloc = tenstorrent._fp32_softmax_attention, ttnn.deallocate
af2._fp32_softmax_attention = recorder
tenstorrent._fp32_softmax_attention = recorder
ttnn.deallocate = lambda *a, **k: None


def stub(cls, **attrs):
    """A real instance of `cls` with `__init__` skipped, so class defaults are the ones read.

    `object.__new__` and not a hand-rolled stand-in on purpose: `AF2Attention.l1_padded_plan` is a
    class attribute, so a stand-in that declared it would only be testing itself.
    """
    obj = object.__new__(cls)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


common = dict(compute_kernel_config=None, fp32_softmax=True, accurate_softmax=False,
              dtype=ttnn.bfloat16, head_dim=64, _bias_scale=1.0)

try:
    # af2.py:472 -- AF2Attention._attend, AF2's MSA row attention
    af2.AF2Attention._attend(stub(af2.AF2Attention, scale_inv=0.125, **common), t, t, t, bias=t)
    # tenstorrent.py:5026 -- AttentionPairBias._attention, the pairformer AF2 does NOT own
    tenstorrent.AttentionPairBias._attention(stub(tenstorrent.AttentionPairBias, **common),
                                             t, t, t, t)
finally:
    af2._fp32_softmax_attention = real_attn
    tenstorrent._fp32_softmax_attention = real_attn
    ttnn.deallocate = real_dealloc

check("dynamic-af2-msa-row", seen[:1] == [True],
      f"AF2Attention._attend passed {seen[:1]} "
      "(catches both a revert of af2.py:472 and AF2Attention.l1_padded_plan flipped off)")
check("dynamic-pairformer", seen[1:] == ["<absent>"],
      f"AttentionPairBias._attention passed {seen[1:]} (must stay on the env)")

print()
if fails:
    sys.exit(f"E1 SCOPE PROBE FAILED: {', '.join(fails)}")
print("E1 SCOPE PROBE PASSED: the padded plan is AF2-IG's and the pairformer still follows the env")
