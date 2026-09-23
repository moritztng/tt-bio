"""ttnn's fused SDPA is reached from exactly one place, and that place never passes it no mask.

With `attn_mask=None` the op returns the wrong attention at some chunk configs (q_chunk = k_chunk
= 128 at any head dim on Wormhole's 8x9 grid and Blackhole's 13x10; PCC 0.2-0.9 against fp32),
while the same call with an all-zero mask is right. esmc and saprot were fixed at their own call
sites first, which left every other call site free to reintroduce it. `tenstorrent.fused_sdpa` is
now the only caller, and the scan below fails on a new call that goes around it.
"""
import ast
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[1] / "tt_bio"
OP = "scaled_dot_product_attention"


def _is_ttnn_transformer(node):
    return (isinstance(node, ast.Attribute) and node.attr == "transformer"
            and isinstance(node.value, ast.Name) and node.value.id == "ttnn")


def _references():
    """(file, line, enclosing function) for every reference to ttnn's fused SDPA in tt_bio."""
    out = []
    for path in sorted(PKG.rglob("*.py")):
        if "_vendor" in path.parts:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = path.relative_to(PKG.parent).as_posix()

        def walk(node, fn):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    walk(child, child.name)
                    continue
                if isinstance(child, ast.Attribute) and child.attr == OP \
                        and _is_ttnn_transformer(child.value):
                    out.append((rel, child.lineno, fn))
                elif isinstance(child, ast.ImportFrom) and (child.module or "").startswith("ttnn") \
                        and any(a.name == OP for a in child.names):
                    out.append((rel, child.lineno, fn))
                elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name) \
                        and child.func.id == "getattr" and len(child.args) >= 2 \
                        and isinstance(child.args[1], ast.Constant) and child.args[1].value == OP:
                    out.append((rel, child.lineno, fn))
                walk(child, fn)

        walk(tree, None)
    return out


def test_every_fused_sdpa_call_goes_through_the_wrapper():
    refs = _references()
    bypass = [r for r in refs if r[0] != "tt_bio/tenstorrent.py" or r[2] != "fused_sdpa"]
    assert not bypass, (
        "these reach ttnn.transformer.scaled_dot_product_attention without "
        f"tenstorrent.fused_sdpa, which is the only place a missing mask is replaced: {bypass}")
    assert len(refs) == 1, refs


def test_the_scan_sees_a_bypass(tmp_path, monkeypatch):
    # A scan that finds nothing proves nothing unless it can find something.
    (tmp_path / "tt_bio").mkdir()
    (tmp_path / "tt_bio" / "m.py").write_text(
        "import ttnn\ndef f(q, k, v):\n    return ttnn.transformer.scaled_dot_product_attention(q, k, v)\n"
        "from ttnn.transformer import scaled_dot_product_attention\n"
        "g = getattr(ttnn.transformer, 'scaled_dot_product_attention')\n")
    monkeypatch.setattr(__import__(__name__), "PKG", tmp_path / "tt_bio")
    assert [(r[1], r[2]) for r in _references()] == [(3, "f"), (4, None), (5, None)]


def test_the_wrapper_never_forwards_none(monkeypatch):
    ttnn = pytest.importorskip("ttnn")
    ten = pytest.importorskip("tt_bio.tenstorrent")
    seen, freed = [], []

    class T:
        def __init__(self, shape):
            self.shape = shape

        def device(self):
            return "dev"

    monkeypatch.setattr(ttnn.transformer, OP,
                        lambda q, k, v, **kw: seen.append(kw) or "o", raising=False)
    monkeypatch.setattr(ttnn, "zeros", lambda shape, **kw: ("zeros", tuple(shape), kw["device"]))
    monkeypatch.setattr(ttnn, "deallocate", freed.append)
    q, k = T((2, 16, 128, 64)), T((2, 16, 96, 64))
    assert ten.fused_sdpa(q, k, k, None, scale=0.125) == "o"
    assert seen[-1]["attn_mask"] == ("zeros", (1, 1, 128, 96), "dev")
    assert seen[-1]["is_causal"] is False and freed == [seen[-1]["attn_mask"]]
    own = object()
    ten.fused_sdpa(q, k, k, own, scale=0.125)
    assert seen[-1]["attn_mask"] is own and len(freed) == 1


def test_a_ragged_length_masks_only_the_padded_keys():
    torch = pytest.importorskip("torch")
    esmc = pytest.importorskip("tt_bio.esmc")
    ids = torch.zeros(1, 126, dtype=torch.long)
    _, mask, key_valid, _, n = esmc.bucket_token_axis(ids, bucket=32)
    assert n == 126 and mask.shape == (1, 128, 128)
    assert torch.all(mask[..., :126] == 0) and torch.all(torch.isinf(mask[..., 126:]))
    assert torch.all(key_valid[:, :, 126:] == 0)
