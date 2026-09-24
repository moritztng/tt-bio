#!/usr/bin/env python3
"""Put wk/bcx-heads' unified head verbs into an OF3T tree, byte for byte, and nothing else.

    transplant.py <of3t-tree> [<bcx-heads-tree>]

The two helpers are copied out of bcx-heads' `tt_bio/autograd.py` by `def` block, so the OF3T
tree runs the same source, not a retyping of it. In `taped_ttnn.py` the two backward bodies
that differ are replaced; every anchor must match exactly once or this refuses.
"""
import pathlib
import re
import sys

of3t = pathlib.Path(sys.argv[1])
bcx = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else pathlib.Path(__file__).resolve().parents[2])


def block(src, name):
    m = re.search(rf"^def {name}\(.*?(?=^\S)", src, re.S | re.M)
    assert m, name
    return m.group(0)


def swap(s, old, new):
    assert s.count(old) == 1, old[:80]
    return s.replace(old, new)


src = (bcx / "tt_bio/autograd.py").read_text()
helpers = block(src, "split_heads_value") + block(src, "merge_heads_value")

p = of3t / "tt_bio/autograd.py"
s = p.read_text()
assert "split_heads_value" not in s
anchor = "\ndef triangle_attention("
s = swap(s, anchor, "\n" + helpers + "\n" + anchor)
s = swap(s, '"reshape",', '"reshape", "split_heads_value", "merge_heads_value",')
p.write_text(s)

p = of3t / "tt_bio/taped_ttnn.py"
s = p.read_text()
CONCAT_BW = (                     # of3t-cropwall's form, then main's (of3t-stackexact carries it)
    '''            t = ttnn.reshape(ttnn.transpose(g, -2, -1), [B, H, dh, L])
            x.add_grad(ttnn.transpose(t, -2, -1))''',
    '''            x.add_grad(ttnn.permute(ttnn.reshape(g, [B, L, H, dh]), [0, 2, 1, 3]))''')
old = [c for c in CONCAT_BW if s.count(c) == 1]
assert len(old) == 1, "no known nlp_concat_heads backward body"
s = swap(s, old[0], '''            x.add_grad(ag.split_heads_value(g, H))''')
s = swap(s, '''                rows = ttnn.experimental.nlp_concat_heads(g)''',
         '''                rows = ttnn.reshape(ag.merge_heads_value(g), [B, 1, L, H * dh])''')
assert "if d > 5.0e-2 * (sc + 1.0e-30):" in s
p.write_text(s)
print("transplanted into", of3t)
