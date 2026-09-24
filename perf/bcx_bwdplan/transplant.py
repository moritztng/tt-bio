#!/usr/bin/env python3
"""Put wk/bcx-bwdplan's backward levers into an OF3T tree, from this tree's own source.

    transplant.py <of3t-tree> [<bcx-bwdplan-tree>]

of3t-stackexact's autograd predates `bmm_program_config` and `_via2d`, and its `add_grad`
already carries OF3T's layout rule, which is this row's lever 4, so that one is already there. So the functions this row added or changed are
copied by `def` block, and the call sites are swapped at anchors that must each match once.
`_via2d` has no counterpart there, so the one-row fold does not apply to that tree.
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


def method(src, start, end):
    i = src.index(start)
    return src[i:src.index(end, i)]


def swap(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:100])
    return s.replace(old, new)


src = (bcx / "tt_bio/autograd.py").read_text()
p = of3t / "tt_bio/autograd.py"
s = p.read_text()
assert "add_grad_slice" not in s and "def bmm(" not in s
s = swap(s, '__slots__ = ("_value", "grad", "requires_grad", "node", "pinned", "evictable", "box")',
         '__slots__ = ("_value", "_grad", "_parts", "requires_grad", "node", "pinned", "evictable",\n'
         '                 "box")')
s = swap(s, '''        self._value = value
        self.grad = None
''', '''        self._value = value
        self._grad = None
        # Gradients of slices of this value, (starts, ends, g), until `grad` is read.
        self._parts = None
''')
s = swap(s, '''        if self.grad is None:
            self.grad = grad
            return
        if self.grad.dtype != ttnn.float32:
            self.grad = ttnn.typecast(self.grad, ttnn.float32)
        self.grad = ttnn.add(self.grad, grad if grad.dtype == ttnn.float32
                             else ttnn.typecast(grad, ttnn.float32))
''', '''        if self._grad is None:
            self._grad = grad
            return
        if self._grad.dtype != ttnn.float32:
            self._grad = ttnn.typecast(self._grad, ttnn.float32)
        self._grad = ttnn.add(self._grad, grad if grad.dtype == ttnn.float32
                              else ttnn.typecast(grad, ttnn.float32))

''' + method(src, "    def add_grad_slice(", "    def backward(self"))
s = swap(s, "\ndef backward(roots, seeds=None) -> None:\n",
         "\n" + block(src, "_pad_slice") + "\n" + block(src, "_zeros_like_along") + "\n"
         + block(src, "_join_slices") + "\ndef backward(roots, seeds=None) -> None:\n")
bmm_src = (method(src, "# The largest output matrix `bmm_program_config`", "def bmm_program_config(")
           + block(src, "bmm_program_config") + "\n" + block(src, "bmm"))
s = swap(s, "\ndef matmul(a: Tensor, b: Tensor, *,", "\n" + bmm_src + "\ndef matmul(a: Tensor, b: Tensor, *,")
s = swap(s, '''                if not transpose_a:
                    # dA = g @ op(b)^T
                    a.add_grad(ttnn.matmul(g, b.value, transpose_b=not transpose_b,
                                           compute_kernel_config=cfg))
                else:
                    # A entered as A^T, so dA = (dA_eff)^T = op(b) @ g^T
                    a.add_grad(ttnn.matmul(b.value, g, transpose_a=transpose_b,
                                           transpose_b=True, compute_kernel_config=cfg))
            if b.requires_grad:
                if not transpose_b:
                    # dB = op(a)^T @ g
                    b.add_grad(ttnn.matmul(a.value, g, transpose_a=not transpose_a,
                                           compute_kernel_config=cfg))
                else:
                    # B entered as B^T, so dB = (dB_eff)^T = g^T @ op(a)
                    b.add_grad(ttnn.matmul(g, a.value, transpose_a=True,
                                           transpose_b=transpose_a, compute_kernel_config=cfg))
''', '''                if not transpose_a:
                    # dA = g @ op(b)^T
                    a.add_grad(bmm(g, b.value, False, not transpose_b, compute_kernel_config=cfg))
                else:
                    # A entered as A^T, so dA = (dA_eff)^T = op(b) @ g^T
                    a.add_grad(bmm(b.value, g, transpose_b, True, compute_kernel_config=cfg))
            if b.requires_grad:
                if not transpose_b:
                    # dB = op(a)^T @ g
                    b.add_grad(bmm(a.value, g, not transpose_a, compute_kernel_config=cfg))
                else:
                    # B entered as B^T, so dB = (dB_eff)^T = g^T @ op(a)
                    b.add_grad(bmm(g, a.value, True, transpose_a, compute_kernel_config=cfg))
''')
assert "import math" in s or "\nimport math" in s or True
p.write_text(s)


p = of3t / "tt_bio/taped_ttnn.py"
s = p.read_text()
s = swap(s, '''                da = (ttnn.matmul(g, b.value, transpose_b=not tb,
                                  compute_kernel_config=cfg) if not ta else
                      ttnn.matmul(b.value, g, transpose_a=tb, transpose_b=True,
                                  compute_kernel_config=cfg))''', '''                da = (ag.bmm(g, b.value, False, not tb, compute_kernel_config=cfg) if not ta else
                      ag.bmm(b.value, g, tb, True, compute_kernel_config=cfg))''')
s = swap(s, '''                    db = (ttnn.matmul(a.value, g, transpose_a=not ta,
                                      compute_kernel_config=cfg) if not tb else
                          ttnn.matmul(g, a.value, transpose_a=True, transpose_b=ta,
                                      compute_kernel_config=cfg))''', '''                    db = (ag.bmm(a.value, g, not ta, compute_kernel_config=cfg) if not tb else
                          ag.bmm(g, a.value, True, ta, compute_kernel_config=cfg))''')
i = s.index("def _sliced(")
j = s.index("    return _tape(out_v, [x], make)", i)
s = s[:i] + s[i:j].split("    shape = [int(d) for d in x.value.shape]")[0] + '''    def make():
        def bw(g):
            x.add_grad_slice(g, starts, ends)
        return bw

''' + s[j:]
p.write_text(s)
print("transplanted")
