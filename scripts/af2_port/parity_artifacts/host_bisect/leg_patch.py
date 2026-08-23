"""In-tree patcher for the P19 bisect legs. `which` is a comma-separated op list, so a
control can force a branch open and change one line inside it in the same leg."""
import sys

p, which = sys.argv[1], sys.argv[2]
src = open(p).read()

# gate_and_project: the merge kept both parents fixes for the same missing linear_o.bias.
O_ADD = """            ttnn.deallocate(o_in)
            if self.o_bias is not None:
                x_out = ttnn.add_(x_out, self.o_bias)
            return x_out"""
O_INMM = """                    o_in, self.o_weight, self.compute_kernel_config, _dtype(), l1_out=True,
                    bias=self.o_bias,
                )"""
# TriangleAttention.__call__: the chunked-branch gate.
NEED = """        need_chunk = S > SEQ_LEN_MORE_CHUNKING and (self.affinity or not _FAST_MODE or _IS_SMALL_GRID)"""
QKV = """                qkv_chunk = _triatt_qkv.qkv_heads(
                    x_chunk, self.qkv_weight, self.compute_kernel_config,
                    self.n_heads, self.head_dim, _dtype(), qkv_cfg_chunk,
                )"""
# Same double-apply as O_ADD/O_INMM, on linear_g.bias, chunked site only.
G_INMM = """                        bias_tensor=self.g_bias,
"""
G_ADD = """                if self.g_bias is not None:
                    g_chunk = ttnn.add_(g_chunk, self.g_bias)
"""


def sub(old, new):
    global src
    assert src.count(old) == 1, f"{src.count(old)} sites for {old[:60]!r}"
    src = src.replace(old, new)


OPS = {
    # keep the in-matmul bias (AF2 branch form), drop the duplicate elementwise add
    "obias": lambda: sub(O_ADD, """            ttnn.deallocate(o_in)
            return x_out"""),
    # keep the elementwise add (RF3/main form), drop the in-matmul bias
    "mainform": lambda: sub(O_INMM, """                    o_in, self.o_weight, self.compute_kernel_config, _dtype(), l1_out=True,
                )"""),
    # Open the chunked qkv branch at the gate fixture size (208 tokens), far below
    # SEQ_LEN_MORE_CHUNKING. Patching the branch condition rather than the constant keeps the
    # blast radius to this one branch: the constant is read by 12 other call sites, and on a
    # full Blackhole grid _apply_grid_thresholds returns before the
    # TT_BIO_SEQ_LEN_MORE_CHUNKING override, so the env lever is dead on qb2 anyway.
    "forcechunk": lambda: sub(NEED, """        need_chunk = True  # forced: chunked-branch control"""),
    # restore ce464f0a`s dangling reference: self.fuse_qkv is assigned nowhere
    "dark": lambda: sub(QKV, QKV[:-1] + """) if self.fuse_qkv else None"""),
    # drop the in-matmul g_bias so the chunked path applies it once, as unchunked does
    "gfix": lambda: sub(G_INMM, ""),
    # the other half of the pair: keep in-matmul, drop the chunked elementwise add
    "gfix_inmm": lambda: sub(G_ADD, ""),
}

for op in which.split(","):
    if op not in OPS:
        raise SystemExit(f"unknown patch op {op}")
    OPS[op]()
open(p, "w").write(src)
print(f"# patched {which}")
