import sys
p, which = sys.argv[1], sys.argv[2]
src = open(p).read()
DUP = """            ttnn.deallocate(o_in)
            if self.o_bias is not None:
                x_out = ttnn.add_(x_out, self.o_bias)
            return x_out"""
INMM = """                    o_in, self.o_weight, self.compute_kernel_config, _dtype(), l1_out=True,
                    bias=self.o_bias,
                )"""
if which == "obias":
    # keep the in-matmul bias (AF2 branch form), drop the duplicate elementwise add
    assert src.count(DUP) == 1, f"DUP sites: {src.count(DUP)}"
    src = src.replace(DUP, """            ttnn.deallocate(o_in)
            return x_out""")
elif which == "mainform":
    # keep the elementwise add (RF3/main form), drop the in-matmul bias
    assert src.count(INMM) == 1, f"INMM sites: {src.count(INMM)}"
    src = src.replace(INMM, """                    o_in, self.o_weight, self.compute_kernel_config, _dtype(), l1_out=True,
                )""")
else:
    raise SystemExit(f"unknown patch {which}")
open(p, "w").write(src)
print(f"# patched {which}")
