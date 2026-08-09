"""Bit-exact comparison of two w3_bench dumps: python3 w3_cmp.py stock dualnoc"""
import sys, torch
a = torch.load("/tmp/w3_%s.pt" % sys.argv[1])
b = torch.load("/tmp/w3_%s.pt" % sys.argv[2])
ok = True
for k in sorted(a):
    eq = torch.equal(a[k], b[k])
    md = (a[k].float() - b[k].float()).abs().max().item()
    ok &= eq
    print("%-18s bit_exact=%-5s max_abs_diff=%g" % (k, eq, md))
print("ALL_BIT_EXACT " + str(ok))
