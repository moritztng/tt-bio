import sys, torch
r = torch.load(sys.argv[1], weights_only=False); tt = torch.load(sys.argv[2], weights_only=False); up = torch.load(sys.argv[3], weights_only=False)
ref = r["ref"].reshape(-1, 384).double(); small = up[-1].abs().amax(0) < 512
wh_out = tt[91][:ref.shape[0]].double()
assert torch.allclose(r["s_in"].reshape(-1,384), up[74]), "s_in != upstream snapshot 74"
print("ref == upstream snapshot 75:", torch.allclose(ref.float(), up[75]))
def tok(x): return x - x.mean(0)
def rel(a, b): return float((a - b).norm() / b.norm())
for k in ("s_wh", "s_bf", "z_bf", "sz_bf", "s_wh_z_bf"):
    o = r[k].reshape(-1, 384).double()
    print(f"{k:10s} tok small rel {rel(tok(o[:,small]), tok(ref[:,small])):.3g}  tokstd {tok(o[:,small]).std():.3g}  "
          f"rel to WH out {rel(tok(o[:,small]), tok(wh_out[:,small])):.3g}")
print("WH out     tok small rel", f"{rel(tok(wh_out[:,small]), tok(ref[:,small])):.3g}", f"tokstd {tok(wh_out[:,small]).std():.3g}")
print("s_in WH vs up rel", rel(r['wh_in'][:ref.shape[0]].double(), r['s_in'].reshape(-1,384).double()), " z absmax", float(r['z_in'].abs().max()))
