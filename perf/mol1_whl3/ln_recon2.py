import sys, torch
sd = torch.load(sys.argv[1], map_location="cpu", weights_only=False, mmap=True); sd = sd.get("model", sd)
pre = "module.pairformer_stack.blocks.43.single_transition."
lw, lb = (sd[pre + k].double() for k in ("layernorm1.weight", "layernorm1.bias"))
tap = torch.load(sys.argv[2], weights_only=False)
g = lambda t: t.reshape(-1, t.shape[-1]).double()[:181]
x = g(tap["in"][0]); xh, wx = g(tap["multiply_"][0]), g(tap["multiply_"][1])
eff = (wx / xh).median(0).values
print("eff w / bf16(w) median", float((eff / lw.to(torch.bfloat16).double()).median()), "eff/w", float((eff/lw).median()))
d = (wx - xh * lw.to(torch.bfloat16).double())
print("wx - xh*bf16(w): max abs", float(d.abs().max()), "rel", float(d.norm() / wx.norm()))
tok = lambda t: t - t.mean(0)
print("tok std per stage: x", float(tok(x).std()), "xh", float(tok(xh).std()), "wx", float(tok(wx).std()))
i = tok(wx).std(0).argmax(); print("chan", int(i), "w", float(lw[i]), "xh mean", float(xh[:, i].mean()), "tok std xh", float(tok(xh)[:, i].std()), "wx tokstd", float(tok(wx)[:, i].std()))
print("wx[:3,i]", wx[:3, i].tolist(), "xh*w", (xh[:3, i] * lw[i]).tolist())
