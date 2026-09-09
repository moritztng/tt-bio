import os, torch, ttnn
c = int(os.environ["TT_VISIBLE_DEVICES"])
d = ttnn.open_device(device_id=0)
try:
    a = torch.randn(1, 1, 64, 64); b = torch.randn(1, 1, 64, 64)
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=d, dtype=ttnn.bfloat16)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=d, dtype=ttnn.bfloat16)
    add = ttnn.to_torch(ttnn.add(ta, tb))
    mm = ttnn.to_torch(ttnn.matmul(ta, tb))
    ea = (add - (a + b)).abs().max().item()
    em = (mm - (a @ b)).abs().max().item() / (a @ b).abs().max().item()
    print("smoke card=%d arch=%s add_abs=%.4g mm_rel=%.4g" % (c, d.arch(), ea, em), flush=True)
finally:
    ttnn.close_device(d)
