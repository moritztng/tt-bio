"""Probe: pick a cycle's MSA rows on the chip from a resident row-major table (ttnn.embedding) vs host cut + upload.

Times both at c730 depth buckets and checks the gathered tensor and its projection are bit-identical to the host way.
    TT_VISIBLE_DEVICES=<chip> python perf/spd/msa_rows_probe.py
"""
import json, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device
dev = get_device()
g = torch.Generator().manual_seed(0)
D, N, c = 9947, 736, 34
tok = torch.randint(0, 32, (D, N), generator=g)
ms = torch.cat([torch.nn.functional.one_hot(tok, 32).to(torch.bfloat16),
                (torch.rand(D, N, 1, generator=g) > 0.8).to(torch.bfloat16),
                torch.rand(D, N, 1, generator=g).to(torch.bfloat16)], -1).unsqueeze(0).contiguous()
W = ttnn.from_torch(torch.randn(34, 128, generator=g).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
t0 = time.perf_counter()
flat = torch.cat([ms.view(D, N * c), ms.new_zeros(1, N * c)])
res = ttnn.from_torch(flat, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.bfloat16)
ttnn.synchronize_device(dev)
print(json.dumps(dict(resident_upload_s=time.perf_counter() - t0, shape=list(res.shape))), flush=True)
for k, bucket in ((9000, 9216), (2000, 2048), (5000, 5120)):
    rows = torch.randperm(D, generator=g)[:k]
    # host way (what staging19 does)
    x_h = ms.new_zeros((1, bucket, N, c)); x_h[:, :k] = ms[:, rows]
    ttnn.synchronize_device(dev); t0 = time.perf_counter()
    xa = ttnn.from_torch(x_h, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    h_host = time.perf_counter() - t0; ttnn.synchronize_device(dev); w_host = time.perf_counter() - t0
    res_d = {}
    for name in ("embed",):
        ts = []
        for r in range(4):
            ttnn.synchronize_device(dev); t0 = time.perf_counter()
            idx = torch.full((1, bucket), D, dtype=torch.int32); idx[0, :k] = rows.to(torch.int32)
            i = ttnn.from_torch(idx, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            gth = ttnn.embedding(i, res, layout=ttnn.ROW_MAJOR_LAYOUT)
            v = ttnn.reshape(gth, (1, bucket, N, c))
            xb = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
            h = time.perf_counter() - t0
            ttnn.synchronize_device(dev); w = time.perf_counter() - t0
            ts.append((h, w))
            ttnn.deallocate(i); ttnn.deallocate(gth)
            if r < 3:
                ttnn.deallocate(xb)
        ya, yb = ttnn.linear(xa, W), ttnn.linear(xb, W)
        bits = lambda u: ttnn.to_torch(u).view(torch.int16)
        print(json.dumps(dict(k=k, bucket=bucket, host_way_host_s=h_host, host_way_wall_s=w_host,
                              embed_host_s=sorted(ts[1:])[1][0], embed_wall_s=sorted(t[1] for t in ts[1:])[1],
                              gather_shape=list(gth.shape) if False else None,
                              x_equal=bool(torch.equal(bits(xa), bits(xb))), proj_equal=bool(torch.equal(bits(ya), bits(yb))))), flush=True)
        ttnn.deallocate(xb)
    ttnn.deallocate(xa)
