import time, ttnn, torch
d = ttnn.open_device(device_id=0)
for shape in ([81, 4, 256, 32], [175, 4, 256, 32], [13, 4, 256, 32], [1, 256, 256, 64]):
    for _ in range(3):
        ttnn.deallocate(ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d))
    ttnn.synchronize_device(d)
    t = time.perf_counter()
    for _ in range(10):
        ttnn.deallocate(ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d))
    th = (time.perf_counter() - t) / 10
    ttnn.synchronize_device(d)
    tw = (time.perf_counter() - t) / 10
    x = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
    ttnn.synchronize_device(d)
    t = time.perf_counter()
    for _ in range(10):
        ttnn.deallocate(ttnn.zeros_like(x))
    zl = (time.perf_counter() - t) / 10
    ttnn.synchronize_device(d)
    zls = (time.perf_counter() - t) / 10
    print(shape, f"zeros host-return {th*1e3:.3f} ms synced {tw*1e3:.3f} ms;",
          f"zeros_like host-return {zl*1e3:.3f} ms synced {zls*1e3:.3f} ms;",
          f"{torch.Size(shape).numel()*2/1e6:.1f} MB", flush=True)
ttnn.close_device(d)
