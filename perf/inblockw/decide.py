from tuned_cfg import _tuned_matmul_config as f
G = (11, 10); L1 = 1532416
cases = [('pair proj c_z', 3200, 4, 4), ('pair trans up', 3200, 4, 16),
         ('pair trans down', 3200, 16, 4), ('trimul in', 3200, 4, 8),
         ('trimul out', 3200, 8, 8), ('bias nt1', 3200, 4, 1),
         ('117 trimul', 512, 8, 8), ('117 trans down', 512, 16, 4),
         ('single trans up', 10, 24, 96), ('single trans down', 10, 96, 24),
         ('single proj', 10, 24, 24), ('117 single proj', 4, 24, 24),
         ('atom proj', 140, 4, 4), ('atom trans up', 140, 4, 16),
         ('117 atom proj', 56, 4, 4)]
for n, mt, kt, nt in cases:
    c = f(mt, kt, nt, 2, G, L1)
    d = 'None (declined)' if c is None else (
        f"{type(c).__name__[6:-13]:26s} bw{c.in0_block_w} "
        f"ob{getattr(c, 'out_block_h', None)}x{getattr(c, 'out_block_w', None)} "
        f"sub{c.out_subblock_h}x{c.out_subblock_w}")
    print(f"{n:18s} mt{mt:5d} kt{kt:3d} nt{nt:3d} macs{mt*kt*nt:8d} -> {d}")
