import ttnn
import tt_bio.tenstorrent as T
T._l1_bank_bytes = lambda: 1461760   # documented BH allocator per-bank number; on-card confirm owed
print('grid', T.COMPUTE_GRID_MAIN, 'cores', T.COMPUTE_GRID_MAIN[0]*T.COMPUTE_GRID_MAIN[1])
def tiles(xs, ws):
    b = 1
    for d in xs[:-2]: b *= d
    return b * -(-xs[-2]//32), -(-xs[-1]//32), -(-ws[-1]//32)
sites = {'pairbias 512': ((1,512,512,256),(256,16)), 'pwa 512': ((512,512,256),(256,1)),
         'template 512': ((1,512,512,256),(256,64)),
         'pairbias 298': ((1,298,320,256),(256,16)), 'pwa 298': ((298,320,256),(256,1)),
         'template 298': ((1,298,320,256),(256,64))}
for name,(xs,ws) in sites.items():
    m,k,n = tiles(xs, ws)
    print(f'== {name}: m_tiles={m} k_tiles={k} n_tiles={n}')
    for cap in (1,2,4,8,16):
        bw = max((d for d in (k,8,4,2,1) if d <= cap and k % d == 0), default=1)
        c = T._pair_proj_program_config(m,k,n,bw,2,False)
        if c is None:
            print(f'   cap{cap:<3d} bw={bw} -> None (REFUSED)')
        else:
            need = (2*bw*(c.out_block_h+c.out_block_w)*2048 + c.out_block_h*c.out_block_w*(2048+4096) + 131072)
            cores = -(-m//c.per_core_M)
            print(f'   cap{cap:<3d} bw={bw} pcM={c.per_core_M} obh={c.out_block_h} obw={c.out_block_w} '
                  f'sh={c.out_subblock_h} sw={c.out_subblock_w} pcN={c.per_core_N} '
                  f'cores={cores}/{T.COMPUTE_GRID_MAIN[0]*T.COMPUTE_GRID_MAIN[1]} L1need={need}B ({need/1461760:.2f}x bank)')
