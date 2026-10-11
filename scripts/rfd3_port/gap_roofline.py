#!/usr/bin/env python3
import json, sys
d = json.load(open('/home/moritz/.coworker/wt/spd-gap-rfd3/perf/p63/traffic_census_lever.json'))
MB = 1e-3  # MB -> GB
P_BW, A_BW, A_TF = 448.0, 2039.0, 312.0
def base():
    out = []
    for o in d['ops']:
        reg = 'DiT' if o['n'] >= 36 or 'DiT' in o['label'] or 'pad x3' in o['label'] else o['region']
        out.append(dict(reg=reg, l=o['label'], G=o['gflop'], R=o['read_MB']*MB, W=o['write_MB']*MB, n=o['n']))
    return out
def drop(ops, keys): return [o for o in ops if not any(k in o['l'] for k in keys)]
SPARSE_ATOM = ['sparse pair bias', 'softmax fp32 [1,4,LP,128]', 'typecast around the softmax', 'scores + PV matmuls']
MASK = 2*4*6080*6080/1e9          # bf16 [4,6080,6080] = 0.296 GB (model.py _ATOM_SDPA_MAX_MASK_BYTES test)
QKV = 4*6080*32*2/1e9
ATOM_G = 4*4*6080*6080*32/1e9     # 18.93 GFLOP / SDPA call over the padded dense key axis
def atom_dense(ops):
    ops = drop(ops, SPARSE_ATOM)
    ops.append(dict(reg='atom', l='atom SDPA mask build (generic_op) x6', G=0, R=6*4*6051*128*2/1e9, W=6*MASK, n=6))
    ops.append(dict(reg='atom', l='atom SDPA dense 6080 keys x9', G=9*ATOM_G, R=9*(MASK+3*QKV), W=9*QKV, n=9))
    return ops
PB = dict(reg='DiT', l='fused pair-bias linear 128->576 x2', G=142.22, R=2*0.1235, W=2*0.5555, n=2)
def normal():
    ops = drop(base(), ['pair_bias linear', 'add(pair_bias', 'typecast bias', 'typecast scores', 'add scores + bias'])
    ops += [PB, dict(reg='DiT', l='fused scores+bias fp32 (generic_op) x36', G=0, R=2*0.571, W=1.142, n=36)]
    return atom_dense(ops)
def fast():
    ops = drop(base(), ['pair_bias linear', 'add(pair_bias', 'typecast bias', 'typecast scores', 'add scores + bias',
                        'permute pair_bias', 'matmul scores', 'softmax fp32', 'typecast attn', 'attn@vv',
                        'concat [z|d', 'slice PAIR', 'rms_norm(258)', 'process_z linear', 'distogram+self'])
    for o in ops:   # RFD3_TRANSITION_B8: pair activations bfp8 (1088 B per 1024-elem tile), residual add stays bf16
        if ('transition_2' in o['l'] or 'z_transition' in o['l']) and 'residual add' not in o['l']:
            o['R'] *= 1088/2048; o['W'] *= 1088/2048
    ops += [PB,
            dict(reg='token', l='process_z collapsed: table lookup + z-half add x2', G=0, R=2*(0.0077+0.247), W=2*2*0.1235, n=2),
            dict(reg='DiT', l='stacked DiT SDPA masks (permute+mask add) x2', G=0, R=2*0.5555, W=2*0.571, n=2),
            dict(reg='DiT', l='DiT SDPA x36', G=2*27.41, R=36*(16*704*704*2+3*704*768*2)/1e9, W=36*704*768*2/1e9, n=36)]
    return atom_dense(ops)
def physics():
    U = 0.1235; ops = []
    ops.append(dict(reg='token', l='process_z (collapsed)', G=0, R=2*U, W=2*U))
    for h, k in ((256, 4), (512, 4)):
        ops += [dict(reg='token', l=f'pair transition H={h}', G=3*2*482240*128*h/1e9, R=U, W=U) for _ in range(k)]
    ops += [dict(reg='token', l='pf to_b', G=15.8/4, R=U, W=685*704*16*2/1e9) for _ in range(4)]
    ops += [dict(reg='DiT', l='pair bias all blocks', G=142.22/2, R=U+0.5555, W=0.5555) for _ in range(2)]
    ops += [dict(reg='DiT', l='DiT SDPA', G=2*27.41/36, R=3*704*768*2/1e9, W=704*768*2/1e9) for _ in range(36)]
    ops.append(dict(reg='DiT', l='DiT linears + weights', G=358.76, R=0.552+0.986, W=0.467))
    ops += [dict(reg='atom', l='atom attn sparse 128', G=4*4*6051*128*32/1e9, R=4*6051*128*2/1e9+3*QKV, W=QKV) for _ in range(9)]
    ops.append(dict(reg='atom', l='atom linears + weights', G=27.49+13.75, R=0.215+0.107+0.072, W=0.322))
    return ops
def roof(ops, tf):
    tp = ta = bwp = 0; byr = {}
    for o in ops:
        B = o['R'] + o['W']
        mp, bp = o['G']/tf, B/P_BW*1e3       # ms (GFLOP/TF = ms)
        ma, ba = o['G']/A_TF, B/A_BW*1e3
        tp += max(mp, bp); ta += max(ma, ba)
        if bp >= mp: bwp += bp
        byr.setdefault(o['reg'], [0, 0, 0]); r = byr[o['reg']]; r[0] += B; r[1] += o['G']; r[2] += max(mp, bp)
    return dict(G=sum(o['G'] for o in ops), B=sum(o['R']+o['W'] for o in ops), p150a_ms=tp, a100_ms=ta,
                ratio=tp/ta, bw_share_of_roof=bwp/tp, by_region={k: [round(x, 2) for x in v] for k, v in byr.items()})
for name, f in (('normal', normal), ('fast', fast), ('physics', physics)):
    for tf in (119.0, 205.0):
        r = roof(f(), tf); print(name, tf, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
