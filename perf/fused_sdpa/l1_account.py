#!/usr/bin/env python3
"""Per-core L1 CB accounting for the fused triangle-attention SDPA, in bytes, no device.

Reproduces the sixteen-CB table `tt_bio/sdpa_generic.build` hands the device (:200-217) with the
persistent mask override `tt_bio/triatt_sdpa.sdpa` applies (:174-178), so a config's L1 cost can be
read and differenced before anything is compiled. Every tile here is bf16 (2048 B): the
intermediate, statistics and scalar CBs are hard-wired bf16 at `sdpa_program_factory.cpp:651-653`
and q/k/v/mask/out are bf16 by the gate's own dtype precondition.

The base is what the device counts on top of the CB high-water mark (kernel binaries, runtime args,
semaphores, dispatch). Anchored on two device refusals, both reported by tt-metal itself:

    S=512  q=512  k=512  kvbf=2   ->  1591808 B   (state/fused-sdpa-adopt.md §2, and re-measured)
    S=768  q=768  k=768  kvbf=2   ->  3115520 B

CB bytes for those two are 1480704 and 3004416, so the base is 111104 B at both, 27.5x apart in
total. It is a constant, not a fraction.

Usage:
    l1_account.py [--sizes 320,512,768,1024] [--kv-buffer-factor 2] [--json out.json]
    l1_account.py --explain 512      # where the one-k-chunk bytes actually go, per CB
"""
import argparse, json

TILE = 32
TILE_B = 2048
# Blackhole p150: 1.5 MiB of L1 per Tensix. The device compares the CB high-water mark against it.
L1_BUDGET = 1572864
BASE = 111104


def div_up(a, b):
    return (a + b - 1) // b


def cb_table(S, q_chunk, k_chunk, DH=32, persistent_mask=True, q_per_core=1, kv_buffer_factor=2):
    Sq_chunk_t, Sk_chunk_t = q_chunk // TILE, k_chunk // TILE
    DHt = vDHt = DH // TILE
    k_num_chunks = div_up(S, k_chunk)
    q_buffer_factor = 2 if q_per_core > 1 else 1
    mask = (k_num_chunks * Sq_chunk_t * Sk_chunk_t if persistent_mask
            else Sq_chunk_t * Sk_chunk_t * 2)
    return {
        'cb0_q': Sq_chunk_t * DHt * q_buffer_factor,
        'cb1_k': Sk_chunk_t * DHt * kv_buffer_factor,
        'cb2_v': Sk_chunk_t * vDHt * kv_buffer_factor,
        'cb3_mask': mask,
        'cb5_scalar': 1,
        'cb7_scalar': 1,
        'cb4_recip_scratch': 1,
        'cb24_qk_im': Sq_chunk_t * Sk_chunk_t,
        'cb25_out_im': Sq_chunk_t * vDHt,
        'cb26_out_accum': Sq_chunk_t * vDHt,
        'cb27_cur_max': Sq_chunk_t,
        'cb28_prev_max': Sq_chunk_t,
        'cb29_cur_sum': Sq_chunk_t,
        'cb30_prev_sum': Sq_chunk_t,
        'cb31_exp_max_diff': Sq_chunk_t,
        'cb16_out': Sq_chunk_t * vDHt,
    }


def report(S, q_chunk, k_chunk, base=BASE, **kw):
    t = cb_table(S, q_chunk, k_chunk, **kw)
    tiles = sum(t.values())
    total = tiles * TILE_B + base
    return {'S': S, 'q_chunk': q_chunk, 'k_chunk': k_chunk,
            'k_num_chunks': div_up(S, k_chunk), 'tiles': t, 'total_tiles': tiles,
            'kv_buffer_factor': kw.get('kv_buffer_factor', 2),
            'cb_bytes': tiles * TILE_B, 'device_total': total,
            'fits': total <= L1_BUDGET, 'over_by': max(0, total - L1_BUDGET),
            'headroom': max(0, L1_BUDGET - total)}


def line(r):
    verdict = 'FITS, %d B spare' % r['headroom'] if r['fits'] else 'OVER by %d' % r['over_by']
    return (f"S={r['S']:5d} q={r['q_chunk']:5d} k={r['k_chunk']:5d} kchunks={r['k_num_chunks']} "
            f"kvbf={r['kv_buffer_factor']} tiles={r['total_tiles']:5d} "
            f"total={r['device_total']:8d}  {verdict}")


def explain(S):
    """Difference the one-k-chunk config against the shipped one, per CB, at the widest q."""
    shipped_k = min(256, S)
    print(f"--- S={S}: where the one-k-chunk bytes go, against the shipped k={shipped_k} ---")
    a = cb_table(S, S, shipped_k)
    b = cb_table(S, S, S)
    c = cb_table(S, S, S, kv_buffer_factor=1)
    print(f"{'CB':22s} {'k=%d' % shipped_k:>9s} {'k=%d' % S:>9s} {'delta B':>10s} "
          f"{'+kvbf=1':>9s}")
    for name in a:
        d = (b[name] - a[name]) * TILE_B
        mark = '  <-- ' if d else ''
        print(f"{name:22s} {a[name]:9d} {b[name]:9d} {d:10d} {c[name]:9d}{mark}")
    for label, t in (('shipped', a), ('one k chunk', b), ('one k chunk, kvbf=1', c)):
        tiles = sum(t.values())
        tot = tiles * TILE_B + BASE
        print(f"{label:22s} tiles={tiles:5d} cb={tiles * TILE_B:8d} total={tot:8d} "
              f"{'FITS, %d B spare' % (L1_BUDGET - tot) if tot <= L1_BUDGET else 'OVER by %d' % (tot - L1_BUDGET)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='320,512,768,1024')
    ap.add_argument('--kv-buffer-factor', type=int, default=None,
                    help='sweep both 2 and 1 when omitted')
    ap.add_argument('--base', type=int, default=BASE)
    ap.add_argument('--explain', type=int, default=None)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    if a.explain:
        explain(a.explain)
        return

    factors = [a.kv_buffer_factor] if a.kv_buffer_factor else [2, 1]
    out = []
    for S in [int(x) for x in a.sizes.split(',')]:
        # every 32-aligned divisor of S, widest first: what `_tri_att_q_chunks` offers against a
        # wide k, since only dividing q_chunks are legal there.
        qs = sorted({S // n for n in range(1, S // TILE + 1)
                     if S % n == 0 and (S // n) % TILE == 0}, reverse=True)
        for kvbf in factors:
            for q_chunk in qs:
                r = report(S, q_chunk, S, base=a.base, kv_buffer_factor=kvbf)
                out.append(r)
                print(line(r))
                if r['fits']:
                    break        # widest q that fits is the one the ladder serves
    if a.json:
        json.dump({'base': a.base, 'budget': L1_BUDGET, 'configs': out},
                  open(a.json, 'w'), indent=1)
        print('wrote', a.json)


if __name__ == '__main__':
    main()
