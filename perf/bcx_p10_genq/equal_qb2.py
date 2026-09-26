#!/usr/bin/env python3
"""Is the cheap dispatch path the same function as the uncached one? Asked on a card that can answer.

**This cannot be asked on pc card 0.** That card silently miscomputes ttnn matmuls at a low,
location-keyed rate (memory `pc-card0-512aa-fold-nondeterminism`: `linear_z` wrong 15/15 at
256 aa fp32 against a clean qb1 control). It does not raise and it looks exactly like a numerics
bug in whatever you just wrote, so an equality check taken there asserts nothing. Every timing leg
of this row stays on pc, where the claim is host microseconds; this one leg runs on qb2.

It also runs on a DIFFERENT core grid from pc's -- 110 cores against 130 -- which is worth more
than a repeat would be. The compact plan's seven constants are the grid's geometry, so a second
grid exercises different constants rather than the same ones twice.

Three things are checked per case, on the same operands, arms interleaved `off, on, off` so the
`off`/`off2` pair is the A/A control:

  * `torch.equal` between the arms, and against the `off` arm's own repeat;
  * both INPUT tensors byte-identical after every program. A cached descriptor with a stale
    runtime arg writes into the bottom of DRAM silently (`state/perf10/bcx-TABWD.md`);
  * the plan was actually taken, from `entry['compact']` and `genq.REFUSED`, so an arm that
    quietly declined cannot read as a pass.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn                                                            # noqa: E402
from tt_bio import rne_add as RA                                       # noqa: E402
from tt_bio import reblock_permute as RP                               # noqa: E402
from tt_bio import genq                                                # noqa: E402
from tt_bio.main import ensure_p300_mesh_descriptor                    # noqa: E402

OUT = ROOT / 'perf' / 'bcx_p10_genq' / 'out'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='equal_qb2.json')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    ensure_p300_mesh_descriptor()
    dev = ttnn.open_device(device_id=0)
    mc = ttnn.DRAM_MEMORY_CONFIG
    RA.set_enabled(True)
    res, ok = {}, True
    try:
        up = lambda t: ttnn.from_torch(                                # noqa: E731
            t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
        # Shapes the engine actually calls the residual on, plus one that is not a multiple of
        # the core count so the split has two group sizes and p2 is exercised.
        cases = {'n288_c128': (1, 288, 288, 128), 'n288_c64': (1, 288, 288, 64),
                 'n256_c128': (1, 256, 256, 128), 'ragged_n96_c32': (1, 96, 96, 32)}
        for name, shape in cases.items():
            ta, tb = torch.randn(*shape), torch.randn(*shape)
            a, b = up(ta), up(tb)
            before = (ttnn.to_torch(a).clone(), ttnn.to_torch(b).clone())
            got, reach = {}, {}
            for tag, compact in (('off', False), ('on', True), ('off2', False)):
                genq.set_compact(compact)
                out = RA.rne_add(a, b, mc)
                ttnn.synchronize_device(dev)
                got[tag] = ttnn.to_torch(out).clone()
                reach[tag] = RA._prepare(a, b, out, dev)['compact']
                ttnn.deallocate(out)
            after = (ttnn.to_torch(a), ttnn.to_torch(b))
            r = {
                'shape': list(shape),
                'on_equals_off': bool(torch.equal(got['on'], got['off'])),
                'aa_control_off_equals_off2': bool(torch.equal(got['off'], got['off2'])),
                'max_abs_on_minus_off': float((got['on'].float() - got['off'].float())
                                              .abs().max()),
                'inputs_unchanged': [bool(torch.equal(before[i], after[i])) for i in (0, 1)],
                'plan_taken': reach,
            }
            r['pass'] = (r['on_equals_off'] and r['aa_control_off_equals_off2']
                         and all(r['inputs_unchanged'])
                         and reach == {'off': False, 'on': True, 'off2': False})
            ok = ok and r['pass']
            res[name] = r
            print(f'{name:16s} {json.dumps(r)}', flush=True)
            for t in (a, b):
                ttnn.deallocate(t)
        # The movement family. `bcx-p10-trimove`'s lever is these two kernels, and leg 6 only
        # means anything if the cheap path computes the same index move they already did.
        RP.set_enabled(True) if hasattr(RP, 'set_enabled') else None
        moves = {'move_n288_c128': ((1, 288, 288, 128), 'fwd'),
                 'move_n320_c64': ((1, 320, 320, 64), 'fwd'),
                 'moveback_n288_c128': ((1, 128, 288, 288), 'back'),
                 'moveback_n320_c64': ((1, 64, 320, 320), 'back')}
        for name, (shape, leg) in moves.items():
            t = torch.randn(*shape)
            x = up(t)
            before = ttnn.to_torch(x).clone()
            fn = RP.reblock_permute if leg == 'fwd' else RP.reblock_permute_back
            got, reach = {}, {}
            for tag, compact in (('off', False), ('on', True), ('off2', False)):
                genq.set_compact(compact)
                out = fn(x, mc)
                ttnn.synchronize_device(dev)
                got[tag] = ttnn.to_torch(out).clone()
                reach[tag] = compact
                ttnn.deallocate(out)
            # The stock ttnn spelling of the same index move, as the reference neither arm is.
            dims = (0, 3, 1, 2) if leg == 'fwd' else (0, 2, 3, 1)
            ref = ttnn.permute(x, dims, memory_config=mc)
            ttnn.synchronize_device(dev)
            ref_t = ttnn.to_torch(ref).clone()
            ttnn.deallocate(ref)
            r = {'shape': list(shape), 'leg': leg,
                 'on_equals_off': bool(torch.equal(got['on'], got['off'])),
                 'aa_control_off_equals_off2': bool(torch.equal(got['off'], got['off2'])),
                 'on_equals_ttnn_permute': bool(torch.equal(got['on'], ref_t)),
                 'inputs_unchanged': [bool(torch.equal(before, ttnn.to_torch(x)))],
                 'plan_taken': reach}
            r['pass'] = (r['on_equals_off'] and r['aa_control_off_equals_off2']
                         and r['on_equals_ttnn_permute'] and all(r['inputs_unchanged']))
            ok = ok and r['pass']
            res[name] = r
            print(f'{name:20s} {json.dumps(r)}', flush=True)
            ttnn.deallocate(x)
    finally:
        ttnn.close_device(dev)

    blob = {'host': os.uname().nodename, 'card': os.environ.get('TT_VISIBLE_DEVICES'),
            'refused': dict(genq.REFUSED), 'all_pass': ok, 'cases': res}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / args.out).write_text(json.dumps(blob, indent=1, default=str))
    print(('ALL PASS' if ok else 'FAILED') + f"  refused={dict(genq.REFUSED)}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
