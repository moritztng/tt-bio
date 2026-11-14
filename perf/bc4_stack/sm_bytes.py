#!/usr/bin/env python3
"""Device bytes the on-card structure module holds while the Evoformer's tape is live, per token axis.

What BindCraft 2's round carries between the Evoformer forward and its backward: the fp32 pair copy, the
key bias, the dropout keeps, the stack's tape, the backward's transient peak, and the pair cotangent it
hands the Evoformer. Weights are real (the fit is about shapes, not values), activations random.

usage: TT_VISIBLE_DEVICES=<c> sm_bytes.py --params <af2 npz> --sizes 224,352,512,672 [--out json]
"""
import argparse, json, sys

import numpy as np


def main(a):
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import af2_structure as S, autograd as A, tenstorrent as T
    raw = np.load(a.params)
    k = S.pack({n: np.asarray(raw[n], np.float64) for n in raw.files if n.startswith(S.ROOT)})
    dev = T.get_device()
    o = S.Card(dev)
    kd = o.upload(k)
    mv = lambda: (lambda v: int(v.total_bytes_allocated_per_bank) * int(v.num_banks))(  # noqa: E731
        ttnn.get_memory_view(dev, ttnn.BufferType.DRAM))
    peak = [0]
    add_grad = A.Tensor.add_grad
    def sampled(self, g):
        add_grad(self, g); peak[0] = max(peak[0], mv())
    A.Tensor.add_grad = sampled
    rng, rows = np.random.default_rng(0), []
    for n in a.sizes:
        ttnn.synchronize_device(dev); base = mv()
        real = n - 20
        mask = np.r_[np.ones(real), np.zeros(n - real)]
        pair16 = ttnn.from_torch(__import__("torch").as_tensor(rng.standard_normal((n, n, S.CZ), np.float32)),
                                 dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        base16 = mv()
        z = A.Tensor(ttnn.typecast(pair16, ttnn.float32), requires_grad=True)
        act0 = o.put(rng.standard_normal((n, S.CS)), grad=True)
        kb = o.put(S.key_bias(mask))
        drops = [(A.Tensor(o.raw(np.ones((n, S.CS)))), A.Tensor(o.raw(np.ones((n, S.CS))))) for _ in range(S.LAYERS)]
        ttnn.synchronize_device(dev); inputs = mv()
        act, traj = S.stack(o, kd, act0, z, kb, drops)
        ttnn.synchronize_device(dev); taped = mv()
        seeds = [o.raw(rng.standard_normal(tuple(int(d) for d in x.shape))) for x in (act, traj[-1][0], traj[-1][1])]
        peak[0] = mv()
        A.backward([act, traj[-1][0], traj[-1][1]], seeds)
        ttnn.synchronize_device(dev); after = mv()
        zg = z.grad
        del act, traj, seeds, drops, kb, act0, z
        ttnn.synchronize_device(dev); held = mv()
        r = dict(n=n, inputs=inputs - base16, tape=taped - inputs, bw_peak=peak[0] - base16,
                 after_bw=after - base16, pair_grad_only=held - base16, pair16=base16 - base)
        rows.append(r); print(json.dumps(r), flush=True)
        del zg, pair16
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=[224, 352, 512, 672])
    ap.add_argument("--out")
    main(ap.parse_args())
