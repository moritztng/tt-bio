#!/usr/bin/env python3
"""Reach of the lnbw residual fold, counted on the real block: for every lnbw result, the state
of the gradient it is added to, and what consumed that value's gradient next (closure round_add,
a flush to float32 by a later contribution, or a leaf/other read)."""
import collections, json, pathlib, sys
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import ttnn
    from tt_bio import bindcraft2, autograd, lnbw
    from tt_bio.af2 import af2_pair_masks
    tr = bindcraft2._Trunk(pathlib.Path("/home/ttuser/bcx_e2e/af2_params/params_model_1_multimer_v3.npz"))
    ag, T = tr.ag, tr.taped
    n, nreal, depth = 288, 261, 2
    torch.manual_seed(0)
    m0, z0 = torch.randn(depth, n, 256) * 0.5, torch.randn(n, n, 128) * 0.5
    seq = torch.zeros(n); seq[:nreal] = 1
    msa_mask = tr.up(seq.expand(depth, n).contiguous())
    pm = af2_pair_masks(seq[:, None] * seq[None, :], tr.device)
    blk = tr.model.device_evoformer[0]
    lnout = {}            # id(result tensor) -> tag
    watch = {}            # id(Tensor) -> state when the lnbw result arrived
    stat = collections.Counter()
    orig_ln, orig_add = lnbw.layer_norm_bw, autograd.Tensor.add_grad
    orig_cg, orig_fl = autograd.Tensor.closure_grad, autograd.Tensor._flush_pend

    def ln(x, g, gamma, eps):
        out = orig_ln(x, g, gamma, eps)
        if active[0]:
            lnout[id(out)] = [int(d) for d in x.shape]
        return out

    def add(self, grad):
        if id(grad) in lnout and active[0]:
            shp = lnout.pop(id(grad))
            st = ("none" if self._grad is None else str(self._grad.dtype).split(".")[-1]) + \
                 ("+pend" if self._pend is not None else "") + ("+parts" if self._parts else "") + \
                 ("+slab" if self._slab is not None else "")
            watch[id(self)] = (st, str(shp))
            stat["arrive " + st + " " + str(shp)] += 1
        return orig_add(self, grad)

    def cg(self):
        w = watch.pop(id(self), None)
        if w is not None:
            stat[f"consumed-by closure_grad ({'round_add' if self._pend is not None else 'no pend'}) {w}"] += 1
        return orig_cg(self)

    def fl(self):
        w = watch.pop(id(self), None)
        if w is not None:
            stat[f"consumed-by flush (later contribution) {w}"] += 1
        return orig_fl(self)

    lnbw.layer_norm_bw, autograd.Tensor.add_grad = ln, add
    autograd.Tensor.closure_grad, autograd.Tensor._flush_pend = cg, fl
    active = [False]
    with bindcraft2.fast_round():
        for rep in range(2):
            active[0] = bool(rep)
            ml, zl = tr.leaf(m0), tr.leaf(z0)
            with T.tape():
                m, z = ag.checkpoint(lambda x, y: blk(x, y, msa_mask, *pm), ml, zl)
            tr.sync()
            seeds = [tr.seed(torch.randn(m0.shape) * 1e-3, m), tr.seed(torch.randn(z0.shape) * 1e-3, z)]
            ag.backward([m, z], seeds)
            tr.sync()
            ag.release_pins()
            if rep:
                for k, w in watch.items():
                    stat[f"never consumed in block (leaf / block input) {w}"] += 1
    stat["lnbw REACH"] = dict(lnbw.REACH)
    print(json.dumps(stat, indent=1, default=str))


if __name__ == "__main__":
    main()
