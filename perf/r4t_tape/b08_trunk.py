"""D43: where does the trunk single repr go wrong on batch08's unindexed (insertion) tokens?

Runs glue -> production device trunk (grad off, the trunk_step route) -> host heads on every candidate of a
dumped batch, and the same glue inputs through the float64 reference trunk (tests/rfd4_reference.trunk on the
real `nn.` weights). Reports rel_l2 of the single repr and of the insertion logits on insertion rows vs the
rest, and checks the reference chain against upstream's recorded fp32 insertion logits first.

    TT_VISIBLE_DEVICES=<card> python3 perf/r4t_tape/b08_trunk.py --batch ~/r4t/step/batch08.npz --out DIR
TT_BIO_LEVERS=none gives the C arm; TT_BIO_SDPA_OWN_FORWARD=1 as trunk_step's P/C both use.
"""
import argparse
import os
import sys

import numpy as np


def rel(a, b, rows=None):
    a, b = a.double(), b.double()
    if rows is not None:
        a, b = a[rows], b[rows]
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lpr", default=os.path.expanduser("~/r4t/tape/b08/lpr"))
    ap.add_argument("--no-ref", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    import torch
    import ttnn
    from tt_bio import tenstorrent as T, autograd as ag
    from tt_bio.rfd4 import network as N, engine as E, trunk as TR
    sys.path.insert(0, a.lpr)
    from loss_parity_reference import load_batch
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "..", "tests"))
    import rfd4_reference as REF

    dev = T.get_device()
    state, config = N.load_checkpoint(E._resolve_weights(E.DEFAULT_WEIGHTS))
    cfg = config["nn"]
    glue = N.FlowGlue(state, config)
    trunk = E.load_trunk(glue, state, cfg)
    levers = trunk._levers
    print("levers", sorted(levers), flush=True)

    def put(t, dtype=ttnn.float32):
        return ttnn.from_torch(t.to(torch.float32).contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT,
                               device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    bt, meta, rec, _ = load_batch(a.batch)
    depth = max(1, int(meta["depth"]))
    with torch.no_grad():
        cs = [glue.conditioning(bt, None)]
        for cyc in range(1, depth):
            cs.append(glue.conditioning(bt, depth - cyc - 1) if glue.condition_on_recycle_remaining else cs[0])
        s0, z0 = glue.initial_states(bt)
    mask = bt["padding_mask"].float()
    ins = bt["insertion_source_idx"]
    print("depth", depth, "s0", tuple(s0.shape), "z0", tuple(z0.shape), "padding", mask.sum(1).tolist(), flush=True)
    print("insertion_source_idx distinct", torch.unique(ins).tolist()[:12], flush=True)

    mk = TR.trunk_masks(mask, dev, dtype=trunk.dtype, residual_dtype=trunk.residual_dtype)
    ag.install()
    with T.lever_scope(levers), ag.tape():
        with ag.no_grad():
            sk, zk = trunk(put(s0), put(z0, trunk.pair_dtype(mk)), [put(c, trunk.dtype) for c in cs], mk)
            sdev = ttnn.to_torch(getattr(sk, "value", sk)).float().reshape(s0.shape)
            zdev = ttnn.to_torch(getattr(zk, "value", zk)).float().reshape(z0.shape)

    def heads(s):
        with torch.no_grad():
            o = glue.heads(s.float(), bt)
        return o["insertion"]["logits"].double()

    ldev = heads(sdev)
    torch.save(dict(sdev=sdev, s0=s0, mask=mask, ins=ins, ldev=ldev), os.path.join(a.out, "dev.pt"))
    lrec = torch.as_tensor(rec["insertion/logits"]).double()
    print("dev insertion logits vs upstream recorded: all", rel(ldev, lrec), flush=True)
    if a.no_ref:
        return
    w = {k[3:]: v.double() for k, v in state.items() if k.startswith("nn.")}
    srefs, zrefs = [], []
    with torch.no_grad():
        for b in range(s0.shape[0]):
            sl = slice(b, b + 1)
            sr, zr = REF.trunk(w, s0[sl].double(), z0[sl].double(), [c[sl].double() for c in cs], mask[sl].double(),
                               cfg["nlayers"], every_n=cfg["update_pair_repr_every_n"], n_heads=cfg["nheads"])
            srefs.append(sr)
            zrefs.append(zr)
            print("ref cand", b, "done", flush=True)
    sref = torch.cat(srefs)
    lref = heads(sref)
    torch.save(dict(sref=sref, lref=lref), os.path.join(a.out, "ref.pt"))
    print("REF CHAIN: ref insertion logits vs upstream recorded:", rel(lref, lrec), flush=True)
    for b in range(s0.shape[0]):
        valid = mask[b].bool()
        insr = (ins[b] >= 0) & valid   # assumption printed above: unindexed rows carry a source idx >= 0
        rest = (~insr) & valid
        zr = zrefs[b][0]
        print(f"cand {b}: n_ins {int(insr.sum())} n_rest {int(rest.sum())} | s rel ins {rel(sdev[b], sref[b], insr):.3e} "
              f"rest {rel(sdev[b], sref[b], rest):.3e} | z rel {rel(zdev[b][valid][:, valid], zr[valid][:, valid]):.3e} "
              f"| ins-logit rel ins {rel(ldev[b], lref[b], insr):.3e} rest {rel(ldev[b], lref[b], rest):.3e} "
              f"| |s| ins {float(sref[b][insr].norm(dim=-1).mean()) if insr.any() else 0:.2f} rest {float(sref[b][rest].norm(dim=-1).mean()):.2f}",
              flush=True)


if __name__ == "__main__":
    main()
