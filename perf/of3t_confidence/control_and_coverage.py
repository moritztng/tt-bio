"""PROTOCOL SS3e (negative control) and SS6 (coverage) for the confidence head.

SS3e. A gate nobody has watched fail is not a gate. Two perturbations, each applied to
EXACTLY ONE parameter's device gradient, and the requirement is that the check moves on
that tensor and on no other, to the last digit printed.

  "Which check fails if our model is replaced by zeros?" -- this one, and the zeros arm
  below runs it: with every device gradient zeroed, the relative L2 is 1.0 on every
  tensor, because ||0 - ref|| / ||ref|| = 1. A check that could not tell zeros from the
  model would read 0 there.

SS6. Coverage is demonstrated, not argued. For each confidence term, the term's logits
are seeded ALONE and the gradient that arrives at si_trunk and zij_trunk is measured. A
non-zero norm there is the term's gradient contribution reaching the trunk, which is the
half of SS6 that the host round-trip made impossible. The other half, a non-zero weight,
is read from `of3_loss_weights` -- upstream's own per-(stage, dataset) table -- so the
two halves are reported together per (stage, dataset) rather than assumed from each other.
"""
import os, pickle, sys, torch, ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forward_vs_float64 import ca_walk
from grad_device import HEADS, rel_l2, record_uploads, walk, their_name, S
from tt_bio import autograd as ag
from tt_bio.tenstorrent import get_device
from tt_bio.openfold3_confidence import OF3ConfidenceHead
from tt_bio.train.losses import of3_loss_weights, OF3_LOSS_OVERRIDES, OF3_CROP_TOKENS

TERM_LOGIT = {"plddt": "plddt_logits", "resolved": "experimentally_resolved_logits",
              "pae": "pae_logits", "pde": "pde_logits", "distogram": "distogram_logits"}


def main():
    sd = torch.load(os.path.expanduser("~/of3-weights/of3-p2-155k.pt"), map_location="cpu",
                    weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    aux = {k[len("aux_heads."):]: v for k, v in sd.items() if k.startswith("aux_heads.")}
    g = pickle.load(open(os.path.expanduser("~/of3_ref_out.pkl"), "rb"))["intermediates"]
    si_input = g["input_embedder_real"]["out"][0].float()
    si_trunk, zij_trunk = (t.float() for t in g["pairformer_stack_real"]["out"])
    N = si_trunk.shape[0]
    repr_x = ca_walk(N)
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
        packer_l1_acc=True)
    reg, restore = record_uploads()
    head = OF3ConfidenceHead(aux, dev, ckc)
    restore()
    up = lambda x, dt=ttnn.bfloat16: ttnn.from_torch(
        x.float().unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
    si_d, zt_d = up(si_input), up(zij_trunk)
    oh_d = head.distance_onehot(repr_x)
    head.forward_device(si_d, up(si_trunk, ttnn.float32), zt_d, oh_d)

    params = {}
    for key, dev_t in list(head._wd_cache.items()):
        name, transposed, _dt = key
        leaf = ag.parameter(dev_t)
        head._wd_cache[key] = leaf
        params[name] = (leaf, (lambda x: x.t()) if transposed else (lambda x: x), "aux",
                        None, name)
    for path, owner, attr, t in walk(head.pf):
        rec = reg.get(id(t))
        if rec is None:
            continue
        weights, key, src, transform, _ = rec
        probe = transform(src)
        if probe.shape == src.t().shape and torch.equal(probe, src.t()):
            inv = lambda x: x.t()
        elif probe.shape == src.shape and torch.equal(probe, src):
            inv = lambda x: x
        else:
            continue
        tn = their_name(path, key)
        if tn is None:
            continue
        leaf = ag.parameter(t)
        setattr(owner, attr, leaf)
        params[tn[3]] = (leaf, inv, tn[0], tn[1], tn[2])

    rg, rb = torch.load(f"{S}/ref_aux_grad.pt"), torch.load(f"{S}/ref_blk_grad.pt")
    seed = torch.load(f"{S}/seed.pt")["seed"]
    dn = lambda t: torch.Tensor(ttnn.to_torch(t)).double()

    def run(only=None):
        """One taped forward+backward. ``only`` seeds a single head. Returns the leaves."""
        for leaf, *_ in params.values():
            leaf.grad = None
        sti = ag.Tensor(up(si_trunk, ttnn.float32), requires_grad=True)
        zti = ag.Tensor(up(zij_trunk), requires_grad=True)
        with ag.tape():
            out = head.forward_device(ag.Tensor(si_d), sti, zti, ag.Tensor(oh_d))
        roots, seeds = [], []
        for k in (HEADS if only is None else [only]):
            o = out[k]
            sv = seed[k].reshape(tuple(int(d) for d in o.shape))
            roots.append(o)
            seeds.append(ttnn.from_torch(sv.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                                         dtype=ttnn.bfloat16))
        ag.backward(roots, seeds)
        return sti, zti

    def table(scale=None, target=None, zero=False):
        out = {}
        for their, (leaf, inv, where, bi, lookup) in params.items():
            if leaf.grad is None:
                continue
            gd = inv(dn(leaf.grad))
            if zero:
                gd = torch.zeros_like(gd)
            elif scale is not None and their == target:
                gd = gd * scale
            ref = rb[bi].get(lookup) if where == "blk" else rg.get(lookup)
            if ref is None or tuple(gd.shape) != tuple(ref.shape):
                continue
            out[their] = rel_l2(gd, ref)
        return out

    run()
    base = table()
    # A tensor comfortably inside the bar, so a 1 % nudge is visible rather than buried.
    target = min(base, key=lambda k: base[k])
    print(f"SS3e negative control. Target: {target}\n  baseline {base[target]:.6e}")
    for pct, sc in ((1, 1.01), (10, 1.10)):
        t2 = table(scale=sc, target=target)
        moved = [k for k in base if abs(t2[k] - base[k]) > 1e-12 * max(1.0, base[k])]
        over = [k for k in t2 if t2[k] > 5e-2 and base[k] <= 5e-2]
        print(f"  +{pct:2d}%  {target} -> {t2[target]:.6e}   tensors that moved: {len(moved)} "
              f"({'only the target' if moved == [target] else moved[:3]})"
              f"   newly over the 5.0e-02 bar: {over if over else 'none'}")
    z = table(zero=True)
    bad = [k for k, v in z.items() if abs(v - 1.0) > 1e-9]
    print(f"  zeros arm: {len(z)} tensors, all read 1.0" if not bad
          else f"  zeros arm: {len(bad)} tensors did NOT read 1.0 -- {bad[:3]}")

    print("\nSS6 coverage. Gradient reaching the trunk, per confidence term, seeded alone:")
    print(f"  {'term':12s} {'||dL/d si_trunk||':>19s} {'||dL/d zij_trunk||':>20s}")
    reach = {}
    for term, logit in TERM_LOGIT.items():
        sti, zti = run(only=logit)
        a = float(dn(sti.grad).norm()) if sti.grad is not None else 0.0
        b = float(dn(zti.grad).norm()) if zti.grad is not None else 0.0
        reach[term] = (a, b)
        print(f"  {term:12s} {a:19.6g} {b:20.6g}")
    print("\n  weight x reach, per (stage, dataset), from upstream's own table:")
    for stage, tbl in OF3_LOSS_OVERRIDES.items():
        for ds in sorted(tbl):
            w = of3_loss_weights(stage, ds)
            fired = [t for t in TERM_LOGIT
                     if w.get(t, 0.0) != 0.0 and max(reach[t]) > 0.0]
            off = [t for t in TERM_LOGIT if w.get(t, 0.0) == 0.0]
            print(f"    {stage:18s} {ds:26s} crop {OF3_CROP_TOKENS[stage]:3d}  "
                  f"fires {sorted(fired)}  weight-zero {sorted(off)}")


if __name__ == "__main__":
    main()
