#!/usr/bin/env python3
"""Is the structural-token refiner a pure function of its input tensor's VALUES?

Swapping `ttnn.concat` for `from_torch(torch.cat(...))` in the expander moves the fold by 7.89 A,
and the two tensors those produce are identical: same values, same padding under a contraction over
the padded axis (0 mismatch in 115.8 M elements at 1902 tokens), same dtype, layout, padded shape
and memory config. Every isolated variable -- the host read, its timing, chunk free order, a global
address shift, a targeted reallocate -- is byte-neutral on its own.

That leaves one possibility worth testing directly: that a ttnn op's result depends on how its
input buffer was produced and not only on what is in it. This runs the refiner twice in one
process, on two tensors that are byte-identical by construction -- the expander's own output, and a
`from_torch(to_torch(...))` round trip of it -- and compares the two results.

    differ  -> the refiner is not a pure function of its input values, which explains the whole
               five-pass paradox and is a ttnn-level finding, not an expander one
    equal   -> the refiner is pure, and the fold difference enters somewhere after it

Run on a small target; the assembly branch is irrelevant here because both inputs are built inside
the probe.

    TT_VISIBLE_DEVICES=26 python3 scripts/probe_refiner_purity.py predict \
      examples/abag_xm/9ncy.yaml --model opendde-abag ...
"""
import os
import sys

import torch
import ttnn

SINK = os.environ.get("TT_BIO_PURITY_MARK", "/tmp/refiner_purity.txt")


def _mark(msg):
    with open(SINK, "a") as fh:
        fh.write(msg + "\n")


def _cmp(name, a, b):
    same = torch.equal(a, b)
    line = f"[PURITY] {name} shape={tuple(a.shape)} equal={same}"
    if not same:
        d = (a.float() - b.float()).abs()
        ref = a.float().abs().max().item() or 1.0
        line += (f" max|diff|={d.max().item():.6g} rel={d.max().item() / ref:.3e}"
                 f" mismatch={(d > 0).sum().item()}/{d.numel()}")
    _mark(line)
    return same


def _install():
    import tt_bio.opendde as od
    from tt_bio.tenstorrent import get_device
    orig = od.OpenDDE.expand_and_refine

    def wrapped(self, ifd, s_inputs_res, s_res, z_res, *, extra_attn_bias=True,
                return_attn_bias=False):
        dev = get_device()
        s_inputs_st, s_st, z_st, attn_bias = self.expander(ifd, s_inputs_res, s_res, z_res)
        Ns, c_z, c_s = s_st.shape[0], self.expander.c_z, self.expander.c_s

        # Host copies, so the second pass starts from exactly the same values. The refiner
        # updates s and z in place (PairformerLayer uses ttnn.add_), so it cannot be run twice
        # on the same tensors.
        h_z = ttnn.to_torch(ttnn.reshape(z_st, (1, Ns, Ns, c_z)))
        h_s = ttnn.to_torch(ttnn.reshape(s_st, (1, Ns, c_s)))
        h_b = ttnn.to_torch(ttnn.reshape(attn_bias, (1, 1, Ns, Ns))) if extra_attn_bias else None

        def run(z4, tag):
            s3 = ttnn.from_torch(h_s, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
            bias = (ttnn.from_torch(h_b, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
                    if h_b is not None else None)
            s_ref, z_ref = self.refiner(s3, z4, extra_attn_bias=bias)
            out = (ttnn.to_torch(s_ref), ttnn.to_torch(z_ref))
            _mark(f"[PURITY] ran refiner on {tag}")
            return out

        # Snapshot every device tensor the refiner holds. If the first call differs from every
        # later one, the simplest explanation is that it writes to its own weights, and then
        # A is the clean result and B/C/D are the corrupted ones.
        def _weights(obj, seen=None, depth=0):
            seen = seen if seen is not None else set()
            if depth > 4 or id(obj) in seen:
                return
            seen.add(id(obj))
            d = getattr(obj, "__dict__", None)
            if not isinstance(d, dict):
                return
            for k, v in list(d.items()):
                if isinstance(v, ttnn.Tensor):
                    yield f"{type(obj).__name__}.{k}", v
                elif isinstance(v, (list, tuple)):
                    for i, e in enumerate(v):
                        if isinstance(e, ttnn.Tensor):
                            yield f"{type(obj).__name__}.{k}[{i}]", e
                        else:
                            yield from _weights(e, seen, depth + 1)
                elif isinstance(v, dict):
                    for kk, e in v.items():
                        if isinstance(e, ttnn.Tensor):
                            yield f"{type(obj).__name__}.{k}[{kk}]", e
                else:
                    yield from _weights(v, seen, depth + 1)

        pre = {}
        for nm, t in _weights(self.refiner):
            try:
                pre[nm] = ttnn.to_torch(t).clone()
            except Exception:
                pass
        _mark(f"[PURITY] snapshotted {len(pre)} refiner tensors before the first call")

        # A: the expander's own tensor, built by ttnn.concat.
        zA = ttnn.reshape(z_st, (1, Ns, Ns, c_z))
        sA, zrA = run(zA, "A = expander output (ttnn.concat)")

        changed = []
        for nm, t in _weights(self.refiner):
            if nm in pre:
                try:
                    if not torch.equal(pre[nm], ttnn.to_torch(t)):
                        changed.append(nm)
                except Exception:
                    pass
        _mark(f"[PURITY] refiner tensors changed by the first call: {len(changed)} "
              f"{changed[:8]}")

        # B and C: byte-identical values, buffers from from_torch. Two of them, because A-vs-B
        # alone cannot tell "the result depends on how the buffer was built" from "the result
        # depends on how many times the refiner has been called" -- a lazily built weight cache
        # or an in-place write to a weight would both make the FIRST call the odd one out.
        zB = ttnn.from_torch(h_z, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        _cmp("input z A vs B", h_z, ttnn.to_torch(zB))
        sB, zrB = run(zB, "B = from_torch(to_torch(A))")

        zC = ttnn.from_torch(h_z, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        sC, zrC = run(zC, "C = a second from_torch copy, identical to B")

        # D: concat provenance again, but run LAST. B==C already shows repeated calls agree, so
        # this separates the two survivors: if D matches A the difference is how the buffer was
        # built, if D matches B/C it is that A was simply the first call.
        k = Ns // 2
        zD = ttnn.concat(
            [ttnn.from_torch(h_z[:, :k].contiguous(), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16),
             ttnn.from_torch(h_z[:, k:].contiguous(), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16)], dim=1)
        _cmp("input z A vs D", h_z, ttnn.to_torch(zD))
        sD, zrD = run(zD, "D = ttnn.concat provenance, run last")

        _cmp("A vs B  s_ref", sA, sB)
        _cmp("A vs B  z_ref", zrA, zrB)
        _cmp("B vs C  s_ref", sB, sC)      # same provenance, differs only in call order
        _cmp("B vs C  z_ref", zrB, zrC)
        _cmp("A vs D  z_ref", zrA, zrD)    # same provenance, different call order
        _cmp("B vs D  z_ref", zrB, zrD)    # different provenance, both late calls

        # Carry on with A so the fold behaves as it normally would.
        zfin = ttnn.from_torch(h_z, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        s3 = ttnn.from_torch(h_s, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        bias = (ttnn.from_torch(h_b, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
                if h_b is not None else None)
        s_ref, z_ref = self.refiner(s3, zfin, extra_attn_bias=bias)
        result = (s_inputs_st, ttnn.reshape(s_ref, (Ns, c_s)), z_ref)
        return (*result, attn_bias) if return_attn_bias else result

    od.OpenDDE.expand_and_refine = wrapped


from tt_bio.main import cli  # noqa: E402

_install()

if __name__ == "__main__":
    sys.exit(cli(standalone_mode=True))
