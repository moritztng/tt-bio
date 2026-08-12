#!/usr/bin/env python3
"""Does pre-warming the refiner make the fold stop depending on the expander's assembly?

The refiner's first call returns something different from every call after it, on byte-identical
input and regardless of how the input buffer was built (A != B = C = D, 4.1 % relative on 97.6 % of
z_ref). The refiner also holds 26 device tensors before that call and 146 after, so most of its
weights are uploaded lazily during it -- the first call runs interleaved with its own weight
uploads, and every opendde fold takes exactly that path.

If that interleaving is what makes the fold sensitive to ambient state, then calling the refiner
once and throwing the result away should remove the sensitivity: run the real fold on the SECOND
call, with everything warm. Two folds, one per expander branch, same seed:

    both hashes equal   -> confirmed, and warming is a candidate fix
    still different     -> the first-call effect is real but is not the mechanism

    TT_BIO_OPENDDE_HOST_ZSTRUCT=1 python3 scripts/probe_refiner_warmup.py predict ...
"""
import os
import sys

import ttnn

MARK = os.environ.get("TT_BIO_WARM_MARK", "/tmp/refiner_warmup.txt")


def _mark(msg):
    with open(MARK, "a") as fh:
        fh.write(msg + "\n")


def _install():
    import tt_bio.opendde as od
    from tt_bio.tenstorrent import get_device
    orig_call = od.OpenDDE.expand_and_refine

    def wrapped(self, ifd, s_inputs_res, s_res, z_res, *, extra_attn_bias=True,
                return_attn_bias=False):
        dev = get_device()
        s_inputs_st, s_st, z_st, attn_bias = self.expander(ifd, s_inputs_res, s_res, z_res)
        Ns, c_z, c_s = s_st.shape[0], self.expander.c_z, self.expander.c_s
        h_z = ttnn.to_torch(ttnn.reshape(z_st, (1, Ns, Ns, c_z)))
        h_s = ttnn.to_torch(ttnn.reshape(s_st, (1, Ns, c_s)))
        h_b = ttnn.to_torch(ttnn.reshape(attn_bias, (1, 1, Ns, Ns))) if extra_attn_bias else None

        # The expander's own copies are dead once they are on the host, and at 1902 tokens
        # holding them alongside the warm-up's and the real call's costs 2.8 GiB each.
        ttnn.deallocate(z_st)

        def fresh(t):
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

        # Warm-up call, result discarded. Its only job is to materialise the lazily uploaded
        # weights so the call that counts runs with everything already resident.
        _wz = fresh(h_z)
        w_s, w_z = self.refiner(fresh(h_s), _wz,
                                extra_attn_bias=(fresh(h_b) if h_b is not None else None))
        ttnn.deallocate(w_s)
        ttnn.deallocate(w_z)
        _mark("[WARM] discarded warm-up refiner call")

        s_ref, z_ref = self.refiner(fresh(h_s), fresh(h_z),
                                    extra_attn_bias=(fresh(h_b) if h_b is not None else None))
        result = (s_inputs_st, ttnn.reshape(s_ref, (Ns, c_s)), z_ref)
        return (*result, attn_bias) if return_attn_bias else result

    od.OpenDDE.expand_and_refine = wrapped


from tt_bio.main import cli  # noqa: E402

_install()

if __name__ == "__main__":
    sys.exit(cli(standalone_mode=True))
