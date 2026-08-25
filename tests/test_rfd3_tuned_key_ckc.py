"""`_TUNED_MM_CACHE`'s key has to carry the compute-kernel config.

`_calibrate_linear` proves a program config bit-exact *against the default call under one
compute-kernel config*. The key was shapes, dtypes, bias and core grid, so a caller that passed a
second fidelity would have been handed a config pinned under the first. Latent rather than live --
every RFD3 call site passes the one `_default_compute_kernel_config()` -- but
`build_token_initializer` takes a `compute_kernel_config` from outside, which is the hole.

Host-only: keying is arithmetic. `WormholeComputeKernelConfig` constructs without a device.
"""
from __future__ import annotations

import ttnn

from tt_bio.rfd3.model import _ckc_key


def ckc(fidelity=ttnn.MathFidelity.HiFi4, fp32=True, packer=True):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fidelity, math_approx_mode=False,
                                            fp32_dest_acc_en=fp32, packer_l1_acc=packer)


def test_the_config_object_itself_is_not_usable_as_a_key():
    """Why `_ckc_key` reads the fields: the ttnn object hashes and reprs by identity."""
    a, b = ckc(), ckc()
    assert a is not b
    assert hash(a) != hash(b) and repr(a) != repr(b)


def test_field_equal_configs_key_the_same():
    assert _ckc_key(ckc()) == _ckc_key(ckc())


def test_a_different_fidelity_keys_differently():
    assert _ckc_key(ckc()) != _ckc_key(ckc(fidelity=ttnn.MathFidelity.LoFi))
    assert _ckc_key(ckc()) != _ckc_key(ckc(fp32=False))
    assert _ckc_key(ckc()) != _ckc_key(ckc(packer=False))


def test_no_config_is_its_own_key():
    assert _ckc_key(None) is None
