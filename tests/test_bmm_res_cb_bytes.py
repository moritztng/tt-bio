"""`bmm_res.cb_bytes` prices the circular buffers `trimul_bres` plans, so an L1 neighbour can be decided up front.

On a Wormhole Galaxy chip at 512 aa the trimul's L1 `a` operand (TT_BIO_OF3L1_TMA) was refused by the device once
per trimul direction and process, because the gate reserved 640 KiB per core and bres plans ~1.2 MB. The two
refusals quote the CB region's end (.103, OpenFold3 --fast 512, spd-galaxy of3hang 10-18); both sit one fixed base
above `cb_bytes`, and `a` was allocated at 1032192.

Run: python3 -m pytest tests/test_bmm_res_cb_bytes.py   (needs ttnn importable, no device)
"""
import ttnn

from tt_bio import bmm_res

WH_L1 = 1499136                  # the refusal's L1 size
WH_BANK = 1395424                # unreserved bytes per core on a Wormhole 8x9 grid
WH_BASE = WH_L1 - WH_BANK
A_AT = 1032192                   # where the L1 `a` [1, 64, 512, 512] bf16 landed, 72 cores


def test_cb_bytes_reproduce_the_devices_refusals():
    b16 = ttnn.bfloat16
    assert WH_BASE + bmm_res.cb_bytes(16, 16, b16, b16, False) == 1283360
    assert WH_BASE + bmm_res.cb_bytes(16, 16, b16, b16, True) == 1316128


def test_a_operand_cannot_sit_beside_bres_at_512():
    a_per_core = WH_L1 - A_AT
    assert a_per_core * 72 >= 64 * 512 * 512 * 2
    assert a_per_core > WH_BANK - bmm_res.cb_bytes(16, 16, ttnn.bfloat16, ttnn.bfloat16, True)


def test_declines_without_fp32_dest():
    ckc = ttnn.WormholeComputeKernelConfig(fp32_dest_acc_en=False)
    assert bmm_res.cb_bytes(16, 16, ttnn.bfloat16, ttnn.bfloat16, False, ckc) == 0
