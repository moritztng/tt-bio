#!/usr/bin/env python3
"""Does the shipped distogram head symmetrize twice?

`DistogramHead.__call__` does `ttnn.add(z, permute(z))` itself, and
`tests/test_esmfold2.py::test_distogram_head` feeds it a RAW z and compares against
`ref(z + z.transpose(-2, -3))`. The production call site is the vendored reference at
`modeling_esmfold2.py:992`, which passes `z + z.transpose(-2, -3)` into the adapter.

If both readings are right the port computes Linear(2*(z + zT)) where the reference computes
Linear(z + zT). This measures it instead of asserting it: same weights, same z, three arms.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))



def main():
    import torch
    import tt_bio.tenstorrent as T
    import tt_bio.esmfold2 as tt_ef2
    assert Path(T.__file__).resolve().is_relative_to(ROOT), "tt_bio from %s" % T.__file__

    T.get_device()
    torch.manual_seed(0)
    L = 64
    z = torch.randn(1, L, L, 256)
    z_sym = z + z.transpose(-2, -3)

    # tests/esmfold2_reference.py::make_distogram_head, inlined: it imports a checkout that
    # is not on this host, and the head it builds is one seeded nn.Linear(256, 64).
    torch.manual_seed(0)
    ref = torch.nn.Linear(256, 64).eval()
    ref_out = ref(z_sym)                       # what the model is supposed to produce

    mod = tt_ef2.DistogramHeadModel()
    mod.load_state_dict(ref.state_dict(), strict=False)

    port_raw = mod(z)                           # the unit test's calling convention
    port_prod = mod(z_sym)                      # the production call site's convention

    def pcc(a, b):
        a, b = a.flatten().double(), b.flatten().double()
        a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm()))

    res = {
        "L": L,
        "pcc_unit_test_convention": round(pcc(port_raw, ref_out), 6),
        "pcc_production_convention": round(pcc(port_prod, ref_out), 6),
        "max_abs_unit_test": round(float((port_raw - ref_out).abs().max()), 6),
        "max_abs_production": round(float((port_prod - ref_out).abs().max()), 6),
        "ref_out_absmean": round(float(ref_out.abs().mean()), 6),
        # If the port doubles its input, Linear(2x) = 2*Linear(x) - bias, so this reconstructs it.
        "max_abs_production_vs_doubled_model":
            round(float((port_prod - (2 * ref_out - ref.bias)).abs().max()), 6),
    }
    print(json.dumps(res, indent=1))
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
