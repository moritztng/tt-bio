#!/usr/bin/env python3
"""Is the ladder EXACT, or merely close? Compare the MSA module's own output, not a structure.

A structure RMSD cannot separate "this transform is wrong" from "the sampler amplified a last bit",
and the campaign's hard stop is the first of those. So take the thing the ladder actually changes:
`MSA.__call__`'s output `z`, on the FIRST trunk call of a fold, where both arms are fed byte-identical
inputs. Run one arm, abort the fold with a sentinel, run the other, compare the two fp32 readbacks
element by element.

Exact equality here means the padded rows contribute exactly 0.0 and the real rows are summed in the
same order at both rungs, which is the claim the audit made from the source.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


class Done(Exception):
    pass


def main() -> int:
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T
    import tt_baseline as B

    B.RECYCLING_STEPS = B._resolve_recycling_steps(None, "boltz2")
    snap = list(sys.path)
    sys.path.insert(0, str(ROOT / "perf" / "other512"))
    from fold_ab_multi import patch_boltz2_cfg
    sys.path[:] = snap
    patch_boltz2_cfg()
    T.get_device()
    fix = ROOT / "perf" / "size512" / "fixtures"
    one_fold, meta, _state = B.build_fold(
        "boltz2", HERE / f".msa_{size}", fix / f"cdk2x2_{size}.yaml", fix / f"cdk2x2_{size}.a3m")

    mod = T.MSA
    o_mod = mod.__dict__["__call__"]
    box: dict = {}

    # POISON. The padded depth rows are filled with garbage instead of zeros. If the answer does
    # not move, the padded region provably cannot reach a real token -- which is what makes the
    # rung change a reassociation of the SAME terms rather than a different computation. It is the
    # acceptance test token_axis.pad_poison already defines for the token axis, on the depth axis.
    # At 512 and 298 tokens the token pad is 0, so the only 4-D pad with a non-zero last pair is
    # the MSA depth one; the counter below asserts exactly one call was poisoned per forward.
    o_pad = torch.nn.functional.pad
    poison = {"on": False, "n": 0}

    def _pad(x, pad, *a, **k):
        out = o_pad(x, pad, *a, **k)
        if poison["on"] and len(pad) == 6 and pad[5] > 0 and out.dim() == 4:
            out[:, x.shape[1]:, :, :] = 7.5
            poison["n"] += 1
        return out

    torch.nn.functional.pad = _pad

    def w_mod(self_obj, *a, **k):
        box["depth"] = int(a[1].shape[1])
        out = o_mod(self_obj, *a, **k)
        box["z"] = ttnn.to_torch(out).float().clone()
        raise Done

    mod.__call__ = w_mod
    grabs = {}
    try:
        for arm, pois in (("0", False), ("1", False), ("0", False), ("0", True)):
            os.environ["TT_BIO_MSA_LADDER"] = arm
            poison["on"], poison["n"] = pois, 0
            box.clear()
            try:
                one_fold()
            except Done:
                pass
            except Exception as exc:                      # the sentinel may be wrapped
                if not isinstance(getattr(exc, "__cause__", None), Done) and "z" not in box:
                    raise
            key = f"{arm}p" if pois else arm
            if pois:
                assert poison["n"] == 1, f"poisoned {poison['n']} pads, expected exactly 1"
            grabs.setdefault(key, []).append((box["depth"], box["z"]))
            print(f"  arm {key}: padded depth {box['depth']}, z {tuple(box['z'].shape)}"
                  + (f", poisoned {poison['n']} pad" if pois else ""), flush=True)
    finally:
        mod.__call__ = o_mod
        torch.nn.functional.pad = o_pad
        os.environ.pop("TT_BIO_MSA_LADDER", None)

    d_off, z_off = grabs["0"][0]
    d_off2, z_off2 = grabs["0"][1]
    d_on, z_on = grabs["1"][0]
    res = {
        "size": size, "depth_off": d_off, "depth_on": d_on,
        "AA_equal": bool(torch.equal(z_off, z_off2)),
        "AA_max_abs": float((z_off - z_off2).abs().max()),
        "AB_equal": bool(torch.equal(z_off, z_on)),
        "AB_max_abs": float((z_off - z_on).abs().max()),
        "AB_mean_abs": float((z_off - z_on).abs().mean()),
        "z_max_abs": float(z_off.abs().max()),
        "negative_control_must_be_False": bool(torch.equal(z_off, z_off + 1)),
    }
    d_p, z_p = grabs["0p"][0]
    res["poison_depth"] = d_p
    res["poison_equal"] = bool(torch.equal(z_off, z_p))
    res["poison_max_abs"] = float((z_off - z_p).abs().max())
    (HERE / f"z_parity_{size}.json").write_text(json.dumps(res, indent=2) + "\n")
    for k, v in res.items():
        print(f"  {k:32s} {v}", flush=True)
    T.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
