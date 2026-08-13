"""Is `_apply_template_host` a pure no-op when delta is zero, or does its slice+repad zero the pad?

With `template_mask.sum() == 0` the TemplateModule returns an exactly zero delta (measured, 4/4
recycles), so the host round trip's only remaining effect is structural: it takes z_rec, slices off
the padded rows/cols, adds zero, and re-pads with zeros. Skipping the call is therefore bit-exact
IFF the padded region of z_rec is already zero when the call is made.

If it is not, skipping would leave whatever the previous op left in the pad, which is a real
difference even if downstream masks it -- so this is checked rather than assumed.
"""
from __future__ import annotations
import argparse, json, os, shutil, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="perf/dsfix/fixtures/bg_R3.yaml")
    ap.add_argument("--out", default="perf/bgdeep/template_pad.json")
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import ttnn

    REC = {"calls": []}
    cls = next(o for _, o in vars(T).items()
               if isinstance(o, type) and "_apply_template_host" in vars(o))
    _orig = cls._apply_template_host

    def probe(self, z_rec, st):
        seq_len, seq_pad = st["seq_len"], st["seq_pad"]
        z_full = self._to_torch(z_rec)                      # the PADDED tensor, as it arrives
        core = z_full[:, :seq_len, :seq_len, :]
        rebuilt = torch.nn.functional.pad(core, (0, 0, 0, seq_pad, 0, seq_pad)) if seq_pad else core
        pad_region_nonzero = int(torch.count_nonzero(z_full).item()
                                 - torch.count_nonzero(core).item())
        REC["calls"].append({
            "seq_len": seq_len, "seq_pad": seq_pad,
            "padded_shape": tuple(z_full.shape),
            "pad_region_nonzero_elems": pad_region_nonzero,
            "pad_region_absmax": float((z_full - rebuilt).abs().max().item()),
            "skip_is_bit_exact": bool(torch.equal(z_full, rebuilt)),
        })
        print("  " + json.dumps(REC["calls"][-1]), flush=True)
        return _orig(self, z_rec, st)

    cls._apply_template_host = probe

    workdir = Path.home() / "bgpad_out"
    if workdir.exists():
        shutil.rmtree(workdir)
    argv = ["run", a.spec, "--output", str(workdir), "--num_designs", "1",
            "--protocol", "protein-anything", "--steps", "design",
            "--device_ids", os.environ.get("TT_VISIBLE_DEVICES", "0")]
    from tt_bio.main import _run_boltzgen_cli
    try:
        _run_boltzgen_cli("tt-bio design", argv)
    except SystemExit as e:
        if e.code not in (None, 0):
            raise

    Path(a.out).write_text(json.dumps(REC, indent=1))
    ok = all(c["skip_is_bit_exact"] for c in REC["calls"]) if REC["calls"] else False
    print(f"RESULT calls={len(REC['calls'])} skip_is_bit_exact={ok}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
