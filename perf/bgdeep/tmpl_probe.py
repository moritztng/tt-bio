"""Is the 3.455 s/design host template round-trip computing anything?

`_apply_template_host` runs BoltzGen's TemplateModule once per recycling iteration via an 85 MB
device->host->device round trip. Its own comment says the module "is called unconditionally every
iteration (no has_templates gate)". `_build_static` separately computes
`has_templates = template_recycle is not None and template_mask is not None and template_mask.any()`,
and the host path is taken exactly when that is False.

So: does bg_R3 carry any templates at all, and if not, is `delta` identically zero? If it is, gating
the call on `template_mask.any()` is bit-exact by construction and worth the whole 3.455 s/design.
If delta is non-zero with no templates, the module is contributing a real learned bias and the gate
would change results -- which is a NO-GO, not a bug to work around.

Makes no perf claim; it decides whether a gate is legitimate.
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
    ap.add_argument("--num-designs", type=int, default=1)
    ap.add_argument("--out", default="perf/bgdeep/template_noop.json")
    a = ap.parse_args()

    import tt_bio.tenstorrent as T

    REC = {"calls": []}
    cls = None
    for name, obj in vars(T).items():
        if isinstance(obj, type) and "_apply_template_host" in vars(obj):
            cls = obj
            print(f"hooking {name}._apply_template_host", flush=True)
            break
    if cls is None:
        print("NOT FOUND", flush=True)
        return 1

    _orig = cls._apply_template_host

    def probe(self, z_rec, st):
        seq_len, seq_pad = st["seq_len"], st["seq_pad"]
        z_torch = self._to_torch(z_rec)[:, :seq_len, :seq_len, :]
        with torch.no_grad():
            delta = self.template_module_torch(
                z_torch, st["feats"], st["pair_mask_unpad"], use_kernels=self.use_kernels
            )
        tm = st["feats"].get("template_mask")
        REC["calls"].append({
            "has_templates_flag": bool(st.get("has_templates")),
            "template_recycle_is_none": self.template_recycle is None,
            "template_mask_present": tm is not None,
            "template_mask_any": (bool(tm.any().item()) if tm is not None else None),
            "template_mask_sum": (float(tm.sum().item()) if tm is not None else None),
            "delta_shape": tuple(delta.shape),
            "delta_absmax": float(delta.abs().max().item()),
            "delta_absmean": float(delta.abs().mean().item()),
            "delta_all_zero": bool(torch.count_nonzero(delta).item() == 0),
            "z_absmax": float(z_torch.abs().max().item()),
        })
        print("  " + json.dumps(REC["calls"][-1]), flush=True)
        z_new = z_torch + delta
        if seq_pad:
            z_new = torch.nn.functional.pad(z_new, (0, 0, 0, seq_pad, 0, seq_pad))
        z_out = self._from_torch(z_new)
        import ttnn
        ttnn.deallocate(z_rec)
        return z_out

    cls._apply_template_host = probe

    workdir = Path.home() / "bgtmpl_out"
    if workdir.exists():
        shutil.rmtree(workdir)
    argv = ["run", a.spec, "--output", str(workdir), "--num_designs", str(a.num_designs),
            "--protocol", "protein-anything", "--steps", "design",
            "--device_ids", os.environ.get("TT_VISIBLE_DEVICES", "0")]
    from tt_bio.main import _run_boltzgen_cli
    try:
        _run_boltzgen_cli("tt-bio design", argv)
    except SystemExit as e:
        if e.code not in (None, 0):
            raise

    Path(a.out).write_text(json.dumps(REC, indent=1))
    n = len(REC["calls"])
    allzero = all(c["delta_all_zero"] for c in REC["calls"]) if n else False
    print(f"RESULT calls={n} all_delta_zero={allzero}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
