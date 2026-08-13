"""Bit-exactness of the template no-op gate, checked in the live fold against the path it replaces.

Inside every gated call, compute BOTH the device mask multiply and the original host round trip on
the same input `z_rec`, and `torch.equal` them. This is the right bar per §5 bar 1: the change is
claimed bit-exact, so it is checked at the tensor, not at the design (BoltzGen is not bit-stable
end to end -- the A/A structural floor is 2.0-7.2 A).
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
    ap.add_argument("--out", default="perf/bgdeep/template_noop_parity.json")
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import ttnn

    REC = {"calls": []}
    _noop = T.TrunkModule._apply_template_noop
    _host = T.TrunkModule._apply_template_host

    def probe(self, z_rec, st):
        # host arm first, from a copy, so the original z_rec survives for the device arm
        z_copy = ttnn.clone(z_rec)
        z_host = _host(self, z_copy, st)          # consumes z_copy
        t_host = self._to_torch(z_host)
        ttnn.deallocate(z_host)

        z_dev = _noop(self, z_rec, st)            # consumes z_rec
        t_dev = self._to_torch(z_dev)

        REC["calls"].append({
            "shape": tuple(t_dev.shape),
            "equal": bool(torch.equal(t_dev, t_host)),
            "maxdiff": float((t_dev.float() - t_host.float()).abs().max().item()),
            "nonzero_dev": int(torch.count_nonzero(t_dev).item()),
            "nonzero_host": int(torch.count_nonzero(t_host).item()),
        })
        print("  " + json.dumps(REC["calls"][-1]), flush=True)
        return z_dev

    T.TrunkModule._apply_template_noop = probe

    workdir = Path.home() / "bgnoop_out"
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
    ok = bool(REC["calls"]) and all(c["equal"] for c in REC["calls"])
    print(f"RESULT calls={len(REC['calls'])} all_bit_exact={ok}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
