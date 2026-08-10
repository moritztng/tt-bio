#!/usr/bin/env python3
"""E1 parity — main-as-reference coords PCC. The integration envelope needs cached CPU refs
(BLOCKED-REF-REGEN), so instead fold both arms on device with the same seed and compare
the final atom coords PCC. A non-bit-exact change that preserves the structure lands at
PCC ~1.0 vs the unedited main. One arm per process; saves coords to a .pt for the
comparator.
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--target", type=Path, default=ROOT / "examples" / "prot300.yaml")
    ap.add_argument("--a3m", type=Path, default=ROOT / "scripts/gpu_vs_tt/fixtures/prot300.a3m")
    ap.add_argument("--msa-dir", type=Path, default=ROOT / ".msa_d1par")
    ap.add_argument("--out", type=Path, required=True, help="coords .pt output")
    a = ap.parse_args()

    import torch
    import tt_baseline as B
    one_fold, meta, state = B.build_fold("protenix-v2", a.msa_dir, a.target, a.a3m, hoist=True)

    captured = {}
    orig_fold = state.model.fold

    def wrapped(*args, **kw):
        out = orig_fold(*args, **kw)
        captured["coords"] = out[0]
        return out
    state.model.fold = wrapped

    cold_t, cold_m = one_fold()  # cold compile
    assert cold_m.get("msa"), "no MSA"
    t, m = one_fold()  # warm fold, coords captured
    coords = captured["coords"]
    if not isinstance(coords, torch.Tensor):
        coords = torch.as_tensor(coords)
    coords = coords.detach().cpu().float()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arm": a.arm, "wall_s": round(t, 3), "plddt": m.get("plddt"),
                "coords": coords, "shape": list(coords.shape)}, a.out)
    print(f"[{a.arm}] wall={t:.3f}s plddt={m.get('plddt')} coords_shape={list(coords.shape)} "
          f"coords_mean={coords.mean().item():.4f} coords_std={coords.std().item():.4f}",
          flush=True)
    state.reset()
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
