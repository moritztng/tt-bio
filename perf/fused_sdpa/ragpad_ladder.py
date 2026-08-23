#!/usr/bin/env python3
"""Bit-exactness ladder for the ragged-tile-tail fix.

The claim the fix has to earn is narrow and checkable exactly: `_sdpa_masked` must be a NO-OP at
every sequence length that already divides 32, and must change the answer at every length that does
not. Bit-exactness is the right bar for the aligned half -- an "unchanged to five digits" reading
cannot rule out a regression at 704/736/832/864/928/992, the six padded lengths whose k_chunk
divisor interaction has already produced two real bugs in this path (K3/K5).

Both arms run in one process on freshly uploaded operands, because `ttnn.pad` aliases and writes the
mask into the caller's physical tile tail: sharing one uploaded bias across arms silently masks the
tail for whichever arm runs second.
"""
from __future__ import annotations

import argparse, json, pathlib, sys

import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import ttnn  # noqa: E402
import tt_bio.tenstorrent as T  # noqa: E402

assert pathlib.Path(T.__file__).is_relative_to(REPO), T.__file__


def up(dev, x, dtype=ttnn.bfloat16):
    return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)


def one(dev, S: int, heads: int, dim: int, seed: int, ragpad: bool):
    """One `_tri_att_sdpa` call at length S, with the fix on or off. Returns the fp32 output."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(S, heads, S, dim, generator=g, dtype=torch.float32) * 0.5
    k = torch.randn(S, heads, S, dim, generator=g, dtype=torch.float32) * 0.5
    v = torch.randn(S, heads, S, dim, generator=g, dtype=torch.float32) * 0.5
    # A real triangle-attention bias: mostly well below zero, which is what lets exp(0) on an
    # unmasked padded column win the row. A bias centred on zero hides the defect.
    b = (torch.randn(1, heads, S, S, generator=g, dtype=torch.float32) - 4.0)
    qd, kd, vd, bd = up(dev, q), up(dev, k), up(dev, v), up(dev, b)
    T._SDPA_RAGGED_PAD = ragpad
    o = T._tri_att_sdpa(qd, kd, vd, bd, float(dim) ** 0.5)
    out = ttnn.to_torch(o).float()
    for t in (o,):
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="288,298,320,512,580,704,736,832,864,928,992")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # tt_bio owns the device open (lease enforcement + the pinned-open refusal), so go through it
    # rather than ttnn.open_device: an unpinned open brings up every visible chip.
    dev = T.get_device()
    rows = []
    try:
        for S in [int(x) for x in a.lens.split(",") if x]:
            try:
                off = one(dev, S, a.heads, a.dim, a.seed, False)
                on = one(dev, S, a.heads, a.dim, a.seed, True)
            except Exception as exc:  # noqa: BLE001 -- a length over L1 is a skip, not a result
                rows.append({"S": S, "status": f"skip: {type(exc).__name__}: {str(exc)[:120]}"})
                print(f"S={S:5d} SKIP {type(exc).__name__}")
                continue
            same = bool(torch.equal(off, on))
            d = (on - off).abs()
            rel = float(d.pow(2).sum().sqrt() / off.pow(2).sum().sqrt().clamp_min(1e-30))
            rows.append({"S": S, "status": "ok", "aligned": S % 32 == 0,
                         "bit_exact": same, "max_abs_delta": float(d.max()), "rel_delta": rel,
                         "ragged_calls": T.SDPA_RAGGED_SITES.get("tri_att", [0, 0])[0]})
            tag = "aligned" if S % 32 == 0 else "RAGGED "
            verdict = ("bit-exact, fix is a no-op" if same
                       else f"CHANGED rel={rel:.4e}")
            ok = "PASS" if same == (S % 32 == 0) else "FAIL"
            print(f"S={S:5d} {tag} {ok}  {verdict}")
    finally:
        pass

    good = [r for r in rows if r.get("status") == "ok"]
    aligned_ok = all(r["bit_exact"] for r in good if r["aligned"])
    ragged_ok = all(not r["bit_exact"] for r in good if not r["aligned"])
    print(f"\naligned lengths all bit-exact: {aligned_ok}   "
          f"ragged lengths all changed: {ragged_ok}   (n={len(good)})")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(
            {"heads": a.heads, "dim": a.dim, "seed": a.seed,
             "aligned_all_bit_exact": aligned_ok, "ragged_all_changed": ragged_ok,
             "rows": rows}, indent=2))
        print("wrote", a.out)
    return 0 if (aligned_ok and ragged_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
