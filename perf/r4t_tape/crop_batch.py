"""Crop a dumped upstream batch (`batch<NN>.npz`) to its first N tokens, for a whole-network gradcheck
that does not fit the card at the dump's 256-token padding (the fp32-exact control keeps every float32
pair activation on the tape; batch00 at 256 ran the card out of DRAM).

Every `in/` array axis of the token length is cut, so pair features lose rows and columns together.
Refuses a crop that would drop a real token or leave a token index pointing past the cut. `out/` and
`loss/` (upstream's recorded outputs) are dropped: they belong to the 256-token step; the gradcheck
compares the device against its own float64 rebuild of the cropped step.
"""
import argparse
import numpy as np

INDEX_KEYS = ("tok_idx", "insertion_source_idx", "chiral_frame_atoms", "tetrahedral_vertices")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    z = np.load(a.src)
    pm = z["in/padding_mask"]
    L = pm.shape[1]
    if pm[:, a.n:].any():
        raise SystemExit(f"crop to {a.n} drops real tokens (last real index {np.nonzero(pm.any(0))[0].max()})")
    out = {}
    for k in z.files:
        if k.startswith(("out/", "loss/")):
            continue
        v = z[k]
        if k.startswith("in/"):
            v = v[tuple(slice(0, a.n) if (i > 0 and s == L) else slice(None) for i, s in enumerate(v.shape))]
            if k.rsplit("/", 1)[-1] in INDEX_KEYS and v.size and v.max() >= a.n:
                raise SystemExit(f"{k} indexes past the crop ({v.max()} >= {a.n})")
        out[k] = v
    np.savez(a.dst, **out)
    print(f"{a.src} -> {a.dst}: {L} -> {a.n} tokens, {len(out)} arrays")


if __name__ == "__main__":
    main()
