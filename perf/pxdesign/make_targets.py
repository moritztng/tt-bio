"""Build the PXDesign target-size ladder: one protein, one fixed epitope, growing target context.

    python perf/pxdesign/make_targets.py --cif 1DP0.cif --chain A \
        --sizes 128,256,512,768 --binder-length 80 --out-dir perf/pxdesign/targets

Binder-design cost scales with the TARGET, so the ladder has to vary target length and hold
everything else fixed. Holding the epitope fixed is the part that matters: if the hotspots moved
with the crop, each rung would be a different design problem and the ladder would measure two
things at once. So the crop grows outward from one surface patch chosen once, and every rung keeps
the same three hotspot residues and the same binder length.

The hotspots are picked by exposure: per-residue CB neighbour count inside 10 A, lowest count =
most solvent-exposed. The seed hotspot is the most exposed residue in the middle third of the
chain (so the largest crop still fits inside the chain), plus the two most exposed residues within
10 A of it. No PyMOL, no biotite: the CIF ATOM records are enough.
"""

import argparse
import json
import math
import pathlib


def read_chain(cif_path, chain):
    """label_seq_id -> {atom_name: (x, y, z)} for one label_asym_id."""
    res = {}
    for line in pathlib.Path(cif_path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        f = line.split()
        if len(f) < 14 or f[6] != chain:
            continue
        try:
            rid = int(f[8])
            xyz = (float(f[10]), float(f[11]), float(f[12]))
        except ValueError:
            continue
        res.setdefault(rid, {})[f[3]] = xyz
    return res


def rep_atom(atoms):
    for n in ("CB", "CA"):
        if n in atoms:
            return atoms[n]
    return next(iter(atoms.values())) if atoms else None


def pick_hotspots(res, lo, hi, n=3, radius=10.0):
    pts = {r: rep_atom(a) for r, a in res.items()}
    pts = {r: p for r, p in pts.items() if p}
    ids = sorted(pts)

    def neighbours(r):
        p = pts[r]
        return sum(1 for q in ids
                   if q != r and math.dist(p, pts[q]) <= radius)

    exposure = {r: neighbours(r) for r in ids}
    inner = [r for r in ids if lo <= r <= hi]
    seed = min(inner, key=lambda r: (exposure[r], r))
    near = sorted((r for r in ids if r != seed and math.dist(pts[seed], pts[r]) <= radius),
                  key=lambda r: (exposure[r], r))
    return [seed] + near[:n - 1], exposure


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cif", required=True)
    ap.add_argument("--chain", default="A")
    ap.add_argument("--sizes", default="128,256,512,768")
    ap.add_argument("--binder-length", type=int, default=80)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="lacz")
    a = ap.parse_args()

    res = read_chain(a.cif, a.chain)
    ids = sorted(res)
    n_res = len(ids)
    sizes = [int(x) for x in a.sizes.split(",")]
    biggest = max(sizes)
    if biggest > n_res:
        raise SystemExit("chain %s has %d residues, cannot crop %d" % (a.chain, n_res, biggest))

    # the seed hotspot must sit far enough from both termini that the largest crop stays inside
    half = biggest // 2
    lo, hi = ids[0] + half, ids[-1] - half
    hotspots, exposure = pick_hotspots(res, lo, hi)
    centre = sorted(hotspots)[len(hotspots) // 2]

    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cif_dst = pathlib.Path(a.cif).resolve()
    import hashlib
    cif_sha = hashlib.sha256(cif_dst.read_bytes()).hexdigest()

    manifest = {"cif": cif_dst.name, "cif_sha256": cif_sha,
                "cif_source": "https://files.rcsb.org/download/%s" % cif_dst.name.upper(), "chain": a.chain, "chain_residues": n_res,
                "hotspots": hotspots, "hotspot_exposure": {str(h): exposure[h] for h in hotspots},
                "centre_residue": centre, "binder_length": a.binder_length, "rungs": {}}

    for n in sizes:
        start = max(ids[0], centre - n // 2)
        end = start + n - 1
        if end > ids[-1]:
            end, start = ids[-1], ids[-1] - n + 1
        assert all(start <= h <= end for h in hotspots), (n, start, end, hotspots)
        label = "%s_%d" % (a.name, n)
        yaml_path = out / ("%s.yaml" % label)
        yaml_path.write_text(
            "target:\n"
            "  file: \"%s\"\n"
            "  chains:\n"
            "    %s:\n"
            "      crop: [\"%d-%d\"]\n"
            "      hotspots: %s\n"
            "binder_length: %d\n"
            % (cif_dst.resolve(), a.chain, start, end, json.dumps(hotspots), a.binder_length))
        manifest["rungs"][label] = {"yaml": yaml_path.name, "crop": [start, end],
                                    "target_residues": n}
        print("%-14s crop %d-%d (%d res) hotspots %s" % (label, start, end, n, hotspots))

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("chain %s: %d residues, hotspots %s (CB neighbours within 10 A: %s)"
          % (a.chain, n_res, hotspots, [exposure[h] for h in hotspots]))


if __name__ == "__main__":
    main()
