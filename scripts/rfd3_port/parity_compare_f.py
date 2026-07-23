"""Parity gate for the from-PDB featurizer: compare the ported
``tt_bio.rfd3_featurize.featurize`` output against a captured reference ``f``
golden (the ``token_initializer.in_f_<key>.pt`` files produced by
``capture_all.py`` on the vast.ai reference instance) for the SAME PDB+contig.

This is the actual accuracy gate for the featurizer slice. It is NOT a pass
until the per-key PCC / bit-exactness is within the reference noise floor AND
the shape/dtype contract matches. Run it once a binder-``f`` capture exists:

  .venv/bin/python scripts/rfd3_port/parity_compare_f.py <capture_dir> <pdb> <contig>

where <capture_dir> holds token_initializer.in_f_*.pt + a meta.json describing
the input PDB + contig that produced the golden.
"""
import os, sys, glob, json, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from tt_bio.rfd3_featurize import featurize
from tt_bio.rfd3_input import InputSpecification


def load_golden_f(cap_dir):
    f = {}
    for p in glob.glob(os.path.join(cap_dir, "token_initializer.in_f_*.pt")):
        k = os.path.basename(p)[len("token_initializer.in_f_"):-3]
        t = torch.load(p, map_location="cpu", weights_only=True)
        if t.is_floating_point() and t.dtype != torch.float32:
            t = t.float()
        f[k] = t
    return f


def pcc(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    if a.numel() != b.numel():
        return None
    if a.numel() < 2 or float(a.std()) == 0 or float(b.std()) == 0:
        return float((a == b).all()) if a.numel() else 1.0
    a = a - a.mean(); b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp(min=1e-12))


def main(cap_dir, pdb, contig):
    golden = load_golden_f(cap_dir)
    spec = InputSpecification.from_dict({"input": pdb, "contig": contig})
    spec.validate()
    ported = featurize(pdb, spec)

    gkeys = set(golden)
    pkeys = set(ported)
    print(f"golden keys: {len(gkeys)}  ported keys: {len(pkeys)}")
    missing = gkeys - pkeys
    extra = pkeys - gkeys
    if missing:
        print(f"  MISSING in ported (gate fail): {sorted(missing)}")
    if extra:
        print(f"  extra in ported (not in golden): {sorted(extra)}")

    rows = []
    for k in sorted(gkeys & pkeys):
        g, p = golden[k], ported[k]
        shape_ok = (tuple(g.shape) == tuple(p.shape))
        dtype_ok = (str(g.dtype) == str(p.dtype))
        # align shapes if possible (e.g. ref_atom_name_chars may be [L,256] vs [L,4,64])
        if not shape_ok and g.numel() == p.numel():
            p = p.reshape(g.shape)
            shape_ok = True
        if not shape_ok:
            rows.append((k, "SHAPE-MISMATCH", f"{tuple(g.shape)} vs {tuple(p.shape)}", None))
            continue
        if g.dtype != p.dtype:
            rows.append((k, "DTYPE-MISMATCH", f"{g.dtype} vs {p.dtype}", None))
            continue
        bit = bool((g == p).all())
        pc = pcc(g, p)
        verdict = "BIT-EXACT" if bit else (f"PCC={pc:.6f}" if pc is not None else "CONST")
        rows.append((k, verdict, "", bit))

    print(f"\n{'key':40s} {'verdict':14s} {'bit':5s} detail")
    print("-" * 80)
    n_bit = 0
    for k, v, d, bit in rows:
        print(f"{k:40s} {v:14s} {str(bit):5s} {d}")
        if bit:
            n_bit += 1
    print(f"\n{n_bit}/{len(rows)} keys bit-exact; {len(missing)} missing, {len(extra)} extra")
    # gate: every key bit-exact OR (PCC>=0.9999 for float keys)
    ok = (not missing) and all(
        r[1] == "BIT-EXACT" or (r[3] is False and r[1].startswith("PCC") and float(r[1].split("=")[1]) >= 0.9999)
        for r in rows if r[1] not in ("SHAPE-MISMATCH", "DTYPE-MISMATCH")
    ) and not any(r[1] in ("SHAPE-MISMATCH", "DTYPE-MISMATCH") for r in rows)
    print(f"\nFEATURIZER PARITY: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    cap = sys.argv[1]
    meta_path = os.path.join(cap, "meta.json")
    if len(sys.argv) >= 4:
        pdb, contig = sys.argv[2], sys.argv[3]
    elif os.path.exists(meta_path):
        m = json.load(open(meta_path))
        pdb, contig = m["input"], m["contig"]
    else:
        print("usage: parity_compare_f.py <cap_dir> <pdb> <contig>  (or a meta.json in cap_dir)")
        sys.exit(2)
    sys.exit(main(cap, pdb, contig))
