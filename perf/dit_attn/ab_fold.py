#!/usr/bin/env python3
"""One 298 aa fold arm: wall clock, and the output coordinates for the parity check.

Reuses ``scripts/gpu_vs_tt/tt_baseline.build_fold`` so this fold goes through exactly
the same path as the committed baseline numbers -- production config, MSA cache seeded,
one cold fold absorbing kernel compile, then N warm folds. The extra thing it does is
keep the CIF the last warm fold wrote and dump its atom coordinates to .npy, so the
before and after arms can be compared in float64 without re-running anything.

    TT_MESH_GRAPH_DESC_PATH=<ttnn>/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:perfwar-dit-attention-fusion PYTHONPATH=$PWD \
      python3 perf/dit_attn/ab_fold.py --model protenix-v2 --repeat 3 --tag after \
        --out perf/dit_attn/fold_protenix-v2_after.json
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))
from tt_baseline import build_fold  # noqa: E402

TARGETS = {
    "protenix-v2": ("examples/prot300.yaml", "scripts/gpu_vs_tt/fixtures/prot300.a3m"),
    "opendde": ("examples/prot300.yaml", "scripts/gpu_vs_tt/fixtures/prot300.a3m"),
}


def cif_coords(path: Path) -> np.ndarray:
    """(n_atom, 3) float64 from an mmCIF _atom_site loop, in file order."""
    cols, rows, in_loop = [], [], False
    for line in path.read_text().splitlines():
        s = line.strip()
        if s.startswith("_atom_site."):
            cols.append(s.split(".", 1)[1].split()[0])
            in_loop = True
            continue
        if in_loop:
            if not s or s.startswith("#") or s.startswith("loop_") or s.startswith("_"):
                if rows:
                    break
                in_loop = False
                cols = []
                continue
            rows.append(s.split())
    ix, iy, iz = (cols.index(c) for c in ("Cartn_x", "Cartn_y", "Cartn_z"))
    return np.array([[float(r[ix]), float(r[iy]), float(r[iz])] for r in rows], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(TARGETS))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="298", choices=["117", "298"],
                    help="117 aa is the small-N non-regression check: core_grid on a 2-tile-wide "
                         "output is exactly the kind of change that can invert at small N")
    a = ap.parse_args()

    target, a3m = (ROOT / p for p in TARGETS[a.model])
    if a.size == "117":
        target = ROOT / "examples/prot.yaml"
        a3m = ROOT / "scripts/gpu_vs_tt/fixtures/prot117.a3m"
    msa_dir = ROOT / "perf" / "dit_attn" / "msa_cache"
    one_fold, meta, _state = build_fold(a.model, msa_dir, target, a3m)

    cold_s, cold_m = one_fold()
    assert cold_m.get("msa"), "fold ran without an MSA -- cache seeding failed"
    times, plddt = [], None
    for i in range(a.repeat):
        t, m = one_fold()
        times.append(t)
        plddt = m["plddt"]
        print(f"[{a.tag}/{a.model}] warm fold {i + 1}/{a.repeat}: {t:.3f} s "
              f"plddt={plddt:.4f}", flush=True)

    cifs = sorted(Path(meta["struct_dir"]).glob("*.cif"))
    assert len(cifs) == 1, [p.name for p in cifs]
    xyz = cif_coords(cifs[0])
    npy = Path(a.out).with_suffix(".npy")
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy, xyz)

    res = dict(model=a.model, tag=a.tag, host="tt-quietbox2",
               card=meta.get("card_type"), aiclk_mhz=meta.get("aiclk_mhz"),
               n_tokens=cold_m.get("n_tokens"), n_residues=cold_m.get("n_residues"),
               cold_s=round(cold_s, 3), load_s=meta["load_s"],
               warm_s=[round(t, 3) for t in times],
               median_s=round(st.median(times), 3), min_s=round(min(times), 3),
               plddt=plddt, n_atoms=int(xyz.shape[0]), coords_npy=str(npy),
               stamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)

    # Close the card, then leave without running interpreter shutdown: the spawn worker
    # pool and the device-lease flock deadlock each other at exit and the process sits in
    # locks_lock_inode_wait forever, holding the card against the next arm.
    try:
        from tt_bio.tenstorrent import cleanup
        cleanup()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
