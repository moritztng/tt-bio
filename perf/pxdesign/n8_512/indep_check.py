"""Are the eight N_sample=8 backbones eight independent designs, or one design eight times?

Moritz's rule: a batched design counts only if each slot is an independent, production-valid
design that clears the same bar as batch 1. PXDesign's N_sample is documented as backbones out of
ONE trajectory, so this is not rhetorical.

Two questions, answered separately:
  distinctness  pairwise CA RMSD across the binder chain of the eight generated backbones, with no
                superposition beyond the shared target frame (every sample is written in the target's
                own frame, so the raw CA distance is the honest comparison)
  quality       every per-design metric AF2-IG reports, N=8 against the N=1 arm's own designs
"""
import glob, itertools, json, math, os, re, sys, csv, statistics as st

def ca_coords(path):
    """Binder CA coordinates, in file order. mmCIF written by write_design_cifs, binder alone or
    binder+target; take the chain with the fewest residues, which is the 80-residue binder."""
    rows = []
    with open(path) as f:
        inloop, cols = False, {}
        for line in f:
            s = line.strip()
            if s.startswith("_atom_site."):
                cols[s.split(".")[1]] = len(cols); inloop = True; continue
            if inloop and (s.startswith("#") or s.startswith("loop_") or s.startswith("_")):
                if rows: break
                continue
            if inloop and s and not s.startswith("#"):
                p = s.split()
                if len(p) < len(cols): break
                rows.append(p)
    if not rows:
        return {}
    g = lambda p, k: p[cols[k]]
    per = {}
    for p in rows:
        if g(p, "label_atom_id") != "CA":
            continue
        ch = g(p, "label_asym_id")
        per.setdefault(ch, []).append((float(g(p, "Cartn_x")), float(g(p, "Cartn_y")),
                                       float(g(p, "Cartn_z"))))
    return per

def rmsd(a, b):
    n = min(len(a), len(b))
    return math.sqrt(sum((a[i][0]-b[i][0])**2 + (a[i][1]-b[i][1])**2 + (a[i][2]-b[i][2])**2
                         for i in range(n)) / n)

out = {}
for rep in sorted(glob.glob("/work/out/laczc512_prev_n8_b2_rep*")):
    cifs = sorted(glob.glob(rep + "/global_run_0/laczc_512/seed_*/predictions/*_sample_*.cif"),
                  key=lambda p: int(re.search(r"sample_(\d+)", p).group(1)))
    if not cifs:
        continue
    sets = []
    for c in cifs:
        per = ca_coords(c)
        if not per:
            continue
        ch = min(per, key=lambda k: len(per[k]))
        sets.append((os.path.basename(c), per[ch], {k: len(v) for k, v in per.items()}))
    pw = [(a[0], b[0], round(rmsd(a[1], b[1]), 3)) for a, b in itertools.combinations(sets, 2)]
    ident = [x for x in pw if x[2] < 0.01]
    out[os.path.basename(rep)] = {
        "n_backbones": len(sets),
        "chain_sizes": sets[0][2] if sets else None,
        "binder_ca": len(sets[0][1]) if sets else None,
        "pairwise_ca_rmsd_min": min(x[2] for x in pw) if pw else None,
        "pairwise_ca_rmsd_median": round(st.median([x[2] for x in pw]), 3) if pw else None,
        "pairwise_ca_rmsd_max": max(x[2] for x in pw) if pw else None,
        "identical_pairs": ident,
        "n_pairs": len(pw),
    }

def summaries(pat):
    rows = []
    for f in sorted(glob.glob(pat + "/design_outputs/laczc_512/summary.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                r["_rep"] = f.split("/")[3]
                rows.append(r)
    return rows

METRICS = ["af2_plddt", "af2_ptm", "af2_iptm", "af2_ipAE", "af2_monomer_plddt", "af2_monomer_ptm",
           "af2_binder_pred_design_rmsd", "af2_complex_pred_design_rmsd", "Rg"]
qual = {}
for arm, pat in (("n1", "/work/out/laczc512_prev_n1_b2_rep*"),
                 ("n8", "/work/out/laczc512_prev_n8_b2_rep*")):
    rows = summaries(pat)
    d = {"n_designs": len(rows),
         "n_distinct_sequences": len({r["sequence"] for r in rows}),
         "af2ig_success": sum(r["AF2-IG-success"] == "True" for r in rows),
         "af2ig_easy_success": sum(r["AF2-IG-easy-success"] == "True" for r in rows)}
    for m in METRICS:
        v = [float(r[m]) for r in rows if r.get(m) not in (None, "")]
        if v:
            d[m] = {"min": min(v), "median": round(st.median(v), 4), "max": max(v)}
    qual[arm] = d

print(json.dumps({"distinctness": out, "quality": qual}, indent=1))
