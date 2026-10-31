"""Mol-1 TT vs upstream on one mol1-ref golden with the SAME sampler draws: each TT sample replays upstream's recorded
init/rot/trans/eps stream and is graded against upstream's matching sample, so the deviation is the port's alone.
upstream's own sample-to-sample spread (different draws, same trunk) is printed beside it as the seed floor.
TT is fed upstream's features.pt (its ref_pos conformers included) over the worker's featurization of the same job.
Usage: TT_VISIBLE_DEVICES=<card> python perf/mol1_smoke/shared_draw.py SEED_DIR PRED_DIR CASES_JSON CKPT N_CYCLE OUT_JSON
   or: python perf/mol1_smoke/shared_draw.py --pack SEED_DIR PRED_DIR CASES_JSON BUNDLE_PT   (CPU: everything a remote
       grade needs in one file, a few % of the goldens' size)
       TT_VISIBLE_DEVICES=<card> python perf/mol1_smoke/shared_draw.py --bundle BUNDLE_PT CKPT N_CYCLE OUT_JSON"""
import glob
import itertools
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

from tt_bio.main import ensure_p300_mesh_descriptor

ensure_p300_mesh_descriptor()
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts" / "mol1_io")]
import compare_features as cf  # noqa: E402
from perf.clocksample import during  # noqa: E402
from tt_bio import weights  # noqa: E402
from tt_bio.protenix import Protenix  # noqa: E402


def cif_atoms(path):
    cols, xyz, names, hetero = [], [], [], []
    for line in open(path):
        if line.startswith("_atom_site."):
            cols.append(line.strip().split(".", 1)[1])
        elif line.startswith(("ATOM", "HETATM")) and cols:
            r = dict(zip(cols, line.split()))
            xyz.append([float(r["Cartn_x"]), float(r["Cartn_y"]), float(r["Cartn_z"])])
            names.append(r["label_atom_id"])
            hetero.append(line.startswith("HETATM"))
    return torch.tensor(xyz, dtype=torch.float64), names, torch.tensor(hetero)


def superpose(a, b, on):
    """a moved onto b by the Kabsch fit of a[on] to b[on]."""
    ca, cb = a[on].mean(0), b[on].mean(0)
    u, _, vt = torch.linalg.svd((a[on] - ca).T @ (b[on] - cb))
    d = torch.diag(torch.tensor([1.0, 1.0, float(torch.sign(torch.det(u @ vt)))], dtype=torch.float64))
    return (a - ca) @ (u @ d @ vt) + cb


def kabsch_rmsd(a, b):
    return float((superpose(a, b, slice(None)) - b).norm(dim=-1).pow(2).mean().sqrt())


def grade(a, b, ca, lig):
    out = {"all_atom_A": kabsch_rmsd(a, b), "ca_A": kabsch_rmsd(a[ca], b[ca])}
    if lig.any():
        out["ligand_after_ca_fit_A"] = float((superpose(a, b, ca)[lig] - b[lig]).norm(dim=-1).pow(2).mean().sqrt())
    return out


def summary(rows):
    return {k: {"median": round(statistics.median(r[k] for r in rows), 3), "max": round(max(r[k] for r in rows), 3)}
            for k in rows[0]}


def pack(seed_dir, pred_dir, cases):
    """The case as the grade needs it: merged features, one draw stream per sample, upstream's coordinates per draw."""
    up_feats = torch.load(f"{seed_dir}/features.pt", weights_only=False)
    draws = torch.load(f"{seed_dir}/draws.pt", weights_only=False)
    starts = [i for i, (t, _v) in enumerate(draws) if t == "init"] + [len(draws)]
    streams = [draws[a:b] for a, b in zip(starts, starts[1:])]
    name = Path(seed_dir).parent.name
    job = next(j for j in json.load(open(cases)) if j["name"] == name)
    mode = "no_msa" if "nomsa" in name else "msa"
    table = str(weights.fetch("mol1-ccd") / "ccd_conformers.npz")
    feats = cf.tt_features(job, mode, Path(tempfile.mkdtemp()), conformer_table=table)

    # upstream's tensor wherever both sides build the key: the grade then measures the model, not the featurizer
    replaced, differed, skipped = [], {}, []
    for k, v in up_feats.items():
        if k in feats and tuple(np.shape(feats[k])) == tuple(v.shape):
            ours = np.asarray(feats[k])
            if not np.array_equal(ours, v.numpy().astype(ours.dtype)):
                differed[k] = float(np.abs(ours.astype(np.float64) - v.numpy().astype(np.float64)).max())
            feats[k] = v.numpy().astype(ours.dtype)
            replaced.append(k)
        elif k in feats:
            skipped.append(f"{k}: tt {tuple(np.shape(feats[k]))} upstream {tuple(v.shape)}")

    # upstream names its CIFs by ranking score (sample_0 = best), not by draw order: pair each CIF with the draw
    # chunk whose last recorded denoiser output it is (sampler.pt, ~0.05 A against ~12 A between chunks)
    sam = torch.load(f"{seed_dir}/sampler.pt", weights_only=False)
    n_step = len(sam["noise_schedule"]) - 1
    last = [sam["calls"][(k + 1) * n_step - 1]["x_denoised"].reshape(-1, 3).double() for k in range(len(streams))]
    up = {}
    for p in sorted(glob.glob(f"{pred_dir}/{name}/seed_101/predictions/*_sample_*.cif")):
        xyz, names, het = cif_atoms(p)
        k = min(range(len(last)), key=lambda i: kabsch_rmsd(xyz, last[i]))
        assert k not in up and kabsch_rmsd(xyz, last[k]) < 0.5, (p, k)
        up[k] = xyz
    assert sorted(up) == list(range(len(streams))), sorted(up)
    return {"name": name, "feats": {k: torch.as_tensor(v) for k, v in feats.items()}, "streams": streams,
            "up": [up[k] for k in range(len(streams))],
            "ca": torch.tensor([n == "CA" and not h for n, h in zip(names, het)]),
            "lig": up_feats["is_ligand"].bool(),
            "features": {"from_upstream": replaced, "differed": differed, "shape_mismatch": skipped}}


def run(b, ckpt, n_cycle, out):
    feats, streams, up, ca, lig = b["feats"], b["streams"], b["up"], b["ca"], b["lig"]
    model = Protenix.load_from_checkpoint(ckpt)
    with during() as clk, torch.no_grad():
        t = time.time()
        x, conf = model.fold(feats, n_step=200, n_sample=len(streams), seed=101, n_cycles=int(n_cycle),
                             return_confidence=True, draws=[iter(s) for s in streams])
        wall = time.time() - t
    x = torch.as_tensor(x).reshape(len(streams), -1, 3).double()
    assert x.shape[1] == len(ca), (x.shape, len(ca))
    paired = [grade(x[k], up[k], ca, lig) for k in range(len(up))]
    # every TT sample against every upstream sample: a draw-faithful port is closest on the diagonal
    cross = [[round(kabsch_rmsd(x[a], up[c]), 2) for c in range(len(up))] for a in range(len(x))]
    torch.save({"tt": x.float(), "up": [u.float() for u in up]}, out.replace(".json", ".coords.pt"))
    floor = [grade(up[a], up[c], ca, lig) for a, c in itertools.combinations(range(len(up)), 2)]
    confs = conf if isinstance(conf, list) else [conf]
    res = {"case": b["name"], "n_token": int(feats["restype"].shape[0]), "n_atom": len(ca), "n_cycle": int(n_cycle),
           "n_step": 200, "samples": len(streams), "bar_A": 0.60,
           "tt_vs_upstream_same_draws": summary(paired), "per_sample": paired, "all_atom_tt_x_upstream_A": cross,
           "diag_is_row_min": sum(min(range(len(r)), key=r.__getitem__) == i for i, r in enumerate(cross)),
           "upstream_sample_floor": summary(floor),
           "plddt_tt": [round(float(torch.as_tensor(c["plddt"]).float().mean()), 4) for c in confs],
           "fold_s": round(wall, 1), "aiclk": clk.summary().get(0), "card": os.environ.get("TT_VISIBLE_DEVICES"),
           "features": b["features"]}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("tt_vs_upstream_same_draws", "diag_is_row_min", "upstream_sample_floor",
                                          "plddt_tt", "fold_s", "aiclk", "features")}, indent=1))


if sys.argv[1] == "--pack":
    torch.save(pack(*sys.argv[2:5]), sys.argv[5])
elif sys.argv[1] == "--bundle":
    run(torch.load(sys.argv[2], weights_only=False), *sys.argv[3:6])
else:
    seed_dir, pred_dir, cases, ckpt, n_cycle, out = sys.argv[1:7]
    run(pack(seed_dir, pred_dir, cases), ckpt, n_cycle, out)
