#!/usr/bin/env python3
"""PXDesign's preset graph, end to end, on tt-bio.

    TT_VISIBLE_DEVICES=0 python3 scripts/pxdesign_port/preset_run.py \\
        --yaml tests/fixtures/pxdesign/PDL1.yaml --preset preview --n_sample 4 --out /tmp/pv

Four stages, in upstream's order and with upstream's settings, read out of
`pxdbench/pxd_configs/eval.py` (the eval half of PXDesign ships as a separate package,
`github.com/bytedance/PXDesignBench`, which is not installed here; only its configuration is
used, and it is quoted at each call site):

  1. the generator, `tt_bio.pxdesign` on device, from a target structure file;
  2. ProteinMPNN on the generated backbone, upstream's checkout on host. tt-bio purged its own
     module in July on purpose, and PXDesign's `af2_easy` / `af2_opt` thresholds were calibrated
     on MPNN sequences, so substituting a different inverse-folder changes what the filter means;
  3. AF2-IG, the tt-bio **host torch** arm (`tt_bio.af2_reference`), complex then binder monomer.
     Not the ttnn arm: pass 12 measured it flipping one accept/reject in 26 designs and ask 5628b
     is open on it, so making it a preset's default would silently lower the confidence of every
     verdict the preset prints. `--af2-device` offers it as a second column;
  4. the filters and `pre_filter_preview`'s ranking.

`extended` adds the Protenix filter, which is `--preset extended`.

**Scale.** The shipped presets are `N_sample=100` (preview) and `N_sample=500` (extended) at
304 s a design of host AF2, i.e. 8.4 h and 42 h of AF2 alone. Every run here is at a stated
`--n_sample` and every number it prints carries it. The claim this script supports is "every
stage of the preset graph ran", not "the preset's published yield reproduces".
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

MPNN = Path.home() / "scratch" / "ProteinMPNN"
AF2_PARAMS = Path.home() / "pxd_tool_weights" / "af2" / "params_model_1_ptm.npz"
CKPT = Path("~/pxdesign_release_data/checkpoint/pxdesign_v0.1.0.pt").expanduser()

# pxdbench/pxd_configs/eval.py:41-46, the binder MPNN block.
MPNN_CFG = {"weights": "vanilla_model_weights", "model_name": "v_48_020",
            "rm_aa": "C", "temperature": 0.0001}
# pxdbench/pxd_configs/eval.py:47-54, the binder AF2 block. model_ids [0] is model_1_ptm.
AF2_RECYCLES = 3
# pxdbench/pxd_configs/eval.py:84-95.
FILTERS = {
    "af2_easy": {"pLDDT": (">", 0.8), "i_pTM": (">", 0.5), "i_pAE": ("<", 0.35),
                 "bound_unbound_RMSD": ("<", 3.5)},
    "af2_opt": {"pLDDT": (">", 0.9), "unscaled_i_pAE": ("<", 7.0),
                "af2_binder_pred_design_rmsd": ("<", 1.5)},
}
CA = 1                       # index of CA in AF2's atom37 order
AA3TO1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
          "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
          "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
          "TYR": "Y", "VAL": "V"}


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """CA-on-CA RMSD after optimal superposition, exactly `pxdbench/metrics/Kalign.py`:
    Kabsch with a determinant sign fix, then the plain root-mean-square over all N points."""
    if p.shape != q.shape or len(p) < 3:
        raise ValueError(f"kabsch_rmsd: {p.shape} vs {q.shape}")
    cp, cq = p.mean(0), q.mean(0)
    h = (p - cp).T @ (q - cq)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    aligned = (q - cq) @ r.T + cp
    return float(np.sqrt(((p - aligned) ** 2).sum() / len(p)))


# --------------------------------------------------------------------------- stage 1

def write_design_pdb(path: Path, coords, feats, res_name) -> tuple[int, str]:
    """One generated sample as chain A (target) plus chain B (binder backbone).

    Upstream writes the diffusion output as a cif and converts it; the layout that matters
    downstream is target first, binder last, which is what `af2_data.complex_features` and
    ProteinMPNN's `--pdb_path_chains B` both assume. The binder's residues are labelled GLY
    because the whole point of the stage that follows is that they have no identity yet.
    """
    a2t = feats["atom_to_token_idx"].long().tolist()
    names = ["".join(chr(c + 32) for c in row.tolist()).strip()
             for row in feats["ref_atom_name_chars"].argmax(-1)]
    binder = [r == "xpb" for r in res_name]
    n_target = res_name.index("xpb")
    serial, chain_res = 0, [0, 0]
    with path.open("w") as fh:
        for want_binder in (False, True):
            last_tok = None
            for i, t in enumerate(a2t):
                if binder[t] != want_binder:
                    continue
                if t != last_tok:
                    chain_res[want_binder] += 1
                    last_tok = t
                x, y, z = (float(v) for v in coords[i])
                nm = names[i]
                fh.write("ATOM  %5d %s %3s %s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
                         % (serial + 1, nm if len(nm) >= 4 else " %-3s" % nm,
                            "GLY" if want_binder else res_name[t],
                            "B" if want_binder else "A", chain_res[want_binder],
                            x, y, z, nm[0].rjust(2)))
                serial += 1
            fh.write("TER\n")
        fh.write("END\n")
    return serial, "".join(AA3TO1.get(r, "X") for r in res_name[:n_target])


def generate(yaml_path: Path, out: Path, n_sample: int, n_step: int, seed: int) -> list[dict]:
    import torch
    from tt_bio.pxdesign.inputs import design_inputs_from_yaml
    from tt_bio.pxdesign.model import ProtenixDesign
    sys.path.insert(0, str(REPO / "scripts" / "pxdesign_port"))
    from design_e2e import _as_model_dtypes, target_reproduction

    feats = _as_model_dtypes(design_inputs_from_yaml(yaml_path))
    res_name = feats["condition"]["res_name"]
    t0 = time.time()
    model = ProtenixDesign.load_from_checkpoint(str(CKPT))
    coords = model.design(feats, n_step=n_step, n_sample=n_sample, seed=seed)
    gen_s = round(time.time() - t0, 1)
    repro = target_reproduction(coords, feats)
    n_binder = sum(1 for r in res_name if r == "xpb")

    rows = []
    for s in range(coords.shape[0]):
        pdb = out / f"design_{s}.pdb"
        atoms, target_seq = write_design_pdb(pdb, coords[s], feats, res_name)
        rows.append({"sample": s, "design_pdb": str(pdb), "atoms": atoms,
                     "n_token": len(res_name), "binder_len": n_binder,
                     "target_seq": target_seq,
                     "target_rmsd": round(repro[s]["target_rmsd"], 3),
                     "contacts_below_5A": repro[s]["n_contacts_below_5A"]})
    return rows, {"generate_s": gen_s,
                  "coords_sha16": __import__("hashlib").sha256(
                      coords.contiguous().numpy().tobytes()).hexdigest()[:16]}


# --------------------------------------------------------------------------- stage 2

def run_mpnn(pdb: Path, out: Path, seed: int) -> str:
    """One sequence for chain B at upstream's binder settings (`num_seqs: 1`,
    `temperature: 0.0001`, `rm_aa: "C"`, original weights)."""
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, str(MPNN / "protein_mpnn_run.py"),
         "--pdb_path", str(pdb), "--pdb_path_chains", "B", "--out_folder", str(out),
         "--num_seq_per_target", "1", "--sampling_temp", str(MPNN_CFG["temperature"]),
         "--omit_AAs", MPNN_CFG["rm_aa"], "--seed", str(seed), "--batch_size", "1",
         "--path_to_model_weights", str(MPNN / MPNN_CFG["weights"]),
         "--model_name", MPNN_CFG["model_name"]],
        check=True, cwd=str(MPNN), stdout=subprocess.DEVNULL)
    fa = next((out / "seqs").glob("*.fa")).read_text().splitlines()
    # entry 0 of the fasta is MPNN echoing its input; the designed sequence is entry 1
    return fa[3].strip().split("/")[-1]


# --------------------------------------------------------------------------- stage 3/4

def to_torch(a):
    import torch
    if a.dtype == np.bool_:
        return torch.from_numpy(a)
    if a.dtype.kind in "iu":
        return torch.from_numpy(a.astype(np.int64))
    return torch.from_numpy(a.astype(np.float32))


def af2_pass(model, feats_np: dict, *, initial_guess: bool, binder_len: int | None):
    from tt_bio.af2_confidence import confidence_scalars
    from tt_bio.af2_data import initial_recycle_state
    from tt_bio.af2_reference import run_recycles
    feats = {k: to_torch(v) for k, v in feats_np.items()}
    prev = {k: to_torch(v) for k, v in
            initial_recycle_state(feats_np, initial_guess=initial_guess).items()}
    last = None
    for out in run_recycles(model, feats, prev, num_recycles=AF2_RECYCLES):
        last = out
    scalars = confidence_scalars(last["plddt_logits"], last["pae_logits"], last["pae_breaks"],
                                 feats["seq_mask"], feats["asym_id"], binder_len=binder_len)
    return scalars, last["structure"]["final_atom_positions"].numpy()


def design_ca(pdb: Path) -> tuple[np.ndarray, np.ndarray]:
    """CA coordinates of the design PDB, split target / binder, in file order."""
    a, b = [], []
    for line in pdb.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            (b if line[21] == "B" else a).append(xyz)
    return np.array(a), np.array(b)


def score_design(model, row: dict, seq: str) -> dict:
    """The AF2 half of the graph: complex then binder monomer, and the three RMSDs between
    them (`pxdbench/tools/af2/main_af2_{complex,monomer}.py`, CA-only Kabsch)."""
    from tt_bio.af2_data import complex_features, monomer_features
    pdb = Path(row["design_pdb"])
    n_binder = row["binder_len"]

    t0 = time.time()
    cx, cx_pos = af2_pass(model, complex_features(str(pdb), seq),
                          initial_guess=True, binder_len=n_binder)
    mono, mono_pos = af2_pass(model, monomer_features(seq),
                              initial_guess=False, binder_len=None)
    af2_s = round(time.time() - t0, 1)

    tgt_ca, bnd_ca = design_ca(pdb)
    pred_cx_ca, pred_bnd_ca = cx_pos[:, CA], cx_pos[-n_binder:, CA]
    pred_mono_ca = mono_pos[:, CA]
    return {
        "seq": seq, "af2_s": af2_s,
        "pLDDT": round(cx["plddt"], 4), "pTM": round(cx["ptm"], 4),
        "i_pTM": round(cx["i_ptm"], 4), "pAE": round(cx["pae"], 4),
        "i_pAE": round(cx["i_pae"], 4),
        "unscaled_i_pAE": round(cx["unscaled_i_pae"], 4),
        "monomer_pLDDT": round(mono["plddt"], 4),
        "af2_complex_pred_design_rmsd":
            round(kabsch_rmsd(np.concatenate([tgt_ca, bnd_ca]), pred_cx_ca), 3),
        "bound_unbound_RMSD": round(kabsch_rmsd(pred_mono_ca, pred_bnd_ca), 3),
        "af2_binder_pred_design_rmsd": round(kabsch_rmsd(pred_mono_ca, bnd_ca), 3),
    }


# --------------------------------------------------------------------------- stage 5

def apply_filters(row: dict) -> dict:
    out = {}
    for name, conds in FILTERS.items():
        ok = all(row[k] > bar if op == ">" else row[k] < bar for k, (op, bar) in conds.items())
        out[f"{name}_success"] = bool(ok)
    return out


def pre_filter_preview(rows: list[dict], min_total_return: int, max_success_return: int,
                       rmsd_threshold: float = 4.0) -> list[dict]:
    """`pxdesign/runner/helpers.py:32`, transcribed. Buckets on (af2_opt, af2_easy, RMSD),
    lower is better, ties broken by `unscaled_i_pAE` ascending; successes are buckets 1-4,
    and failures pad the table only up to `min_total_return`."""
    for r in rows:
        a, ae = r["af2_opt_success"], r["af2_easy_success"]
        rm = r["af2_complex_pred_design_rmsd"] < rmsd_threshold
        r["bucket"] = (1 if (a and ae) else 2 if a else 3 if (ae and rm) else
                       4 if ae else 5 if rm else 6)
        r["pass_af2"] = r["bucket"] in (1, 2, 3, 4)
    key = lambda r: (r["bucket"], r["unscaled_i_pAE"])
    success = sorted([r for r in rows if r["bucket"] < 5], key=key)[:max_success_return]
    out = list(success)
    if len(out) < min_total_return:
        out += sorted([r for r in rows if r["bucket"] >= 5], key=key)[:min_total_return - len(out)]
    for i, r in enumerate(out, start=1):
        r["rank"] = i
    return out


# --------------------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--preset", default="preview", choices=["preview", "extended"])
    ap.add_argument("--n_sample", type=int, default=4)
    ap.add_argument("--n_step", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--af2-params", default=str(AF2_PARAMS))
    ap.add_argument("--stages", default="all",
                    help="comma-separated subset of generate,mpnn,af2,rank; 'all' runs the graph")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stages = ({"generate", "mpnn", "af2", "rank"} if args.stages == "all"
              else set(args.stages.split(",")))
    manifest = out / "manifest.json"
    state = json.loads(manifest.read_text()) if manifest.exists() else {}

    if "generate" in stages:
        rows, meta = generate(Path(args.yaml), out, args.n_sample, args.n_step, args.seed)
        state.update(meta, rows=rows, yaml=str(args.yaml), preset=args.preset,
                     n_sample=args.n_sample, n_step=args.n_step, seed=args.seed)
        manifest.write_text(json.dumps(state, indent=1))
        print(f"[preset] generate: {len(rows)} samples, {meta['generate_s']}s, "
              f"sha16={meta['coords_sha16']}", flush=True)

    if "mpnn" in stages:
        for r in state["rows"]:
            r["seq"] = run_mpnn(Path(r["design_pdb"]), out / f"mpnn_{r['sample']}", args.seed + 1)
            print(f"[preset] mpnn sample {r['sample']}: {r['seq']}", flush=True)
        manifest.write_text(json.dumps(state, indent=1))

    if "af2" in stages:
        import torch
        from tt_bio.af2_reference import load_af2_model
        from tt_bio.af2_weights import load_af2_state_dict
        model = load_af2_model(load_af2_state_dict(args.af2_params),
                               template=True, trunk_dtype=torch.bfloat16)
        for r in state["rows"]:
            r.update(score_design(model, r, r["seq"]))
            r.update(apply_filters(r))
            print(f"[preset] af2 sample {r['sample']}: {r['af2_s']}s pLDDT {r['pLDDT']:.3f} "
                  f"i_pTM {r['i_pTM']:.3f} i_pAE {r['i_pAE']:.3f} "
                  f"unscaled {r['unscaled_i_pAE']:.2f} bu_RMSD {r['bound_unbound_RMSD']:.2f} "
                  f"binder_rmsd {r['af2_binder_pred_design_rmsd']:.2f} "
                  f"complex_rmsd {r['af2_complex_pred_design_rmsd']:.2f} "
                  f"easy={r['af2_easy_success']} opt={r['af2_opt_success']}", flush=True)
            manifest.write_text(json.dumps(state, indent=1))

    if "rank" in stages:
        n = args.n_sample
        ranked = pre_filter_preview(state["rows"], min_total_return=n, max_success_return=n)
        state["ranked"] = [{k: r[k] for k in
                            ("rank", "sample", "bucket", "pass_af2", "seq", "pLDDT", "i_pTM",
                             "i_pAE", "unscaled_i_pAE", "bound_unbound_RMSD",
                             "af2_binder_pred_design_rmsd", "af2_complex_pred_design_rmsd",
                             "af2_easy_success", "af2_opt_success")} for r in ranked]
        manifest.write_text(json.dumps(state, indent=1))
        print(f"\n[preset] {args.preset} ranking, n_sample={args.n_sample} "
              f"(shipped preset is 100 for preview, 500 for extended)")
        print(f"{'rank':>4} {'smp':>3} {'bkt':>3} {'pLDDT':>6} {'i_pTM':>6} {'i_pAE':>6} "
              f"{'unsc':>6} {'bu_rms':>7} {'bnd_rms':>7} {'cx_rms':>7} {'easy':>5} {'opt':>5}")
        for r in state["ranked"]:
            print(f"{r['rank']:4d} {r['sample']:3d} {r['bucket']:3d} {r['pLDDT']:6.3f} "
                  f"{r['i_pTM']:6.3f} {r['i_pAE']:6.3f} {r['unscaled_i_pAE']:6.2f} "
                  f"{r['bound_unbound_RMSD']:7.2f} {r['af2_binder_pred_design_rmsd']:7.2f} "
                  f"{r['af2_complex_pred_design_rmsd']:7.2f} "
                  f"{str(r['af2_easy_success']):>5} {str(r['af2_opt_success']):>5}")
    print(f"[preset] -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
