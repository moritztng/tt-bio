#!/usr/bin/env python3
"""Fold-level parity for the three eltwise/norm fusion gates, on cdk2x2_298 and cdk2x2_512.

Same protocol and the same scorer lineage as the boltz-2 lever rows
(`perf/b2x-flag-levers/ab_flag_levers.py`, `perf/other512/cif_rmsd.py`,
`perf/b2x-flag-levers/domain_split.py`), pointed at the models these fusions actually
reach: openfold3 and protenix. Boltz-2 has no fused site (the only `tenstorrent.py`
site sits behind `fp32_raw_matmul_attention`, set at exactly one place --
`protenix.py:939`), so boltz2 is a free by-construction control: its CIF must be
byte-identical across arms.

Arms in ONE process, sharing a model load, a device and a program cache. The gates are
module globals the helpers read at call time, so the arm is set by assignment, not by
the environment -- and the environment is asserted NOT to pin any of them, because a
pin would silently serve every arm the same code.

    off    all three gates False -- the two-op chains, i.e. main's arithmetic
    on     all three gates True  -- the fused ops
    off2   a repeat of `off`, which is the A/A floor and the determinism control

Scoring: cdk2x2_298 all-atom and CA Kabsch RMSD of `on` against `off`. cdk2x2_512 is
the chimeric fixture whose hinge saturates whole-molecule RMSD for any non-bit-exact
change, so it is read per pseudo-domain (memory
`cdk2x2-chimeric-fixture-cannot-score-non-bit-exact-parity`); the whole-molecule number
is recorded but is not the verdict.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, socket, sys, tempfile, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "other512"))
sys.path.insert(0, str(REPO / "perf" / "b2x-flag-levers"))

FIX = REPO / "perf" / "size512" / "fixtures"
GATES = ("FUSE_SCALE_ADD", "FUSE_MASK_ADD", "FUSE_NORM_RESIDUAL")

#: arm -> (set of gates to turn ON, seed offset). A per-gate arm is what makes the verdict
#: per-site: the three fusions are independent ops and a single blanket reading cannot say
#: which one moved the structure. The `seedN` arms are all-gates-OFF at a different seed --
#: that is the SEED FLOOR, the only thing a non-bit-exact deviation can be judged against
#: (memory bit-exactness-not-required-accuracy-bar-is).
ARMS = {
    "off":    (frozenset(), 0),
    "on":     (frozenset(GATES), 0),
    "scale":  (frozenset({"FUSE_SCALE_ADD"}), 0),
    "mask":   (frozenset({"FUSE_MASK_ADD"}), 0),
    "norm":   (frozenset({"FUSE_NORM_RESIDUAL"}), 0),
    "off2":   (frozenset(), 0),
    "seed1":  (frozenset(), 1),
    "seed2":  (frozenset(), 2),
}
OUT: dict = {"arms": []}
OUT_PATH: Path | None = None


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def _seed_msa(target: Path, a3m_text: str, msa_dir: Path) -> None:
    from tt_bio.main import _read_bio_chains
    chains = _read_bio_chains(target)
    assert len(chains) == 1, f"{target} is not a monomer: {len(chains)} chains"
    seq = chains[0][1]
    rows = a3m_text.split("\n")
    assert rows[1] == seq, "a3m query row does not match the target sequence"
    msa_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(seq.encode()).hexdigest()[:16]
    (msa_dir / f"{h}.a3m").write_text(a3m_text)


def build_cfg(model, msa_dir, struct_dir, recycles, steps, samples, seed):
    return dict(
        model=model, fast=False, output_format="cif",
        recycling_steps=recycles, sampling_steps=steps,
        diffusion_samples=samples, seed=seed, trace=False,
        msa_dir=str(msa_dir), struct_dir=str(struct_dir),
        use_msa_server=False, msa_db_path=None, use_envdb=False, msa_endpoint=None,
        single_sequence=False, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", msa_server_username=None,
        msa_server_password=None, api_key_value=None, max_msa_seqs=8192,
        write_pae=False, write_pde=False, write_embeddings=False, method=None,
    )


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures", default="cdk2x2_298")
    ap.add_argument("--arms", default="off,on,off2")
    ap.add_argument("--recycles", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cifdir", type=Path, required=True)
    a = ap.parse_args()
    OUT_PATH = a.out

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this worktree -- the venv's installed "
        "package would score a different tree (memory "
        "parity-gate-scores-installed-package-not-checkout)")
    import tt_bio.eltwise_fusion as EF
    # An env pin would make every arm the same arm, silently.
    pinned = [g for g in GATES if ("TT_BIO_" + g) in os.environ]
    assert not pinned, f"these gates are pinned in the environment: {pinned}"

    dev = get_device()
    OUT["env"] = dict(
        host=socket.gethostname(), tt_visible_devices=os.environ.get("TT_VISIBLE_DEVICES"),
        lease_cards=os.environ.get("TT_BIO_LEASE_CARDS"), tt_bio_file=_TB.__file__,
        model=a.model, torch=torch.__version__, loadavg=os.getloadavg(),
        started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        protocol=dict(recycles=a.recycles, steps=a.steps, samples=a.samples, seed=a.seed,
                      fixtures=a.fixtures.split(",")),
        gate_defaults={g: getattr(EF, g) for g in GATES})
    try:
        import importlib.metadata as _md
        OUT["env"]["ttnn"] = _md.version("ttnn")
    except Exception:
        pass
    dump()

    work = Path(tempfile.mkdtemp(prefix="eltfuse-parity-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    for name in a.fixtures.split(","):
        _seed_msa(FIX / f"{name}.yaml", (FIX / f"{name}.a3m").read_text(), msa_dir)

    cfg = build_cfg(a.model, msa_dir, struct_dir, a.recycles, a.steps, a.samples, a.seed)
    _ensure_local_artifacts(cfg)
    t_load = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("ttx-eltwise-fusion-sweep", cfg)
    OUT["model_load_s"] = round(time.perf_counter() - t_load, 3)
    dump()

    def fold(arm, fixture):
        want, seed_off = ARMS[arm]
        for g in GATES:
            setattr(EF, g, g in want)
        assert {g for g in GATES if getattr(EF, g)} == set(want)
        cfg["seed"] = a.seed + seed_off
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(FIX / f"{fixture}.yaml", cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        cifs = sorted(struct_dir.rglob("*.cif"))
        assert cifs, f"no CIF written for {arm}/{fixture}"
        keep = a.cifdir / f"{fixture}_{arm}"
        keep.mkdir(parents=True, exist_ok=True)
        dst = keep / f"{fixture}.cif"
        shutil.copy2(cifs[0], dst)
        row = dict(arm=arm, fixture=fixture, gates_on=sorted(want), seed=cfg["seed"],
                   fold_s=round(wall, 3),
                   plddt=metrics.get("complex_plddt", metrics.get("plddt")),
                   cif_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
                   cif=str(dst), loadavg1=round(os.getloadavg()[0], 2))
        OUT["arms"].append(row)
        dump()
        print(f"  {a.model} {fixture} {arm:6s} seed{cfg['seed']} {wall:7.2f}s "
              f"plddt={row['plddt']} sha={row['cif_sha256'][:16]}", flush=True)
        return row

    for fixture in a.fixtures.split(","):
        for arm in a.arms.split(","):
            fold(arm, fixture)

    # ---- score
    from cif_rmsd import read_atoms, kabsch_rmsd
    import numpy as np
    OUT["scores"] = {}
    for fixture in a.fixtures.split(","):
        rows = {r["arm"]: r for r in OUT["arms"] if r["fixture"] == fixture}
        if "off" not in rows:
            continue
        # read_atoms returns (keys, coords[N,3]); match on the key so a different atom
        # ORDER cannot read as a displacement.
        bk, bx = read_atoms(Path(rows["off"]["cif"]))
        sc = {}
        for arm in [x for x in a.arms.split(",") if x != "off"]:
            if arm not in rows:
                continue
            ck, cx = read_atoms(Path(rows[arm]["cif"]))
            common = sorted(set(bk) & set(ck))
            bi = {k: i for i, k in enumerate(bk)}
            ci = {k: i for i, k in enumerate(ck)}
            P = bx[[bi[k] for k in common]]
            Q = cx[[ci[k] for k in common]]
            ca = [k for k in common if k[-2] == "CA"] if common and len(common[0]) >= 2 else []
            sc[arm] = dict(
                all_atom_rmsd=float(kabsch_rmsd(P, Q)), n_atoms=len(common),
                n_off=len(bk), n_arm=len(ck),
                max_atom_dev=float(np.abs(P - Q).max()) if len(common) else None,
                ca_rmsd=(float(kabsch_rmsd(bx[[bi[k] for k in ca]], cx[[ci[k] for k in ca]]))
                         if ca else None),
                n_ca=len(ca),
                bit_exact=(rows[arm]["cif_sha256"] == rows["off"]["cif_sha256"]))
            print(f"  SCORE {a.model} {fixture} {arm} vs off: "
                  f"all-atom {sc[arm]['all_atom_rmsd']:.6f} A "
                  f"CA {sc[arm]['ca_rmsd']} maxdev {sc[arm]['max_atom_dev']} "
                  f"(bit-exact={sc[arm]['bit_exact']}, n={sc[arm]['n_atoms']}/{sc[arm]['n_ca']}CA)",
                  flush=True)
        OUT["scores"][fixture] = sc
    dump()
    print("wrote " + str(a.out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
