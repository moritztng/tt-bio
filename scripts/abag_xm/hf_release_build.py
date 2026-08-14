#!/usr/bin/env python3
"""Build the AbAg-XM HuggingFace release tree from the frozen campaign artifacts on qb1.

Every stage is idempotent and independently re-runnable:

    python3 scripts/abag_xm/hf_release_build.py --stage samples
    python3 scripts/abag_xm/hf_release_build.py --stage structures --jobs 24

Reads only. Nothing under ~/abag_xm is modified; everything is written to --out.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

HOME = Path.home()
DEEPN = HOME / "abag_xm" / "deepn"
GALAXY = DEEPN / "galaxy"
FROZEN = DEEPN / "dataset_n512"
GROUND_TRUTH = HOME / "abag_xm" / "ground_truth"
MSA_CACHE = HOME / "abag_xm" / "msa_cache"
YAML_REF = "origin/wk/abag-xm-deepn-saturation-fullpanel"
YAML_DIR = "examples/abag_xm"

# published model name -> directory name under galaxy/ (also the results-dir prefix)
MODELS = {
    "boltz2": "boltz2",
    "opendde-abag": "opendde",
    "protenix-v2": "protenix",
    "esmfold2": "esmfold2",
}
# the one cell that no published number was computed on (p2-era pipeline artifact)
EXCLUDED_CELLS = {("opendde-abag", "9sbb")}
UNSCORABLE = {"9ly2", "9ly3", "9lz2"}

EXPECTED_ROWS = {
    "boltz2": 83968,
    "opendde-abag": 83456,
    "protenix-v2": 83968,
    "esmfold2": 83968,
}

FLOAT_COLS = [
    "selector", "confidence_score", "ptm", "iptm", "complex_plddt", "dockq",
    "irmsd", "lrmsd", "fnat", "interface_lddt", "cdr_h1_rmsd", "cdr_h2_rmsd",
    "cdr_h3_rmsd", "epitope_jaccard",
]
COLUMN_ORDER = (
    ["model", "target", "chunk", "rank", "sample_id"] + FLOAT_COLS
    + ["seed", "mps", "wall_s", "hardware", "code_sha"]
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_show(repo: Path, path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), "show", f"{YAML_REF}:{path}"],
        check=True, capture_output=True,
    ).stdout


def target_list(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "--name-only", YAML_REF, YAML_DIR + "/"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    targets = sorted(Path(p).stem for p in out if p.endswith(".yaml"))
    assert len(targets) == 164, f"expected 164 fold inputs, got {len(targets)}"
    return targets


def cell_dir(model: str, target: str, chunk: int) -> Path:
    return GALAXY / MODELS[model] / f"{target}_n512_c{chunk}"


# ---------------------------------------------------------------- samples

CAST_REPORT: list[tuple[str, str, float, float]] = []


def build_samples(out: Path) -> None:
    dest = out / "samples"
    dest.mkdir(parents=True, exist_ok=True)
    total = 0
    worst_cast = 0.0
    for model in MODELS:
        df = pd.read_parquet(FROZEN / f"{model}_samples.parquet")
        df = df[df["rung"] == 512].drop(columns=["rung"]).reset_index(drop=True)
        assert len(df) == EXPECTED_ROWS[model], f"{model}: {len(df)} rows"
        assert set(df["model"]) == {model}
        assert df["hardware"].nunique() == 1 and df["code_sha"].nunique() == 1, model

        df["sample_id"] = (
            df["target"] + "_c" + df["chunk"].astype(str) + "_r" + df["rank"].astype(str)
        )
        assert df["sample_id"].is_unique, f"{model}: sample_id collision"

        # every (target, chunk, rank) grid cell present exactly once
        per_target = df.groupby("target").size()
        assert set(per_target) == {512}, f"{model}: cells not all 512 deep"
        assert set(df["chunk"]) == set(range(8)) and set(df["rank"]) == set(range(64))

        for col in FLOAT_COLS:
            v = df[col].to_numpy(dtype="float64")
            v = v[np.isfinite(v)]
            if v.size:
                # absolute for the [0,1] columns, relative for the Angstrom ones:
                # float32 carries ~7 significant digits, so a 50 A RMSD cannot be
                # held to 1e-6 absolute and should not be
                abs_err = np.abs(v.astype("float32") - v)
                err = float(np.max(abs_err / np.maximum(np.abs(v), 1.0)))
                worst_cast = max(worst_cast, err)
                assert err < 1e-6, f"{model}.{col}: float32 cast error {err:g} (relative)"
                CAST_REPORT.append((model, col, float(abs_err.max()), err))
            df[col] = df[col].astype("float32")

        df["chunk"] = df["chunk"].astype("int8")
        df["rank"] = df["rank"].astype("int8")
        df["seed"] = df["seed"].astype("Int32")
        df["mps"] = df["mps"].astype("Int8")
        df["wall_s"] = df["wall_s"].astype("Int32")
        df = df[COLUMN_ORDER]

        df.to_parquet(dest / f"{model}.parquet", compression="zstd", index=False)
        total += len(df)
        print(f"samples/{model}.parquet  {len(df):,} rows  "
              f"{(dest / f'{model}.parquet').stat().st_size / 1e6:.1f} MB")
    assert total == 335360, total
    abs_max = max(CAST_REPORT, key=lambda r: r[2])
    print(f"samples total {total:,} rows")
    print(f"float32 cast: worst relative error {worst_cast:.3g}; "
          f"worst absolute {abs_max[2]:.3g} on {abs_max[0]}.{abs_max[1]}")


# ---------------------------------------------------------------- targets

def _labels_for(target: str) -> tuple[str, dict]:
    """First available labels.json for a target, preferring boltz2."""
    for model in ["boltz2", "protenix-v2", "esmfold2", "opendde-abag"]:
        if (model, target) in EXCLUDED_CELLS:
            continue
        p = cell_dir(model, target, 0) / "labels.json"
        if p.exists():
            return model, json.loads(p.read_text())
    raise FileNotFoundError(f"no labels.json for {target}")


def build_targets(out: Path, repo: Path) -> None:
    dest = out / "targets"
    dest.mkdir(parents=True, exist_ok=True)
    scorable = set()
    for model in MODELS:
        df = pd.read_parquet(FROZEN / f"{model}_samples.parquet", columns=["target", "dockq"])
        scorable |= set(df.loc[df["dockq"].notna(), "target"])

    rows = []
    for target in target_list(repo):
        spec = yaml.safe_load(git_show(repo, f"{YAML_DIR}/{target}.yaml"))
        seqs = {e["protein"]["id"]: e["protein"]["sequence"] for e in spec["sequences"]}
        assert set(seqs) <= {"A", "H", "L"} and {"A", "H"} <= set(seqs), (target, list(seqs))

        _, labels = _labels_for(target)
        dq = labels["samples"][0].get("dockq") or {}
        native = GROUND_TRUTH / f"{target}.cif"
        assert native.exists(), native

        is_scorable = target in scorable
        assert is_scorable == (target not in UNSCORABLE), target
        rows.append({
            "target": target,
            "chains": sorted(seqs),
            "seq_antigen": seqs["A"],
            "seq_h": seqs["H"],
            "seq_l": seqs.get("L"),
            "n_res_total": sum(len(s) for s in seqs.values()),
            "n_res_antigen": len(seqs["A"]),
            "n_res_h": len(seqs["H"]),
            "n_res_l": len(seqs["L"]) if "L" in seqs else None,
            "native_file": f"natives/{target}.cif",
            "native_chain1": dq.get("native_chain1"),
            "native_chain2": dq.get("native_chain2"),
            "interface": dq.get("interface"),
            "chain_map": json.dumps(dq.get("chain_map") or {}, sort_keys=True),
            "msa_a3m": json.dumps(
                {cid: f"msa/{hashlib.sha256(s.encode()).hexdigest()[:16]}.a3m.gz"
                 for cid, s in sorted(seqs.items())}, sort_keys=True),
            "dockq_scorable": is_scorable,
            "note": ("3-way Ab:Ag hetero-hexamer asymmetric unit; the scorer resolves no "
                     "antibody-antigen interface, so dockq is null in every model."
                     if target in UNSCORABLE else None),
        })

    df = pd.DataFrame(rows)
    df["n_res_l"] = df["n_res_l"].astype("Int32")
    for col in ["n_res_total", "n_res_antigen", "n_res_h"]:
        df[col] = df[col].astype("int32")
    assert len(df) == 164, len(df)
    assert set(df.loc[~df["dockq_scorable"], "target"]) == UNSCORABLE
    df.to_parquet(dest / "targets.parquet", compression="zstd", index=False)
    print(f"targets/targets.parquet  {len(df)} rows, "
          f"{(~df['dockq_scorable']).sum()} not DockQ-scorable "
          f"({', '.join(sorted(UNSCORABLE))})")


# ---------------------------------------------------------------- natives / inputs / msa

def build_natives(out: Path, repo: Path) -> None:
    dest = out / "natives"
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for target in target_list(repo):
        src = GROUND_TRUTH / f"{target}.cif"
        dst = dest / f"{target}.cif"
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        digest = sha256_file(src)
        assert sha256_file(dst) == digest, target
        manifest[target] = digest
    (out / "natives_sha256.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(f"natives/  {len(manifest)} files, all sha256-matching ~/abag_xm/ground_truth")


def build_inputs(out: Path, repo: Path) -> None:
    dest = out / "inputs"
    dest.mkdir(parents=True, exist_ok=True)
    for target in target_list(repo):
        (dest / f"{target}.yaml").write_bytes(git_show(repo, f"{YAML_DIR}/{target}.yaml"))
    print(f"inputs/  {len(target_list(repo))} fold inputs, byte-identical to the campaign")


def build_msa(out: Path, repo: Path) -> None:
    dest = out / "msa"
    dest.mkdir(parents=True, exist_ok=True)
    chains = 0
    wanted: dict[str, Path] = {}
    for target in target_list(repo):
        spec = yaml.safe_load(git_show(repo, f"{YAML_DIR}/{target}.yaml"))
        for entry in spec["sequences"]:
            seq = entry["protein"]["sequence"]
            h = hashlib.sha256(seq.encode()).hexdigest()[:16]
            src = MSA_CACHE / f"{h}.a3m"
            assert src.exists(), f"{target}/{entry['protein']['id']}: missing {src}"
            wanted[h] = src
            chains += 1
    print(f"msa: {chains}/{chains} chain sequences resolve to a cached a3m, 0 missing")

    headers = non_uniref = 0
    for h, src in sorted(wanted.items()):
        dst = dest / f"{h}.a3m.gz"
        if not dst.exists():
            with open(src, "rb") as fin, gzip.open(dst, "wb", compresslevel=6) as fout:
                shutil.copyfileobj(fin, fout)
        with open(src, "r") as fh:
            for line in fh:
                if line.startswith(">"):
                    headers += 1
                    if not re.match(r">UniRef", line):
                        non_uniref += 1
    size = sum(p.stat().st_size for p in dest.glob("*.a3m.gz"))
    print(f"msa/  {len(wanted)} files, {size / 1e6:.1f} MB")
    print(f"msa header census: {headers:,} headers, {non_uniref:,} not matching ^>UniRef")


# ---------------------------------------------------------------- structures

def _build_shard(args) -> tuple[str, str, int, int]:
    model, target, out_str = args
    out = Path(out_str)
    dst = out / "structures" / model / f"{target}.parquet"
    if dst.exists():
        try:
            if pq.ParquetFile(dst).metadata.num_rows == 512:
                return model, target, 512, dst.stat().st_size
        except Exception:
            pass
    prefix = MODELS[model]
    ids, chunks, ranks, cifs = [], [], [], []
    for chunk in range(8):
        d = cell_dir(model, target, chunk)
        labels = json.loads((d / "labels.json").read_text())["samples"]
        assert len(labels) == 64, f"{model}/{target}/c{chunk}: {len(labels)} labels"
        struct_dir = d / f"{prefix}_results_{target}" / "structures"
        for s in labels:
            rank = int(s["rank"])
            # basename only: recorded paths point into the label tree and at two host roots
            path = struct_dir / os.path.basename(s["cif"])
            ids.append(f"{target}_c{chunk}_r{rank}")
            chunks.append(chunk)
            ranks.append(rank)
            cifs.append(path.read_text())
    assert len(ids) == 512, len(ids)
    assert len(set(zip(chunks, ranks))) == 512, f"{model}/{target}: grid not 8x64"

    table = pa.table({
        "sample_id": pa.array(ids, pa.string()),
        "chunk": pa.array(chunks, pa.int8()),
        "rank": pa.array(ranks, pa.int8()),
        "cif": pa.array(cifs, pa.string()),
    })
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd", compression_level=9, row_group_size=64)
    tmp.rename(dst)
    return model, target, 512, dst.stat().st_size


def build_structures(out: Path, repo: Path, jobs: int) -> None:
    targets = target_list(repo)
    work = [(m, t, str(out)) for m in MODELS for t in targets
            if (m, t) not in EXCLUDED_CELLS]
    assert len(work) == 655, len(work)
    done = total_bytes = 0
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(_build_shard, w) for w in work]
        for fut in as_completed(futures):
            model, target, rows, size = fut.result()
            done += 1
            total_bytes += size
            if done % 25 == 0 or done == len(work):
                print(f"  {done}/{len(work)} shards, {total_bytes / 1e9:.1f} GB",
                      flush=True)
    shards = list((out / "structures").glob("*/*.parquet"))
    assert len(shards) == 655, f"{len(shards)} shards on disk, expected 655"
    print(f"structures/  655 shards x 512 rows, {total_bytes / 1e9:.2f} GB")


# ---------------------------------------------------------------- verify

def verify(out: Path) -> None:
    sample_ids = set()
    for model in MODELS:
        df = pd.read_parquet(out / "samples" / f"{model}.parquet", columns=["model", "sample_id"])
        sample_ids |= {f"{model}/{s}" for s in df["sample_id"]}
    assert len(sample_ids) == 335360, len(sample_ids)

    struct_ids = set()
    for path in sorted((out / "structures").glob("*/*.parquet")):
        model = path.parent.name
        ids = pq.read_table(path, columns=["sample_id"])["sample_id"].to_pylist()
        assert len(ids) == 512, f"{path}: {len(ids)} rows"
        struct_ids |= {f"{model}/{s}" for s in ids}
    print(f"structures: {len(struct_ids):,} sample_ids over "
          f"{len(list((out / 'structures').glob('*/*.parquet')))} shards")
    assert struct_ids == sample_ids, (
        f"set mismatch: {len(struct_ids - sample_ids)} only in structures, "
        f"{len(sample_ids - struct_ids)} only in samples")
    print("sample_id set equality samples <-> structures: OK, 335,360 both ways")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["samples", "targets", "natives", "inputs", "msa",
                             "structures", "verify", "small"])
    ap.add_argument("--out", default=str(HOME / "hf_stage" / "abag-xm"))
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo)

    stages = ["samples", "targets", "natives", "inputs", "msa"] if args.stage == "small" \
        else [args.stage]
    for stage in stages:
        if stage == "samples":
            build_samples(out)
        elif stage == "targets":
            build_targets(out, repo)
        elif stage == "natives":
            build_natives(out, repo)
        elif stage == "inputs":
            build_inputs(out, repo)
        elif stage == "msa":
            build_msa(out, repo)
        elif stage == "structures":
            build_structures(out, repo, args.jobs)
        elif stage == "verify":
            verify(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
