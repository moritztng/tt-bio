#!/usr/bin/env python3
"""Preflight for the embedding and esmfold2-fast perf-page rows. No torch, no GPU, no card.

Every row this checks costs money to measure. The cheapest place to find a wrong repo name, a
missing weight file, a checkpoint whose config does not have the shape the row claims, or a
fixture that got edited is here, on a CPU box, before an instance is rented. Run it on pc
before renting and again on the box after transfer.

What it proves, per row:
  * the HF repo resolves and carries a weight file the row's loader can actually read
    (safetensors / pytorch_model.bin -- an esm-SDK .pth at a nonstandard path does not count,
    which is exactly how biohub/esmc-300m-2024-12 fails here and biohub/ESMC-300M passes);
  * config.json declares the architecture and the (layers, width) the manifest expects, so a
    row cannot silently measure a different model than the one it is named after;
  * the byte-pinned fixture still hashes to its recorded sha256.

Usage:
    python3 scripts/gpu_vs_tt/embed_preflight.py            # every row
    python3 scripts/gpu_vs_tt/embed_preflight.py --row esmc-6b saprot-1.3b
Exit code is the number of failed checks, so it drops straight into a setup script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

HF = "https://huggingface.co"

# Rows this manifest adds, each with the shape its cell claims. `arch` is config.json's
# `architectures[0]`; `shape` maps a config key to its required value. Both are read off the
# live config rather than trusted, because a checkpoint swap under a familiar repo name is a
# silent way to measure the wrong model.
ROWS = {
    "esmfold2-fast": dict(
        repo="biohub/ESMFold2-Fast", arch="ESMFold2Model",
        weights=("model.safetensors",),
        shape={"folding_trunk.n_layers": 24, "lm_num_layers": 80,
               "msa_encoder.enabled": False, "esmc_id": "biohub/ESMC-6B"},
        note="24-block trunk against ESMFold2's 48, MSA encoder off, same ESMC-6B backbone",
    ),
    "esmc-300m": dict(
        repo="biohub/ESMC-300M", arch="ESMCForMaskedLM",
        weights=("model.safetensors",),
        shape={"n_layers": 30, "d_model": 960, "n_heads": 15},
        note="transformers-format mirror of biohub/esmc-300m-2024-12 (which ships only the SDK .pth)",
    ),
    "esmc-600m": dict(
        repo="biohub/ESMC-600M", arch="ESMCForMaskedLM",
        weights=("model.safetensors",),
        shape={"n_layers": 36, "d_model": 1152, "n_heads": 18},
        note="transformers-format mirror of biohub/esmc-600m-2024-12",
    ),
    "esmc-6b": dict(
        repo="biohub/ESMC-6B", arch="ESMCForMaskedLM",
        weights=("model.safetensors.index.json",),
        shape={"n_layers": 80, "d_model": 2560, "n_heads": 40},
        note="sharded; the same weights ESMFold2's LM backbone loads, so the 25 GB pull is shared",
    ),
    "saprot-35m": dict(
        repo="westlake-repl/SaProt_35M_AF2", arch="EsmForMaskedLM",
        weights=("pytorch_model.bin",),
        shape={"num_hidden_layers": 12, "hidden_size": 480, "vocab_size": 446},
        note="stock transformers ESM-2 over the 446-token fused AA+3Di vocabulary",
    ),
    "saprot-650m": dict(
        repo="westlake-repl/SaProt_650M_AF2", arch="EsmForMaskedLM",
        weights=("pytorch_model.bin",),
        shape={"num_hidden_layers": 33, "hidden_size": 1280, "vocab_size": 446},
        note="same loader as 35M",
    ),
    "saprot-1.3b": dict(
        repo="westlake-repl/SaProt_1.3B_AF2", arch="EsmForMaskedLM",
        weights=("pytorch_model.bin",),
        shape={"num_hidden_layers": 66, "hidden_size": 1280, "vocab_size": 446},
        note="no .pt in this repo, only the transformers bin -- the .pt route would 404",
    ),
}

# The fixture every row above reads. Byte-identical to the protein chain in the page's folding
# fixture perf/size512/fixtures/cdk2x2_512.yaml, so the embed rows and the fold rows are the
# same 512 residues of the same protein.
FIXTURES = {
    "scripts/gpu_vs_tt/fixtures/prot512.seq":
        "141f7d4730ccf17e116016edc4aceee502d8c9769301ece4d1b64beb496ebf8d",
    "perf/size512/fixtures/cdk2x2_512.yaml":
        "24d8b2d8c06e4409995abae024766e316da3175dde7596073b68c7963d2df398",
}


def get_json(url: str, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"User-Agent": "tt-bio-embed-preflight"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def dig(cfg: dict, dotted: str):
    """config.json is nested (folding_trunk.n_layers); walk it or return a marker."""
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return "<absent>"
        cur = cur[part]
    return cur


def check_row(name: str, spec: dict) -> list[str]:
    fails = []
    repo = spec["repo"]
    try:
        info = get_json(f"{HF}/api/models/{repo}")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        return [f"{name}: repo {repo} does not resolve ({type(e).__name__}: {e})"]

    files = {s["rfilename"] for s in info.get("siblings", [])}
    for w in spec["weights"]:
        if w not in files:
            fails.append(f"{name}: {repo} has no {w} -- the row's loader cannot read this repo")
    if "config.json" not in files:
        fails.append(f"{name}: {repo} has no config.json")
        return fails

    try:
        cfg = get_json(f"{HF}/{repo}/resolve/main/config.json")
    except Exception as e:
        return fails + [f"{name}: cannot read {repo} config.json ({type(e).__name__}: {e})"]
    if not cfg:
        return fails + [f"{name}: {repo} config.json is empty -- not a transformers repo"]

    arch = (cfg.get("architectures") or [None])[0]
    if arch != spec["arch"]:
        fails.append(f"{name}: {repo} architectures[0] is {arch!r}, row expects {spec['arch']!r}")
    for key, want in spec["shape"].items():
        got = dig(cfg, key)
        if got != want:
            fails.append(f"{name}: {repo} {key} is {got!r}, row expects {want!r}")

    if not fails:
        shape = " ".join(f"{k}={dig(cfg, k)}" for k in spec["shape"])
        print(f"  OK   {name:14s} {repo:32s} {arch:18s} {shape}")
    return fails


def check_fixtures() -> list[str]:
    fails = []
    for rel, want in FIXTURES.items():
        p = REPO / rel
        if not p.exists():
            fails.append(f"fixture {rel} is missing")
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            fails.append(f"fixture {rel} sha256 {got} != pinned {want}")
        else:
            print(f"  OK   fixture       {rel}  {got[:16]}...")
    # The embed rows only mean the same thing as the fold rows if the sequence they read is the
    # protein chain of the folding fixture. Checked, not assumed.
    seq = (REPO / "scripts/gpu_vs_tt/fixtures/prot512.seq")
    yml = (REPO / "perf/size512/fixtures/cdk2x2_512.yaml")
    if seq.exists() and yml.exists():
        s = seq.read_text().strip()
        if s not in yml.read_text():
            fails.append("prot512.seq is not the protein chain of cdk2x2_512.yaml -- the embed "
                         "rows would not be the same target as the fold rows")
        elif len(s) != 512:
            fails.append(f"prot512.seq is {len(s)} residues, the page scope is 512")
        else:
            print(f"  OK   fixture       prot512.seq is cdk2x2_512.yaml's chain, {len(s)} residues")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--row", nargs="*", choices=sorted(ROWS), default=sorted(ROWS))
    ap.add_argument("--skip-fixtures", action="store_true")
    args = ap.parse_args()

    fails = []
    if not args.skip_fixtures:
        print("fixtures:")
        fails += check_fixtures()
    print("rows:")
    for name in args.row:
        fails += check_row(name, ROWS[name])

    print()
    if fails:
        print(f"PREFLIGHT FAIL ({len(fails)}):")
        for f in fails:
            print("  -", f)
        return len(fails)
    print(f"PREFLIGHT PASS: {len(args.row)} row(s), every repo resolves and every shape matches.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
