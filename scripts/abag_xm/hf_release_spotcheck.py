#!/usr/bin/env python3
"""Spot-check the published AbAg-XM release against the campaign artifacts.

faithfulness: shipped `cif` text is byte-identical to the file the model wrote.
correspondence: the shipped `dockq` value re-scores from the shipped structure.

--source hub    pull the shard from the Hub (the real check)
--source stage  read the staged parquet (fast pre-check)
"""
import argparse, hashlib, json, os, random, subprocess, sys, tempfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

HOME = Path.home()
GALAXY = HOME / "abag_xm/deepn/galaxy"
STAGE = HOME / "hf_stage/abag-xm"
REPO = "Tenstorrent/abag-xm"
WT = Path("/home/ttuser/.coworker/wt/abag-dataset-hf-release")
LABEL_PY = "/home/ttuser/.abag_xm_label_venv/bin/python"
MODELS = {"boltz2": "boltz2", "opendde-abag": "opendde",
          "protenix-v2": "protenix", "esmfold2": "esmfold2"}


def shard_path(source, model, target):
    if source == "stage":
        return STAGE / "structures" / model / f"{target}.parquet"
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, f"structures/{model}/{target}.parquet",
                                repo_type="dataset", token=os.environ["HF_TOKEN"]))


def source_cif(model, target, chunk, rank):
    prefix = MODELS[model]
    d = GALAXY / prefix / f"{target}_n512_c{chunk}"
    labels = json.loads((d / "labels.json").read_text())["samples"]
    rec = next(s for s in labels if int(s["rank"]) == rank)
    return d / f"{prefix}_results_{target}" / "structures" / os.path.basename(rec["cif"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hub", "stage"], default="hub")
    ap.add_argument("--faithfulness", type=int, default=20)
    ap.add_argument("--correspondence", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    samples = pd.concat([pd.read_parquet(STAGE / "samples" / f"{m}.parquet") for m in MODELS])
    targets = pd.read_parquet(STAGE / "targets" / "targets.parquet").set_index("target")

    pool = samples.sample(n=args.faithfulness, random_state=args.seed)
    print(f"== faithfulness ({args.source}) ==")
    ok = 0
    for _, r in pool.iterrows():
        shard = shard_path(args.source, r["model"], r["target"])
        tbl = pq.read_table(shard).to_pandas()
        row = tbl.loc[tbl["sample_id"] == r["sample_id"]].iloc[0]
        got = hashlib.sha256(row["cif"].encode()).hexdigest()
        want = hashlib.sha256(source_cif(r["model"], r["target"], int(r["chunk"]),
                                         int(r["rank"])).read_bytes()).hexdigest()
        ok += got == want
        print(f"  {'OK  ' if got == want else 'FAIL'} {r['model']}/{r['sample_id']}  {got[:16]}")
    print(f"faithfulness: {ok}/{len(pool)} sha256 matches")

    scorable = samples[samples["dockq"].notna()]
    picks = scorable.sample(n=args.correspondence, random_state=args.seed + 1)
    print(f"\n== correspondence ({args.source}) ==")
    agree = 0
    for _, r in picks.iterrows():
        shard = shard_path(args.source, r["model"], r["target"])
        tbl = pq.read_table(shard).to_pandas()
        cif = tbl.loc[tbl["sample_id"] == r["sample_id"], "cif"].iloc[0]
        t = targets.loc[r["target"]]
        with tempfile.TemporaryDirectory() as td:
            mp = Path(td) / "model.cif"
            mp.write_text(cif)
            out = subprocess.run(
                [LABEL_PY, "scripts/abag_xm_dockq_interface.py", str(mp),
                 str(HOME / f"abag_xm/ground_truth/{r['target']}.cif"),
                 t["native_chain1"], t["native_chain2"]],
                cwd=WT, capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": str(WT)})
        if out.returncode != 0:
            print(f"  FAIL {r['model']}/{r['sample_id']}: scorer rc={out.returncode}\n{out.stderr[-400:]}")
            continue
        res = json.loads(out.stdout)
        got = res.get("dockq", res.get("DockQ"))
        delta = abs(float(got) - float(r["dockq"]))
        agree += delta < 1e-6
        print(f"  {'OK  ' if delta < 1e-6 else 'FAIL'} {r['model']}/{r['sample_id']}  "
              f"published {r['dockq']:.9f}  re-scored {float(got):.9f}  |delta| {delta:.3g}")
    print(f"correspondence: {agree}/{len(picks)} agree to < 1e-6")
    return 0 if ok == len(pool) and agree == len(picks) else 1


if __name__ == "__main__":
    sys.exit(main())
