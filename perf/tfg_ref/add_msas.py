"""Fill in MSAs for every panel target with upstream OpenDDE's own search (ColabFold MMseqs2 API).

Upstream silently falls back to a query-only MSA on a network failure, so each result is checked for depth and a
shallow one is retried, then reported. Paths in the JSONs are made relative to the panel root, so inference runs
from there.

usage: PYTHONPATH=<opendde-src> add_msas.py PANEL_DIR [--retries 3] [--shard I/N]
"""
import argparse
import json
import shutil
import time
from pathlib import Path

from runner.msa_search import update_seq_msa


def depth(a3m):
    return sum(1 for line in open(a3m) if line.startswith(">")) if Path(a3m).exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    root = Path(a.panel).resolve()
    i, n = map(int, a.shard.split("/"))
    targets = json.loads((root / "panel.json").read_text())["targets"][i::n]
    report = {}
    for tid in targets:
        d = root / tid
        jobs = {c: d / f"{tid}_{c}.json" for c in ("unconstrained", "contact", "pocket")}
        first = json.loads(jobs["unconstrained"].read_text())[0]
        if all("unpairedMsaPath" in s["proteinChain"] for s in first["sequences"]):
            continue
        for attempt in range(a.retries):
            shutil.rmtree(d / "msa", ignore_errors=True)
            job = update_seq_msa(json.loads(jobs["unconstrained"].read_text())[0], str(d / "msa"))
            depths = [depth(s["proteinChain"].get("unpairedMsaPath", "")) for s in job["sequences"]]
            if min(depths) > 1:
                break
            time.sleep(30 * (attempt + 1))
        report[tid] = depths
        paths = [{k: str(Path(v).relative_to(root)) for k, v in s["proteinChain"].items() if k.endswith("MsaPath")}
                 for s in job["sequences"]]
        for f in jobs.values():
            j = json.loads(f.read_text())
            for s, p in zip(j[0]["sequences"], paths):
                s["proteinChain"].update(p)
            f.write_text(json.dumps(j, indent=1))
        print(tid, "unpaired depths", depths, flush=True)
    (root / f"msa_depths.{i}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
