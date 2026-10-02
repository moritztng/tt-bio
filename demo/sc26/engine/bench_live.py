"""Seconds per fold by length on one warm chip, with the AICLK sampled during each fold.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 python3 demo/sc26/engine/bench_live.py --chip 0 \
        --out demo/sc26/engine/runs/live-chip0

Runs chipworker.py as a child, exactly as the server does, and writes every event it sends to
<out>/events.jsonl, one fold per <out>/<id>.jsonl (the replay format), and a summary to
<out>/summary.json. Lengths are prefixes of human serum albumin (585 aa), so every size is a
real sequence. Each length folds twice: the first pays any program compile for that size, the
second is the warm number the demo can promise. A last pair folds the same sequence with the
frame hook off and on, and compares the final coordinates byte for byte.
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HSA = ("DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCC"
       "AKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDNEETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLD"
       "ELRDEGKASSAKQRLKCASLQKFGERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKL"
       "KECCEKPLLEKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCCAAADPHE"
       "CYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCCKHPEAKRMPCAEDYLSVVLNQLC"
       "VLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICTLSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAF"
       "VEKCCKADDKETCFAEEGKKLVAASQAALGL")
GB1 = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="50,100,200,400")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    jobs = []
    for n in map(int, args.lengths.split(",")):
        for rep in ("cold", "warm"):
            jobs.append({"id": f"hsa{n}-{rep}", "sequence": HSA[:n], "seed": 0})
    jobs += [{"id": "gb1-hookoff", "sequence": GB1, "seed": 0, "frames": False},
             {"id": "gb1-hookon", "sequence": GB1, "seed": 0}]

    p = subprocess.Popen([sys.executable, "-u", str(here / "chipworker.py"), "--chip", str(args.chip)],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                         stderr=open(out / "worker.stderr", "w"))
    log = open(out / "events.jsonl", "w")
    per = {}
    done = {}
    pending = list(jobs)
    t0 = time.time()
    for line in p.stdout:
        log.write(line)
        log.flush()
        ev = json.loads(line)
        if ev.get("id"):
            per.setdefault(ev["id"], open(out / f"{ev['id']}.jsonl", "w")).write(line)
        if ev["type"] == "chip" and ev["state"] == "ready":
            print(f"[{time.time() - t0:7.1f}s] chip ready {ev}", flush=True)
            if pending:
                p.stdin.write(json.dumps(pending.pop(0)) + "\n")
                p.stdin.flush()
            else:
                p.stdin.write('{"type":"quit"}\n')
                p.stdin.flush()
        elif ev["type"] in ("fold_done", "fold_error"):
            done[ev["id"]] = ev
            per[ev["id"]].close()
            print(f"[{time.time() - t0:7.1f}s] {ev['type']} {ev['id']} "
                  f"{ev.get('seconds')} s stages={ev.get('stages')} aiclk={ev.get('aiclk_mhz')} "
                  f"{ev.get('reason', '')}", flush=True)
    p.wait()
    sha = lambda e: hashlib.sha256(base64.b64decode(e["xyz"])).hexdigest() if e and "xyz" in e else None
    summary = {
        "chip": args.chip, "rc": p.returncode, "loadavg_start": os.getloadavg(),
        "folds": {k: {f: v.get(f) for f in ("seconds", "stages", "aiclk_mhz", "n_res", "reason")}
                  for k, v in done.items()},
        "frames": {k: sum(1 for l in open(out / f"{k}.jsonl") if '"type":"frame"' in l) for k in done},
        "hook_off_sha": sha(done.get("gb1-hookoff")), "hook_on_sha": sha(done.get("gb1-hookon")),
    }
    summary["hook_bit_identical"] = summary["hook_off_sha"] == summary["hook_on_sha"] is not None
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
