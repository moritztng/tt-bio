"""Walk one model up the cdk2x2 size ladder on one Wormhole chip and record where it stops.

Answers two different questions that a single "it failed" does not distinguish, and that the
Wormhole campaign has to keep apart:

  * ONE OVERSIZED TENSOR -- a single allocation whose size is determined by the input shape.
    The refusal names a request that is large in absolute terms and the chip still has most of
    its memory free. Fixing it means changing that block's shape, and the wall is monotone in
    sequence length.
  * CUMULATIVE RESIDENCY -- the refusal names a SMALL request while the chip is nearly full.
    Nothing in particular is too big; the fold as a whole no longer fits. Fixing it means
    freeing something earlier, and the wall moves when unrelated code changes.

tt-metal's refusal carries both facts, so the classification is read off the message rather than
guessed: it prints the total request, the bank count, and the per-bank share. On a Wormhole
Galaxy chip that is 12 banks of ~1 GiB against Blackhole's 8 of 3.984 GiB, which is why an
interleaved allocation that fits on a p150a is refused here at a third of the total.

A run that neither succeeds nor is refused is a third outcome, kept separate: a timeout is a
runtime wall, not a memory one, and OpenDDE's 896 is exactly that (a 1500 s serving watchdog,
not an allocation). Conflating the two is how "opendde cannot do 1024" got written down when
the engine folds 1024 fine.

Results append to a JSONL file as each rung finishes, so a killed driver loses one rung, not
the ladder.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

#: tt-metal's allocator refusal. Both the DRAM and the L1 form carry the total request, the
#: number of banks it is spread over, and the per-bank share -- the three numbers that decide
#: whether a wall is one tensor or the sum of everything resident.
_OOM = re.compile(
    r"Out of Memory: Not enough space to allocate (\d+) B (DRAM|L1) buffer "
    r"across (\d+) banks, where each bank needs to store (\d+) B", re.S)

#: Written by the run itself. A rung counts as PASS only if a structure file exists, because a
#: zero exit status has been wrong here before (a worker that swallowed its own child's failure
#: still exited 0). The negative control for this check is any refused rung below: those write
#: no structure, and if this predicate ever passed one, it would be reading the wrong thing.
_STRUCT = ("*.cif", "*.pdb")


def _structures(out_dir: Path) -> list[Path]:
    return [p for pat in _STRUCT for p in out_dir.rglob(pat)]


def classify(stderr: str, rc: int, timed_out: bool) -> tuple[str, dict]:
    """Name the wall, and carry the numbers the name rests on."""
    if timed_out:
        return "TIMEOUT", {}
    m = _OOM.search(stderr)
    if m:
        total, space, banks, per_bank = int(m[1]), m[2], int(m[3]), int(m[4])
        return f"OOM_{space}", {
            "request_bytes": total, "banks": banks, "per_bank_bytes": per_bank,
            "request_gib": round(total / 2**30, 3), "per_bank_mib": round(per_bank / 2**20, 1),
        }
    if rc != 0:
        return "ERROR", {}
    return "NO_STRUCTURE", {}


def run_rung(model: str, yaml_path: Path, device: int, out_root: Path, timeout_s: int,
             env_extra: dict[str, str], extra_args: list[str]) -> dict:
    out_dir = out_root / f"{model}_{yaml_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # ttnn brings up every chip TT_VISIBLE_DEVICES makes visible, not just the one it computes
    # on, so the pin is what keeps this run inside its granted chip range.
    env["TT_VISIBLE_DEVICES"] = str(device)
    env["TT_BIO_LEASE_CARDS"] = str(device)
    env["TT_BIO_LEASE_HOLDER"] = "worker:wh-seqlen-structure"
    env.update(env_extra)
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(yaml_path),
           "--model", model, "--out_dir", str(out_dir), "--accelerator", "tenstorrent",
           "--override", "--debug", *extra_args]
    t0 = time.time()
    timed_out = False
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
        rc, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        rc, out, err = 124, e.stdout or "", e.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
    wall = round(time.time() - t0, 1)
    structs = _structures(out_dir)
    if structs and not timed_out and rc == 0:
        verdict, detail = "PASS", {"structure": str(structs[0])}
    else:
        verdict, detail = classify(out + err, rc, timed_out)
        detail = dict(detail)
    log = out_root / f"{model}_{yaml_path.stem}.log"
    log.write_text(out + "\n===STDERR===\n" + err)
    return {"model": model, "rung": yaml_path.stem, "device": device, "verdict": verdict,
            "wall_s": wall, "rc": rc, "log": str(log), **detail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--rungs", required=True, help="Comma-separated yaml paths, ascending.")
    ap.add_argument("--out", required=True, help="JSONL results file, appended to.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--stop-after-fail", type=int, default=1,
                    help="Stop the ladder after this many consecutive non-PASS rungs.")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the child.")
    ap.add_argument("extra", nargs="*", help="Extra args forwarded to tt-bio predict.")
    a = ap.parse_args()

    env_extra = dict(kv.split("=", 1) for kv in a.env)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fails = 0
    for r in a.rungs.split(","):
        row = run_rung(a.model, Path(r), a.device, Path(a.out_root), a.timeout,
                       env_extra, list(a.extra))
        row["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        fails = 0 if row["verdict"] == "PASS" else fails + 1
        if fails >= a.stop_after_fail:
            print(f"stopping {a.model}: {fails} consecutive non-PASS", flush=True)
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
