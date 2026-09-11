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
import shutil
import subprocess
import sys
import time
from pathlib import Path

#: tt-metal's allocator refusal. Both the DRAM and the L1 form carry the total request, the
#: number of banks it is spread over, and the per-bank share -- the three numbers that decide
#: whether a wall is one tensor or the sum of everything resident.
#: The per-bank clause is optional and the message WRAPS -- tt-metal prints "across 12 banks,"
#: then a newline and indentation before "where each bank needs to store". A pattern that
#: assumed one line matched nothing, which is worse than no classifier: every wall would have
#: read as a plain ERROR and the ladder would have run for hours collecting nothing. Both real
#: forms are in tests/test_whceil_classify.py.
_OOM = re.compile(
    r"Out of Memory: Not enough space to allocate\s+(\d+) B (DRAM|L1) buffer\s+"
    r"across\s+(\d+) banks"
    r"(?:,\s*where each bank needs to store\s+(\d+) B)?", re.S)

#: The parenthetical tt-metal puts LAST, and the only part of the refusal that says how full
#: the chip was. Without it the two wall classes are indistinguishable: a per-bank need that
#: exceeds the bank outright is one oversized tensor, and a small need refused with plenty free
#: but no run to put it in is fragmentation. Optional, because not every refusal carries it.
_OOM_STATE = re.compile(
    r"bank size is\s+(\d+) B\s*\(allocated:\s*(\d+) B,\s*free:\s*(\d+) B,"
    r"\s*largest free block:\s*(\d+) B\)", re.S)


#: The other refusal, and a different failure entirely: L1, not DRAM, and a CLASH rather than a
#: shortage -- a statically allocated circular buffer region overlapping an L1 buffer. It is the
#: class the ceiling table calls non-monotone (OpenDDE folds 544, throws at 576, folds 608),
#: because what clashes depends on what else is resident rather than on the request's size. It
#: does not carry bank numbers, so it can never be classified by the DRAM rule above.
_L1_CLASH = re.compile(
    r"Statically allocated circular buffers in program (\d+) clash with L1 buffers"
    r"(?:.*?L1 buffer allocated at (\d+) and static circular buffer region ends at (\d+))?",
    re.S)

#: The OTHER static-CB failure, and a different thing again: the region does not clash with a
#: neighbour, it does not fit L1 at all. Seen on OpenDDE's 1088 rung, 2471200 B asked of a
#: 1499136 B L1. Unlike the clash this one IS shape-determined -- it is an oversized tensor in
#: L1 rather than in DRAM -- so it gets its own name instead of being folded into the clash.
_L1_OVERSIZE = re.compile(
    r"Statically allocated circular buffers on core range \[[^\]]*\] grow to (\d+) B "
    r"which is beyond max L1 size of (\d+) B", re.S)


def _wall_kind(per_bank: int, bank_size: int | None, free: int | None,
               largest_free: int | None) -> str:
    """Which of the brief's two wall classes this refusal is.

    ONE_OVERSIZED_TENSOR: the per-bank share does not fit an EMPTY bank. No amount of freeing
    helps; the block's shape has to change.
    CUMULATIVE_RESIDENCY: it would fit an empty bank, and there is not enough free.
    FRAGMENTATION: there IS enough free, just not in one run. Kept apart from cumulative
    residency because the fix is different -- compaction rather than holding less.
    """
    if bank_size is None:
        return "UNCLASSIFIED"
    if per_bank > bank_size:
        return "ONE_OVERSIZED_TENSOR"
    if free is not None and per_bank > free:
        return "CUMULATIVE_RESIDENCY"
    if largest_free is not None and per_bank > largest_free:
        return "FRAGMENTATION"
    return "UNCLASSIFIED"

#: What each command writes ONLY when it produced an answer, and the argv shape it takes. A rung
#: counts as PASS only if that artifact exists, because a zero exit status has been wrong here
#: before (a worker that swallowed its own child's failure still exited 0). The negative control
#: is any refused rung: those write none of these.
#:
#: `affinity` deliberately does NOT key on affinity.csv -- that file is written with an `error`
#: column even when every input failed, so it would pass a rung that scored nothing. The
#: per-input json is only written for an input that scored.
_COMMANDS = {
    "predict": (("*.cif", "*.pdb"),
                lambda out: ["--out_dir", str(out), "--accelerator", "tenstorrent",
                             "--override", "--debug"]),
    "affinity": (("*_affinity.json",),
                 lambda out: ["--out_dir", str(out), "--accelerator", "tenstorrent"]),
}


def _artifacts(out_dir: Path, command: str) -> list[Path]:
    return [p for pat in _COMMANDS[command][0] for p in out_dir.rglob(pat)]


def classify(stderr: str, rc: int, timed_out: bool) -> tuple[str, dict]:
    """Name the wall, and carry the numbers the name rests on."""
    if timed_out:
        return "TIMEOUT", {}
    o = _L1_OVERSIZE.search(stderr)
    if o and not _OOM.search(stderr):
        return "OVERSIZE_L1", {"wall_kind": "ONE_OVERSIZED_TENSOR_L1",
                               "request_bytes": int(o[1]), "l1_size_bytes": int(o[2])}
    c = _L1_CLASH.search(stderr)
    if c and not _OOM.search(stderr):
        d = {"wall_kind": "L1_CB_CLASH", "program": int(c[1])}
        if c[2]:
            d.update(l1_buffer_at=int(c[2]), cb_region_ends=int(c[3]))
        return "CLASH_L1", d
    m = _OOM.search(stderr)
    if m:
        total, space, banks = int(m[1]), m[2], int(m[3])
        # Absent, the per-bank share is the interleaved split, which is what the allocator
        # refuses on. Derived rather than dropped, and flagged so a reader knows which it is.
        per_bank = int(m[4]) if m[4] else -(-total // banks)
        d = {
            "request_bytes": total, "banks": banks, "per_bank_bytes": per_bank,
            "per_bank_reported": m[4] is not None,
            "request_gib": round(total / 2**30, 3), "per_bank_mib": round(per_bank / 2**20, 1),
            "wall_kind": "UNCLASSIFIED",
        }
        st = _OOM_STATE.search(stderr[m.start():])
        if st:
            bank_size, alloc, free, largest = (int(st[i]) for i in (1, 2, 3, 4))
            d.update(bank_size_bytes=bank_size, allocated_bytes=alloc, free_bytes=free,
                     largest_free_block_bytes=largest,
                     bank_size_mib=round(bank_size / 2**20, 1),
                     free_mib=round(free / 2**20, 1),
                     largest_free_mib=round(largest / 2**20, 1),
                     wall_kind=_wall_kind(per_bank, bank_size, free, largest))
        return f"OOM_{space}", d
    if rc != 0:
        return "ERROR", {}
    return "NO_STRUCTURE", {}


def run_rung(model: str, yaml_path: Path, device: int, out_root: Path, timeout_s: int,
             env_extra: dict[str, str], extra_args: list[str], command: str = "predict") -> dict:
    out_dir = out_root / f"{model}_{yaml_path.stem}"
    # Emptied, not reused. The PASS predicate is "a structure file exists", and the out dir is
    # keyed by (model, rung) -- so a rerun of a rung that PASSED once and now fails would find
    # the old .cif and report PASS. That is the exact shape of a check that cannot fail, and a
    # relaunch of this ladder is a rerun by construction.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    env = dict(os.environ)
    # ttnn brings up every chip TT_VISIBLE_DEVICES makes visible, not just the one it computes
    # on, so the pin is what keeps this run inside its granted chip range.
    env["TT_VISIBLE_DEVICES"] = str(device)
    env["TT_BIO_LEASE_CARDS"] = str(device)
    env["TT_BIO_LEASE_HOLDER"] = "worker:wh-seqlen-structure"
    env.update(env_extra)
    cmd = [sys.executable, "-m", "tt_bio.main", command, str(yaml_path), "--model", model,
           *_COMMANDS[command][1](out_dir), *extra_args]
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
    structs = _artifacts(out_dir, command)
    if structs and not timed_out and rc == 0:
        verdict, detail = "PASS", {"structure": str(structs[0])}
    else:
        verdict, detail = classify(out + err, rc, timed_out)
        detail = dict(detail)
    # One log per ATTEMPT, not per (model, rung). Reusing the name let a later attempt at the
    # same rung overwrite the log an earlier row points at, and the collector then read the
    # NEW run's recovered-from refusal as the OLD run's cause of death -- which is how a guard
    # refusal and a missing shared library both came back reported as a 2 GiB DRAM wall.
    log = out_root / f"{model}_{yaml_path.stem}_{time.strftime('%H%M%S', time.gmtime(t0))}.log"
    body = out + "\n===STDERR===\n" + err
    log.write_text(body)
    row = {"model": model, "command": command, "rung": yaml_path.stem, "device": device, "verdict": verdict,
           "wall_s": wall, "rc": rc, "log": str(log), **detail}
    # A refusal the engine recovered from is not a wall, but it is the single best evidence
    # that the reactive narrowing is doing its job -- and it is invisible in a PASS row
    # otherwise. Counted for every outcome, named separately from the one that killed the run.
    hits = [int(m[1]) for m in _OOM.finditer(body)]
    clashes = len(_L1_CLASH.findall(body)) + len(_L1_OVERSIZE.findall(body))
    if verdict == "PASS" and (hits or clashes):
        row["refusals_recovered"] = len(hits)
        row["clashes_recovered"] = clashes
        if hits:
            row["largest_recovered_bytes"] = max(hits)
    return row


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
    ap.add_argument("--command", choices=sorted(_COMMANDS), default="predict",
                    help="tt-bio subcommand. nesso1 scores through `affinity`, not `predict`.")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE for the child.")
    ap.add_argument("extra", nargs="*", help="Extra args forwarded to tt-bio predict.")
    a = ap.parse_args()

    env_extra = dict(kv.split("=", 1) for kv in a.env)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fails = 0
    for r in a.rungs.split(","):
        row = run_rung(a.model, Path(r), a.device, Path(a.out_root), a.timeout,
                       env_extra, list(a.extra), command=a.command)
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
