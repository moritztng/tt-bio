#!/usr/bin/env python3
"""Is a fold reproducible across two chips of the same Wormhole Galaxy?

Determinism is part of correctness: the same input, same seed and same model must give the
same answer on chip A and chip B, or a published accuracy number only describes the chip it
was measured on. The integration run showed 32 chips producing bit-identical ESMFold2
structures; this asks the same question of the other models.

The assertion is on the ARTIFACT: the sha256 of every CIF the fold wrote, plus the max
absolute coordinate delta after parsing. A hash match is bit-identical output; a hash
mismatch with a small coordinate delta is a real but bounded numeric difference, which is a
different (and much worse) claim than "deterministic".

Every run also carries its own negative control: the same chip folded at a DIFFERENT seed.
If that control does not differ, the comparison is reading something that does not depend on
the computation (a stale out_dir, a cached result, a constant), and the run is void.

Anything after a bare `--` is passed through to `tt-bio predict` unchanged. It is a
passthrough rather than a --predict-args string because argparse reads a lone value that
starts with a dash as an option: --predict-args "--single_sequence" fails, while
--predict-args "--single_sequence --recycling_steps 3" happens to work, since argparse only
exempts a value containing a space. A flag whose correctness depends on how many words the
caller passed is a trap, and it cost this campaign two determinism legs.

  PYTHONPATH=<checkout> python3 perf/wh-parity/chip_determinism.py \
      --model boltz2 --input examples/hsa_no_msa.yaml --cards 25,26 \
      --out-dir /home/mthuening/scratch/det/boltz2-hsa --json out.json \
      -- --single_sequence --sampling_steps 200 --recycling_steps 3
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
COORD = re.compile(r"^ATOM|^HETATM")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cif_coords(path: Path) -> list[tuple[float, float, float]]:
    """Cartesian x/y/z out of an mmCIF atom_site loop, in file order.

    Parsed positionally off the loop header rather than with a CIF library: the comparison
    must not depend on a parser that could normalise (round, reorder) the very differences
    being measured.
    """
    lines = path.read_text().splitlines()
    tags: list[str] = []
    out: list[tuple[float, float, float]] = []
    in_loop = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("_atom_site."):
            tags.append(s.split(".", 1)[1].split()[0])
            in_loop = True
            continue
        if in_loop and tags:
            if not s or s.startswith(("#", "loop_", "_", "data_")):
                if out:
                    break
                in_loop = False
                tags = []
                continue
            parts = s.split()
            if len(parts) < len(tags):
                continue
            try:
                ix, iy, iz = (tags.index("Cartn_x"), tags.index("Cartn_y"), tags.index("Cartn_z"))
            except ValueError:
                break
            try:
                out.append((float(parts[ix]), float(parts[iy]), float(parts[iz])))
            except ValueError:
                continue
    return out


def structures(out_dir: Path) -> dict[str, Path]:
    """Every structure the fold wrote, keyed by name relative to out_dir."""
    found = {}
    for pat in ("**/*.cif", "**/*.pdb"):
        for p in sorted(out_dir.glob(pat)):
            found[str(p.relative_to(out_dir))] = p
    return found


def fold(model: str, inp: str, card: int, seed: int, out_dir: Path, extra: list[str],
         python: str, timeout: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python, "-u", "-m", "tt_bio.main", "predict", inp, "--model", model,
           "--out_dir", str(out_dir), "--seed", str(seed), "--device_ids", str(card),
           "--override", *extra]
    env = dict(os.environ)
    # One device context per process, and the lease refuses an unpinned open: a process that
    # can see every chip brings up every chip, so the pin is what keeps this to one card.
    env["TT_VISIBLE_DEVICES"] = str(card)
    env["TT_BIO_LEASE_CARDS"] = str(card)
    env["PYTHONPATH"] = str(REPO)
    t0 = time.time()
    log = out_dir / "fold.log"
    with log.open("w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            timeout=timeout, cwd=str(REPO), env=env).returncode
    return {"card": card, "seed": seed, "rc": rc, "wall_s": round(time.time() - t0, 1),
            "out_dir": str(out_dir), "log": str(log),
            "cmd": " ".join(shlex.quote(c) for c in cmd)}


def compare(a: Path, b: Path) -> dict:
    sa, sb = structures(a), structures(b)
    common = sorted(set(sa) & set(sb))
    if not common:
        return {"comparable": False, "reason": f"no common structure file ({sorted(sa)} vs {sorted(sb)})"}
    rows = []
    for name in common:
        ha, hb = sha256(sa[name]), sha256(sb[name])
        ca, cb = cif_coords(sa[name]), cif_coords(sb[name])
        if len(ca) != len(cb):
            rows.append({"file": name, "sha_match": ha == hb, "atoms_a": len(ca),
                         "atoms_b": len(cb), "max_abs_delta": None,
                         "note": "atom count differs"})
            continue
        dmax = max((max(abs(x - y) for x, y in zip(p, q)) for p, q in zip(ca, cb)), default=0.0)
        rows.append({"file": name, "sha_match": ha == hb, "atoms": len(ca),
                     "max_abs_delta": dmax})
    return {"comparable": True, "files": rows,
            "all_sha_match": all(r["sha_match"] for r in rows),
            "max_abs_delta": max((r["max_abs_delta"] or 0.0) for r in rows),
            "missing_on_one_side": sorted(set(sa) ^ set(sb))}


def parse_with_passthrough(ap: argparse.ArgumentParser, argv: list[str]):
    """Split argv on the first bare `--`; everything after it is passed to the fold verbatim.

    Doing the split before argparse sees the list is what makes a passthrough arg that starts
    with a dash safe. argparse's own handling of such a value is length-dependent, so it works
    until the day the caller passes a single flag.
    """
    if "--" in argv:
        cut = argv.index("--")
        return ap.parse_args(argv[:cut]), argv[cut + 1:]
    return ap.parse_args(argv), []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True, help="yaml/fasta the CLI accepts")
    ap.add_argument("--cards", required=True, help="two chip ids, e.g. 25,26")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control-seed", type=int, default=None,
                    help="seed for the negative control on card A (default: --seed + 1)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--timeout", type=float, default=5400.0)
    ap.add_argument("--skip-control", action="store_true",
                    help="only for a model whose sampler is seed-independent; the run then "
                         "proves nothing about the comparison's sensitivity and says so")
    args, extra = parse_with_passthrough(ap, sys.argv[1:])

    cards = [int(c) for c in args.cards.split(",") if c.strip()]
    if len(cards) != 2:
        raise SystemExit("--cards takes exactly two chip ids")
    root = Path(args.out_dir)
    ctrl_seed = args.seed + 1 if args.control_seed is None else args.control_seed

    jobs = [(cards[0], args.seed, root / f"card{cards[0]}_seed{args.seed}"),
            (cards[1], args.seed, root / f"card{cards[1]}_seed{args.seed}")]
    if not args.skip_control:
        jobs.append((cards[0], ctrl_seed, root / f"card{cards[0]}_seed{ctrl_seed}"))

    result = {"model": args.model, "input": args.input, "cards": cards, "seed": args.seed,
              "control_seed": None if args.skip_control else ctrl_seed,
              "predict_args": shlex.join(extra), "folds": []}
    # The two same-seed folds go in parallel (different chips, one context each); the control
    # runs on card A afterwards so it never shares a chip with a live fold.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(fold, args.model, args.input, c, s, d, extra, args.python, args.timeout)
                for c, s, d in jobs[:2]]
        for f in futs:
            result["folds"].append(f.result())
    for c, s, d in jobs[2:]:
        result["folds"].append(fold(args.model, args.input, c, s, d, extra, args.python,
                                    args.timeout))

    bad = [f for f in result["folds"] if f["rc"] != 0]
    if bad:
        result["verdict"] = "ERROR"
        result["detail"] = f"{len(bad)} fold(s) exited nonzero: " + ", ".join(
            f"card{f['card']}/seed{f['seed']} rc={f['rc']} ({f['log']})" for f in bad)
    else:
        cross = compare(jobs[0][2], jobs[1][2])
        result["cross_chip"] = cross
        if not args.skip_control:
            ctrl = compare(jobs[0][2], jobs[2][2])
            result["control_same_chip_other_seed"] = ctrl
            control_ok = ctrl.get("comparable") and not ctrl.get("all_sha_match")
        else:
            control_ok = None
        if not cross.get("comparable"):
            result["verdict"] = "ERROR"
            result["detail"] = cross["reason"]
        elif control_ok is False:
            result["verdict"] = "VOID"
            result["detail"] = ("negative control did not differ: seed "
                                f"{ctrl_seed} on card {cards[0]} produced byte-identical "
                                "structures, so this comparison cannot detect a difference")
        elif cross["all_sha_match"]:
            result["verdict"] = "BIT-IDENTICAL"
            result["detail"] = (f"card {cards[0]} vs {cards[1]}: every structure sha256-equal"
                               + ("" if control_ok else "; NO negative control (--skip-control)"))
        else:
            result["verdict"] = "DIVERGENT"
            result["detail"] = (f"card {cards[0]} vs {cards[1]}: sha differs, max abs coordinate "
                                f"delta {result['cross_chip']['max_abs_delta']:.6g} A")

    print(json.dumps(result, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["verdict"] == "BIT-IDENTICAL" else 1


if __name__ == "__main__":
    sys.exit(main())
