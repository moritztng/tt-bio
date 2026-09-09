"""Walk the Blackhole capacity ladder for the design and embedding models.

`scripts/capacity_gate.py` covers every FOLDING model at a 1536-token bar, and exempts the three
design models plus saprot-1.3b in writing because a 1536-residue FASTA is not a valid input for
any of them. This walks the axis each of those models is actually sized on, on the same hardware,
so the exemptions can be replaced by measured rows in `tt_bio/size_limits.CEILINGS`.

    python3 perf/bhdesign/ladder.py --model esmc-6b --sizes 1536,1968,2048 --out out.jsonl

One rung per SUBPROCESS, always: device state carried between rungs would decide the result
instead of the size, and an OOM leaves a dirty allocator behind. Each rung runs the shipped CLI
with the argv a user types, and passes only when the ARTIFACT is on disk with the right shape --
an exit code of 0 is not evidence (`verify-the-deployed-artifact-not-your-own-change`).

The axis per model, and why it is not the same axis:

    esmc-*, saprot-*   MAX_SEQUENCE -- residues in the longest single sequence. Independent
                       sequences, so what has to fit is the longest, not their sum.
    rfd3               DESIGN_TOTAL -- motif crop + designed length, everything it tokenises.
    pxdesign           DESIGN_TARGET -- conditioned target residues only; the binder is extra.
    boltzgen           atoms in the target, which is what its trunk Pairformer's triangle
                       attention is sized by. Residues would be the wrong denominator.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = os.environ.get("BH_PY", "/home/ttuser/tt-bio-dev/env/bin/python3")

# A real sequence rather than a homopolymer: a repeated single residue is not a workload any
# attention block sees, and a degenerate input can take a different path through a tokenizer.
_SEED_SEQ = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEG"
    "LYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDA"
    "KGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDR"
)


def seq_of(n: int) -> str:
    return (_SEED_SEQ * (n // len(_SEED_SEQ) + 1))[:n]


# --------------------------------------------------------------------------------------------
# Fixtures: one per axis, each producing the input the shipped CLI takes.
# --------------------------------------------------------------------------------------------

def fasta_fixture(work: pathlib.Path, n: int) -> pathlib.Path:
    p = work / f"seq{n}.fasta"
    p.write_text(f">L{n}\n{seq_of(n)}\n")
    return p


def cif_atoms(path: pathlib.Path) -> tuple[list[str], list[list[str]]]:
    """(loop_ column names, ATOM rows) of a small single-loop mmCIF.

    The column order is READ, never assumed. `perf/ceilrfd3/targets/*.cif` are biotite-written
    with `_atom_site.id` LAST, while `perf/pxdesign/make_targets.py` hardcodes the RCSB order
    where it comes second -- so a positional parser silently reads the wrong field on one of the
    two fixture families, which is how a residue count becomes a chain id.
    """
    cols, rows = [], []
    for line in path.read_text().splitlines():
        st = line.strip()
        if st.startswith("_atom_site."):
            cols.append(st.split(".", 1)[1].split()[0])
        elif st.startswith(("ATOM", "HETATM")):
            f = st.split()
            if len(f) == len(cols):
                rows.append(f)
    return cols, rows


def _col(cols, *names):
    for n in names:
        if n in cols:
            return cols.index(n)
    raise KeyError(names)


def cif_stats(path: pathlib.Path) -> tuple[int, int]:
    """(residues, atoms) of a CIF, counted off its ATOM records."""
    cols, rows = cif_atoms(path)
    if not rows:
        return 0, 0
    ch = _col(cols, "label_asym_id", "auth_asym_id")
    sq = _col(cols, "label_seq_id", "auth_seq_id")
    return len({(r[ch], r[sq]) for r in rows}), len(rows)


def rfd3_fixture(work: pathlib.Path, total: int, binder: int, target: pathlib.Path) -> pathlib.Path:
    """A contig spec whose DESIGN_TOTAL is `total`: `crop` motif residues plus `binder` designed.

    The crop is capped by the target chain, so past that the designed length carries the rest --
    which is the honest way to walk this axis above the largest target on hand, and still the
    number the model tokenises.
    """
    ntarget = cif_stats(target)[0]
    crop = min(total - binder, ntarget)
    length = total - crop
    spec_id = f"bh{total}"
    p = work / f"{spec_id}.json"
    p.write_text(json.dumps(
        {spec_id: {"input": str(target), "contig": f"A1-{crop},{length}", "length": str(length)}},
        indent=2))
    return p


def pxdesign_fixture(work: pathlib.Path, target_res: int, target: pathlib.Path,
                     binder: int = 80) -> pathlib.Path:
    p = work / f"px{target_res}.yaml"
    p.write_text(
        "target:\n"
        f'  file: "{target}"\n'
        "  chains:\n"
        "    A:\n"
        f"      crop: [\"1-{target_res}\"]\n"
        "      hotspots: [%d, %d, %d]\n" % (target_res // 2, target_res // 2 + 1, target_res // 2 + 2)
        + f"binder_length: {binder}\n")
    return p


def boltzgen_fixture(work: pathlib.Path, target_res: int, seq: str, binder: int = 80) -> pathlib.Path:
    p = work / f"bg{target_res}.yaml"
    p.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {seq[:target_res]}\n"
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: {'X' * binder}\n"
        "design:\n"
        "  - chain_id: B\n")
    return p


# --------------------------------------------------------------------------------------------
# Mechanism classification. Says WHY, not just WHERE -- the two OOM classes want different fixes.
# --------------------------------------------------------------------------------------------

_MECH = [
    ("dram", re.compile(r"Not enough space to allocate.*DRAM|out of memory.*DRAM", re.I)),
    ("l1", re.compile(r"Not enough space to allocate.*L1|L1 buffer|circular buffer", re.I)),
    ("host_oom", re.compile(r"MemoryError|Cannot allocate memory|Killed|std::bad_alloc", re.I)),
    ("fragmentation", re.compile(r"largest free block", re.I)),
    ("shape", re.compile(r"shape|dimension|assert.*tile|must be a multiple", re.I)),
]


def classify(text: str) -> str:
    for name, rx in _MECH:
        if rx.search(text):
            return name
    return "unknown"


def dram_numbers(text: str) -> dict:
    """The allocator's own numbers out of an OOM throw, so a row can say how far short it was."""
    out = {}
    m = re.search(r"allocate (\d+) B\b", text)
    if m:
        out["requested_b"] = int(m.group(1))
    m = re.search(r"(\d+) B (?:of )?(?:free|available)", text)
    if m:
        out["free_b"] = int(m.group(1))
    m = re.search(r"largest free block[^\d]*(\d+)", text, re.I)
    if m:
        out["largest_free_b"] = int(m.group(1))
    return out


# --------------------------------------------------------------------------------------------
# One rung.
# --------------------------------------------------------------------------------------------

def run_rung(model: str, size: int, args, work: pathlib.Path) -> dict:
    out_dir = work / f"out_{model}_{size}"
    subprocess.run(["rm", "-rf", str(out_dir)], check=False)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["TT_VISIBLE_DEVICES"] = str(args.card)
    env["TT_BIO_LEASE_CARDS"] = str(args.card)
    env["TT_BIO_LEASE_HOLDER"] = args.holder
    env["TT_BIO_SIZE_LIMIT"] = "0"   # the ladder is what MEASURES the ceiling; it cannot obey one

    base = [PY, "-u", "-m", "tt_bio.main"]
    if model.startswith("esmc"):
        fx = fasta_fixture(work, size)
        cmd = base + ["embed", str(fx), "--model", model, "--out_dir", str(out_dir),
                      "--batch_size", "1"]
        checker = ("npz", size)
    elif model.startswith("saprot"):
        fx = fasta_fixture(work, size)
        cmd = base + ["saprot", str(fx), "--model", model, "--out_dir", str(out_dir),
                      "--batch_size", "1"]
        checker = ("npz", size)
    elif model == "rfd3":
        fx = rfd3_fixture(work, size, args.binder, pathlib.Path(args.target))
        cmd = base + ["design", str(fx), "--model", "rfd3", "--from_pdb", "--out_dir", str(out_dir),
                      "--num_timesteps", str(args.steps), "--num_designs", "1",
                      "--devices", str(args.card)]
        checker = ("cif", size)
    elif model == "pxdesign":
        fx = pxdesign_fixture(work, size, pathlib.Path(args.target), args.binder)
        cmd = base + ["design", str(fx), "--model", "pxdesign", "--out_dir", str(out_dir),
                      "--n_step", str(args.steps), "--num_designs", "1", "--devices", str(args.card)]
        checker = ("cif", size + args.binder)
    elif model == "boltzgen":
        fx = boltzgen_fixture(work, size, seq_of(size), args.binder)
        cmd = base + ["design", str(fx), "--model", "boltzgen", "--out_dir", str(out_dir),
                      "--num_designs", "1", "--steps", "design", "--devices", str(args.card),
                      "--debug"]
        checker = ("any", 0)
    else:
        raise SystemExit(f"no fixture for {model}")

    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                           timeout=args.timeout)
        rc, blob = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc = -9
        blob = ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes)
                else (e.stdout or "")) + "\nTIMEOUT"
    wall = round(time.time() - t0, 1)

    rec = {"model": model, "size": size, "rc": rc, "wall_s": wall,
           "cmd": " ".join(cmd[3:]), "arch": "blackhole", "card": args.card,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    ok, detail = check_artifact(checker, out_dir, model)
    rec["artifact"] = detail
    if ok and rc == 0:
        rec["verdict"] = "PASS"
        rec["mechanism"] = "none"
    else:
        rec["verdict"] = "FAIL"
        rec["mechanism"] = classify(blob)
        rec.update(dram_numbers(blob))
        rec["tail"] = blob[-2500:]
    return rec


def check_artifact(checker, out_dir: pathlib.Path, model: str) -> tuple[bool, dict]:
    """The rung's real verdict. Every branch can FAIL -- that is what makes it a check."""
    kind, expect = checker
    if not out_dir.is_dir():
        return False, {"reason": "no output directory"}
    if kind == "npz":
        import numpy as np
        files = sorted(out_dir.glob("*.npz"))
        if not files:
            return False, {"reason": "no .npz written"}
        d = np.load(files[0])
        if "per_residue" not in d:
            return False, {"reason": "npz has no per_residue array"}
        pr = d["per_residue"]
        detail = {"npz": files[0].name, "per_residue": list(pr.shape),
                  "finite": bool(np.isfinite(pr).all()),
                  "nonzero_frac": round(float((pr != 0).mean()), 4)}
        # The negative control: a padded tail that never got masked comes back as rows of zeros,
        # and a wrong bucket comes back with the wrong row count. Both fail here.
        ok = (pr.shape[0] == expect and detail["finite"] and detail["nonzero_frac"] > 0.99)
        return ok, detail
    if kind == "cif":
        cifs = sorted(out_dir.rglob("*.cif"))
        if not cifs:
            return False, {"reason": "no .cif written"}
        res, atoms = cif_stats(cifs[0])
        detail = {"cif": cifs[0].name, "residues": res, "atoms": atoms}
        return (res == expect and atoms > 3 * res), detail
    files = [p for p in out_dir.rglob("*") if p.is_file() and p.stat().st_size > 0]
    return bool(files), {"files": len(files),
                         "names": sorted(p.name for p in files)[:6]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sizes", required=True, help="comma-separated rungs on this model's own axis")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--card", default=os.environ.get("BH_CARD", "1"))
    ap.add_argument("--holder", default="worker:bh-1536-design-embed")
    ap.add_argument("--target", default="perf/ceilrfd3/targets/laczc_1008.cif")
    ap.add_argument("--binder", type=int, default=80)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="stop the walk at the first FAIL (the ceiling is below it)")
    ap.add_argument("--work", type=pathlib.Path, default=ROOT / "perf" / "bhdesign" / "work")
    a = ap.parse_args()

    a.work.mkdir(parents=True, exist_ok=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for size in [int(s) for s in a.sizes.split(",") if s.strip()]:
        rec = run_rung(a.model, size, a, a.work)
        line = json.dumps(rec)
        with a.out.open("a") as fh:
            fh.write(line + "\n")
        print(f"[{rec['ts']}] {a.model} {size}: {rec['verdict']} "
              f"mech={rec['mechanism']} {rec['wall_s']}s {rec['artifact']}", flush=True)
        if rec["verdict"] != "PASS":
            print(rec.get("tail", "")[-1200:], flush=True)
            if a.stop_on_fail:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
