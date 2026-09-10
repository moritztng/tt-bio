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


def cif_chains(path: pathlib.Path) -> dict[str, int]:
    """{chain id: residues} off the ATOM records."""
    cols, rows = cif_atoms(path)
    ch = _col(cols, "label_asym_id", "auth_asym_id")
    sq = _col(cols, "label_seq_id", "auth_seq_id")
    seen: dict[str, set] = {}
    for r in rows:
        seen.setdefault(r[ch], set()).add(r[sq])
    return {k: len(v) for k, v in seen.items()}


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
    """A PXDesign target YAML conditioning on the first `target_res` residues of `target`.

    The axis runs past any single chain on hand, so the crop SPILLS across chains in file
    order: chain A whole, then as much of B as is left, and so on. `read_design_yaml` takes a
    per-chain crop natively. Only the chains a rung actually wants are listed, because a chain
    listed with no crop conditions on the whole thing and would put the rung above its own size.
    """
    remaining = target_res
    lines = ["target:", f'  file: "{target}"', "  chains:"]
    first = True
    for cid, n in cif_chains(target).items():
        if remaining <= 0:
            break
        take = min(n, remaining)
        lines.append(f"    {cid}:")
        lines.append(f'      crop: ["1-{take}"]')
        if first:
            mid = take // 2
            lines.append(f"      hotspots: [{mid}, {mid + 1}, {mid + 2}]")
            first = False
        remaining -= take
    if remaining > 0:
        raise SystemExit(f"pxdesign_fixture: {target} holds {target_res - remaining} residues, "
                         f"{target_res} asked for")
    p = work / f"px{target_res}.yaml"
    p.write_text("\n".join(lines) + f"\nbinder_length: {binder}\n")
    return p


def crop_cif(src: pathlib.Path, n_res: int, dst: pathlib.Path) -> tuple[int, int]:
    """Write the first `n_res` residues of `src` to `dst`. Returns (residues, atoms).

    The atom axis is walked by cropping a REAL structure, never by generating one: a synthetic
    backbone carries ~4 atoms per residue where a deposited one carries ~8, so a ladder built on
    a generated target would report an atom ceiling about twice the one the model really reaches.
    """
    cols, rows = cif_atoms(src)
    ch = _col(cols, "label_asym_id", "auth_asym_id")
    sq = _col(cols, "label_seq_id", "auth_seq_id")
    keep, seen = [], {}
    for r in rows:
        key = (r[ch], r[sq])
        if key not in seen:
            if len(seen) >= n_res:
                break
            seen[key] = True
        keep.append(r)
    idx = cols.index("id") if "id" in cols else None
    out = [f"data_{dst.stem}", "#", "loop_"] + [f"_atom_site.{c} " for c in cols]
    for i, r in enumerate(keep, 1):
        r = list(r)
        if idx is not None:
            r[idx] = str(i)      # renumber, or the crop carries the parent's atom serials
        out.append(" ".join(r))
    raw = dst.with_suffix(".raw.cif")
    raw.write_text("\n".join(out) + "\n#\n")

    # Re-emit through gemmi. The targets on hand are biotite-written and carry `_atom_site`
    # and nothing else, but BoltzGen's mmCIF reader indexes `_entity_poly_seq.entity_id`
    # directly and raises "not found in block" on an atom-only file. `setup_entities()`
    # derives that loop from the chains, so the fixture is well-formed rather than
    # hand-decorated with metadata this ladder would be inventing.
    import gemmi
    st = gemmi.read_structure(str(raw))
    st.setup_entities()
    # setup_entities() assigns the entity but leaves full_sequence EMPTY, and gemmi omits an
    # empty loop on write -- so the category exists in the document and no rows reach the file.
    # Filling it from the residues actually present is what puts _entity_poly_seq on disk.
    for ent in st.entities:
        ent.full_sequence = [r.name for ch in st[0] for r in ch if r.subchain in ent.subchains]
    st.make_mmcif_document().write_file(str(dst))
    if "_entity_poly_seq.entity_id" not in dst.read_text():
        raise SystemExit(f"crop_cif: {dst} has no _entity_poly_seq loop; BoltzGen will refuse it")
    raw.unlink()

    res, atoms = cif_stats(dst)
    if (res, atoms) != (len(seen), len(keep)):
        raise SystemExit(f"crop_cif: gemmi round-trip changed the crop, "
                         f"{len(seen)}res/{len(keep)}atoms -> {res}res/{atoms}atoms")
    return res, atoms


def boltzgen_fixture(work: pathlib.Path, target_res: int, target: pathlib.Path,
                     binder: int = 80) -> tuple[pathlib.Path, int]:
    """A BoltzGen design spec against the first `target_res` residues of `target`.

    Returns the YAML and the target's ATOM count -- the axis this model is sized on. BoltzGen
    takes the `entities:` grammar (a designed chain given as a length, plus a structure file to
    include a chain from), NOT the `sequences:` schema `tt-bio predict` takes; the two are
    different parsers and the predict spelling is silently a different model's input.
    """
    cif = work / f"bgt{target_res}.cif"
    _, atoms = crop_cif(target, target_res, cif)
    p = work / f"bg{target_res}.yaml"
    # `include` defaults to "all" (`data/parse/schema.py`) and the crop above already holds
    # exactly the residues this rung wants. Naming chain A explicitly drops every residue past
    # the first chain the moment the ladder runs above one chain's length. The designed chain is
    # `Z` so it cannot collide with a target chain id once the target carries more than one.
    p.write_text(
        "entities:\n"
        "  - protein:\n"
        "      id: Z\n"
        f"      sequence: {binder}\n"
        "  - file:\n"
        f"      path: {cif.name}\n"
        "      include: all\n")
    return p, atoms

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
# `timeout` is set directly by the runner, never matched here: the harness knows it killed the
# run, and no log pattern can tell that from a throw the run recovered from.


def classify(text: str) -> str:
    for name, rx in _MECH:
        if rx.search(text):
            return name
    return "unknown"


# The lines a reader actually wants out of a failure. ttnn prints a ~200-frame backtrace after
# every throw, so a fixed tail of the blob is backtrace and nothing else: the first pass's FAIL
# rows carry 2500 characters of symbol names and not one word of what the allocator said. These
# patterns pull the message itself out of wherever it landed.
_SAY = re.compile(r"Not enough space to allocate|Out of Memory|TT_FATAL|TT_THROW|"
                  r"^\w*(Error|Exception)\b|Traceback|RuntimeError|MemoryError|Killed", re.M)


def throw_lines(text: str, keep: int = 12) -> list:
    """The failure's own words, backtrace stripped. A row that only carries a backtrace cannot
    say whether it hit a wall or a wedge, and those want opposite responses."""
    out = []
    for line in text.splitlines():
        st = line.strip()
        if not st or st.startswith("---") or st.startswith("["):
            continue        # backtrace frames and the MPI-style per-signal dump
        if _SAY.search(st) and st not in out:
            out.append(st[:400])
            if len(out) >= keep:
                break
    return out


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
    extra: dict = {}
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
                      "--num_timesteps", str(args.steps), "--num_designs", "1"]
        checker = ("cif", size)
    elif model == "pxdesign":
        fx = pxdesign_fixture(work, size, pathlib.Path(args.target), args.binder)
        cmd = base + ["design", str(fx), "--model", "pxdesign", "--out_dir", str(out_dir),
                      "--n_step", str(args.steps), "--num_designs", "1"]
        checker = ("binder", args.binder)
    elif model == "boltzgen":
        fx, atoms = boltzgen_fixture(work, size, pathlib.Path(args.target), args.binder)
        # `--devices` is an ALIAS FOR `--device_ids` on `tt-bio design` (main.py: the option
        # carries both spellings and the dest is `devices`), so it is a comma-separated list of
        # card IDs, not a count. Passing a literal "1" pins the run to card 1 whatever card the
        # task was granted, and every other grant dies at startup with "Requested Tenstorrent
        # device id(s) [1] not available". It reads as a count and only ever worked because the
        # first pass held card 1.
        cmd = base + ["design", str(fx), "--model", "boltzgen", "--out_dir", str(out_dir),
                      "--num_designs", "1", "--steps", "design", "--devices", str(args.card),
                      "--debug"]
        checker = ("designcif", args.binder)
        extra = {"target_atoms": atoms}
    else:
        raise SystemExit(f"no fixture for {model}")

    t0 = time.time()
    timed_out = False
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True,
                           timeout=args.timeout)
        rc, blob = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc = -9
        timed_out = True
        blob = ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes)
                else (e.stdout or "")) + "\nTIMEOUT"
    wall = round(time.time() - t0, 1)

    rec = {"model": model, "size": size, "rc": rc, "wall_s": wall,
           "cmd": " ".join(cmd[3:]), "arch": "blackhole", "card": args.card,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}

    ok, detail = check_artifact(checker, out_dir, model)
    rec["artifact"] = detail
    if ok and rc == 0:
        rec["verdict"] = "PASS"
        rec["mechanism"] = "none"
    else:
        rec["verdict"] = "FAIL"
        # A run the harness killed on ITS OWN clock has no mechanism in the log, and classify()
        # will happily label it off whatever allocator line the run printed and recovered from --
        # boltzgen at 20171 atoms came back `l1` off an absorbed throw when what actually happened
        # is that 3000 s ran out. The wall-clock budget is the ladder's choice, so it is named as
        # such and not as a property of the model.
        # rc -15 is an operator SIGTERM: somebody stopped the rung, the model did not end it.
        # Without this the label comes from classify(), which reads whatever throw the run
        # printed and recovered from -- so a rung stopped by hand at 30 minutes gets filed under
        # the absorbed L1 message as if that had been terminal. Same conflation as the timeout
        # case, different signal.
        if timed_out:
            rec["mechanism"] = "timeout"
            rec["timeout_s"] = args.timeout
        elif rc == -15:
            rec["mechanism"] = "killed"
        else:
            rec["mechanism"] = classify(blob)
        rec.update(dram_numbers(blob))
        rec["throw"] = throw_lines(blob)
        rec["tail"] = blob[-2500:]
    return rec


def rescore_rung(model: str, size: int, args, work: pathlib.Path) -> dict:
    """Re-verify a rung from the artifacts on disk. No subprocess, no device."""
    out_dir = work / f"out_{model}_{size}"
    checker = {"rfd3": ("cif", size), "pxdesign": ("binder", args.binder),
               "boltzgen": ("designcif", args.binder)}.get(model, ("npz", size))
    rec = {"model": model, "size": size, "rc": None, "wall_s": 0.0,
           "cmd": "(rescored from artifacts on disk)", "arch": "blackhole", "card": args.card,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rescored": True}
    if model == "boltzgen":
        cif = work / f"bgt{size}.cif"
        if cif.exists():
            rec["target_atoms"] = cif_stats(cif)[1]
    ok, detail = check_artifact(checker, out_dir, model)
    rec["artifact"] = detail
    rec["verdict"] = "PASS" if ok else "FAIL"
    rec["mechanism"] = "none" if ok else "unknown"
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
    if kind == "binder":
        # A designed BACKBONE: `expect` residues at ~4 atoms each. The atom floor is 3x rather
        # than the 4x a pure backbone gives, so a run that writes a few sidechains still passes,
        # while a stub or a copied single-residue file does not.
        cifs = sorted(out_dir.rglob("*.cif"))
        if not cifs:
            return False, {"reason": "no binder .cif written",
                           "saw": sorted(q.name for q in out_dir.rglob("*") if q.is_file())[:8]}
        for c in cifs:
            res, atoms = cif_stats(c)
            if res == expect and atoms >= 3 * res:
                return True, {"cif": c.name, "residues": res, "atoms": atoms}
        res, atoms = cif_stats(cifs[0])
        return False, {"reason": f"no .cif is a {expect}-residue binder",
                       "cif": cifs[0].name, "residues": res, "atoms": atoms}
    if kind == "designcif":
        # BoltzGen's design step writes the DESIGNED chain, not the target: the artifact that
        # proves the rung ran is a CIF whose residue count is the binder length. "some file
        # landed in out_dir" would pass on a config dump, which is why it is not the check.
        # BoltzGen writes the COMPLEX, not the binder alone: out_dir/<id>.cif is a copy of the
        # target, and out_dir/intermediate_designs/<id>.cif carries the designed chain (exactly
        # `expect` residues) NEXT TO it. So the check is "some chain is the designed one", not
        # "the file has `expect` residues" -- the latter rejects a run that in fact succeeded,
        # and "a .cif landed in out_dir" accepts the input copy.
        cifs = sorted(out_dir.rglob("*.cif"))
        if not cifs:
            return False, {"reason": "no design .cif written",
                           "saw": sorted(q.name for q in out_dir.rglob("*") if q.is_file())[:8]}
        seen = []
        for c in cifs:
            chains = cif_chains(c)
            res, atoms = cif_stats(c)
            seen.append({"cif": c.name, "chains": chains})
            if expect in chains.values() and len(chains) > 1 and atoms >= 3 * res:
                return True, {"cif": c.name, "chains": chains, "residues": res, "atoms": atoms}
        return False, {"reason": f"no .cif carries a designed chain of {expect} residues "
                                 f"alongside the target", "saw": seen[:4]}
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
    ap.add_argument("--rescore", action="store_true",
                    help="do not run: re-check the artifacts a previous walk left in --work")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="stop the walk at the first FAIL (the ceiling is below it)")
    ap.add_argument("--work", type=pathlib.Path, default=ROOT / "perf" / "bhdesign" / "work")
    a = ap.parse_args()

    a.work.mkdir(parents=True, exist_ok=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for size in [int(s) for s in a.sizes.split(",") if s.strip()]:
        if a.rescore:
            # Re-read an artifact a previous run already left on disk. The device work is the
            # expensive part; a checker defect should cost the re-read and not the card time.
            rec = rescore_rung(a.model, size, a, a.work)
        else:
            rec = run_rung(a.model, size, a, a.work)
        line = json.dumps(rec)
        with a.out.open("a") as fh:
            fh.write(line + "\n")
        print(f"[{rec['ts']}] {a.model} {size}: {rec['verdict']} "
              f"mech={rec['mechanism']} {rec['wall_s']}s {rec['artifact']}", flush=True)
        if rec["verdict"] != "PASS":
            for line in rec.get("throw", []):
                print(f"    ! {line}", flush=True)
            print(rec.get("tail", "")[-600:], flush=True)
            if a.stop_on_fail:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
