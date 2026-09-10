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
    """Condition on `target_res` residues of `target`, spilling into later chains as needed.

    Cropping chain A alone caps the axis at chain A's length, and pxdesign does not complain:
    a `crop: ["1-1831"]` against a 1008-residue chain A conditions on 1008 and writes a
    perfectly good 80-residue binder, so the rung reads as a PASS at a size that never ran.
    `big_1831.cif` is exactly that shape (A=1008, B=823), which is how the plateau showed up.
    """
    chains = cif_chains(target)
    total = sum(chains.values())
    if target_res > total:
        raise SystemExit(f"pxdesign_fixture: {target} carries {total} residues, "
                         f"cannot condition on {target_res}")
    take, left = {}, target_res
    for cid in sorted(chains):
        if left <= 0:
            break
        take[cid] = min(chains[cid], left)
        left -= take[cid]
    hot_chain = next(iter(take))
    h = take[hot_chain] // 2
    lines = ["target:", f'  file: "{target}"', "  chains:"]
    for cid, n in take.items():
        lines += [f"    {cid}:", f'      crop: ["1-{n}"]']
        if cid == hot_chain:
            lines.append(f"      hotspots: [{h}, {h + 1}, {h + 2}]")
    lines.append(f"binder_length: {binder}")
    p = work / f"px{target_res}.yaml"
    p.write_text("\n".join(lines) + "\n")
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


_BINDER_CHAIN = "Z"   # never collides with the crop, whose chains are labelled A..H


def boltzgen_fixture(work: pathlib.Path, target_res: int, target: pathlib.Path,
                     binder: int = 80) -> tuple[pathlib.Path, int, int]:
    """A BoltzGen design spec against the first `target_res` residues of `target`.

    Returns the YAML and the target's ATOM count -- the axis this model is sized on. BoltzGen
    takes the `entities:` grammar (a designed chain given as a length, plus a structure file to
    include a chain from), NOT the `sequences:` schema `tt-bio predict` takes; the two are
    different parsers and the predict spelling is silently a different model's input.
    """
    cif = work / f"bgt{target_res}.cif"
    res, atoms = crop_cif(target, target_res, cif)
    # Include EVERY chain the crop produced. Naming one chain caps the axis at that chain's
    # length and BoltzGen does not complain: a 3662-residue crop whose chain A is 1008 conditions
    # on 1008 and still designs a perfectly good 80-residue binder, so the rung reads PASS at a
    # size that never ran. Same cause as the pxdesign fixture, same fix.
    chains = cif_chains(cif)
    include = "".join(f"        - chain:\n            id: {cid}\n" for cid in sorted(chains))
    p = work / f"bg{target_res}.yaml"
    p.write_text(
        "entities:\n"
        "  - protein:\n"
        f"      id: {_BINDER_CHAIN}\n"
        f"      sequence: {binder}\n"
        "  - file:\n"
        f"      path: {cif.name}\n"
        "      include:\n"
        + include)
    return p, atoms, res

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


# A ttnn failure ends in ~200 lines of C++ backtrace and, on interpreter teardown, a few
# hundred nanobind "leaked instance" lines. A fixed-size tail of the blob is therefore all
# noise and none of the cause: the throw that says WHY sits thousands of characters above it.
# So a failed rung keeps the lines that carry a diagnosis, not the lines that came last.
_DIAG = re.compile(
    r"TT_(?:THROW|FATAL|ASSERT)|Not enough space|RuntimeError|MemoryError|Error:|error:|"
    r"Exception|Traceback|^\s+File \"|raise |device contention|out of memory|Killed|"
    r"Timed out|is in use by", re.M)


def diagnosis(text: str, keep: int = 25) -> list[str]:
    """The lines that say why, in order, deduplicated."""
    seen, out = set(), []
    for line in text.splitlines():
        if len(line) > 400 or line.lstrip().startswith("---") or "leaked " in line:
            continue          # a backtrace frame or nanobind teardown noise, not a diagnosis
        if _DIAG.search(line) and line.strip() not in seen:
            seen.add(line.strip())
            out.append(line.rstrip()[:300])
    return out[:keep]


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
    # The grant can be wider than the card this rung computes on: fanning a backlog onto an
    # idle sibling chip means holding a lease on both, while TT_VISIBLE_DEVICES still pins one.
    env["TT_BIO_LEASE_CARDS"] = os.environ.get("BH_LEASE_CARDS", str(args.card))
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
        checker = ("binder", (args.binder, size))
    elif model == "boltzgen":
        fx, atoms, tres = boltzgen_fixture(work, size, pathlib.Path(args.target), args.binder)
        # `--devices` is an ID LIST, not a count: `--devices 1` means physical card 1, and it
        # only looked like a count on qb1 because the grant there happened to BE card 1. On any
        # other card it dies before the model loads with "Requested Tenstorrent device id(s) [1]
        # not available". Left unset it uses every visible card, and TT_VISIBLE_DEVICES has
        # already narrowed that to the one this rung is pinned to.
        cmd = base + ["design", str(fx), "--model", "boltzgen", "--out_dir", str(out_dir),
                      "--num_designs", "1", "--steps", "design", "--debug"]
        checker = ("designcif", (args.binder, tres))
        extra = {"target_atoms": atoms}
    else:
        raise SystemExit(f"no fixture for {model}")

    t0 = time.time()
    # The child streams to a file, not a pipe. A design rung at this size runs for tens of
    # minutes, and an absorbed throw is indistinguishable from slow progress unless the log can
    # be read WHILE it happens -- boltzgen 3662 threw an L1 assert 143 s in and then held the
    # card for the remaining 1057 s of its budget, and the pipe gave that up only after the
    # kill. A file also keeps stderr, which the TimeoutExpired branch used to drop entirely.
    log = work / f"log_{model}_{size}.txt"
    with log.open("w") as fh:
        try:
            rc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh,
                                stderr=subprocess.STDOUT, timeout=args.timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
            fh.write("\nTIMEOUT\n")
    blob = log.read_text(errors="replace")
    wall = round(time.time() - t0, 1)

    rec = {"model": model, "size": size, "rc": rc, "wall_s": wall,
           "cmd": " ".join(cmd[3:]), "arch": "blackhole", "board": args.board,
           "host": os.uname().nodename, "card": args.card,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}

    ok, detail = check_artifact(checker, out_dir, model)
    rec["artifact"] = detail
    if ok and rc == 0:
        rec["verdict"] = "PASS"
        rec["mechanism"] = "none"
    else:
        rec["verdict"] = "FAIL"
        rec["mechanism"] = classify(blob)
        rec.update(dram_numbers(blob))
        rec["diag"] = diagnosis(blob)
        rec["tail"] = blob[-2500:]
    return rec


def rescore_rung(model: str, size: int, args, work: pathlib.Path) -> dict:
    """Re-verify a rung from the artifacts on disk. No subprocess, no device."""
    out_dir = work / f"out_{model}_{size}"
    checker = {"rfd3": ("cif", size), "pxdesign": ("binder", (args.binder, size)),
               "boltzgen": ("designcif", (args.binder, size))}.get(model, ("npz", size))
    rec = {"model": model, "size": size, "rc": None, "wall_s": 0.0,
           "cmd": "(rescored from artifacts on disk)", "arch": "blackhole", "board": args.board,
           "host": os.uname().nodename, "card": args.card,
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
        # A designed BACKBONE: `nres` residues at ~4 atoms each. The atom floor is 3x rather
        # than the 4x a pure backbone gives, so a run that writes a few sidechains still passes,
        # while a stub or a copied single-residue file does not.
        #
        # The binder alone is NOT enough, because its size does not depend on the target: a run
        # that conditioned on 1008 residues when the rung asked for 1831 writes exactly the same
        # 80-residue backbone. So the rung also reads `conditioned_tokens` out of designs.json
        # and requires it to be the size that was asked for. That is what makes this ladder a
        # measurement of the target axis rather than of the binder length.
        nres, want_target = expect
        cifs = sorted(out_dir.rglob("*.cif"))
        if not cifs:
            return False, {"reason": "no binder .cif written",
                           "saw": sorted(q.name for q in out_dir.rglob("*") if q.is_file())[:8]}
        designs = out_dir / "designs.json"
        cond = None
        if designs.is_file():
            recs = json.loads(designs.read_text())
            if recs:
                cond = recs[0].get("conditioned_tokens")
        for c in cifs:
            res, atoms = cif_stats(c)
            if res == nres and atoms >= 3 * res:
                d = {"cif": c.name, "residues": res, "atoms": atoms,
                     "conditioned_tokens": cond, "asked_target": want_target}
                if cond != want_target:
                    d["reason"] = (f"binder is right but the target was not: conditioned on "
                                   f"{cond} tokens, rung asked for {want_target}")
                    return False, d
                return True, d
        res, atoms = cif_stats(cifs[0])
        return False, {"reason": f"no .cif is a {nres}-residue binder",
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
        # The designed chain alone is NOT enough, for the same reason it is not enough for
        # pxdesign: the binder length does not depend on the target, so a run that conditioned
        # on 1008 residues when the rung asked for 3662 writes the identical 80-residue chain.
        # The rung therefore also sums the chains that are NOT the binder and requires that to
        # be the size that was asked for. That is what makes this a measurement of the target
        # axis rather than of the binder length.
        nres, want_target = expect
        cifs = sorted(out_dir.rglob("*.cif"))
        if not cifs:
            return False, {"reason": "no design .cif written",
                           "saw": sorted(q.name for q in out_dir.rglob("*") if q.is_file())[:8]}
        seen = []
        for c in cifs:
            chains = cif_chains(c)
            res, atoms = cif_stats(c)
            seen.append({"cif": c.name, "chains": chains})
            if nres not in chains.values() or len(chains) < 2 or atoms < 3 * res:
                continue
            # Exactly one chain is the binder; everything else is target.
            rest, dropped = dict(chains), False
            for cid, n in sorted(chains.items()):
                if n == nres and not dropped:
                    del rest[cid]
                    dropped = True
            got_target = sum(rest.values())
            d = {"cif": c.name, "chains": chains, "residues": res, "atoms": atoms,
                 "target_residues": got_target, "asked_target": want_target}
            if got_target != want_target:
                d["reason"] = (f"the binder is right but the target was not: conditioned on "
                               f"{got_target} residues, rung asked for {want_target}")
                return False, d
            return True, d
        return False, {"reason": f"no .cif carries a designed chain of {nres} residues "
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
    ap.add_argument("--board", default=os.environ.get("BH_BOARD", "p150a"),
                    help="which Blackhole board this walk ran on; a row without it cannot be "
                         "compared across machines")
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
            print(rec.get("tail", "")[-1200:], flush=True)
            if a.stop_on_fail:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
