#!/usr/bin/env python3
"""p80 -- the accuracy envelope for the gathered atom softmax, per p78_envelope_spec.json.

Three stages, run separately so a device stage can be fanned across cards and a scoring stage
re-run without redesigning anything.

  --stage design   produce one design per seed for one (rung, arm) and write a manifest
  --stage score    refold each design's own binder sequence in isolation with esmfold2, Kabsch
                   the refold onto the designed backbone, record scRMSD and mean refold pLDDT
  --stage decide   read two score manifests and apply the three pre-registered bars

The bars, the sample size, the statistic and the decision rule are NOT in this file. They are in
scripts/rfd3_port/p78_envelope_spec.json, committed before any number existed, and this script
reads them from there. Do not hardcode a margin here: a bar that lives in the harness is a bar
that can be edited after seeing the data.

    # design, per rung and arm, one card each
    env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p3 \
      PYTHONPATH=$PWD python3 -u scripts/rfd3_port/p80_envelope.py --stage design \
        --rung R0 --arm gathered --seeds 1-144 --work perf/p80/R0_gathered
    # score
    ... --stage score --work perf/p80/R0_gathered
    # decide
    ... --stage decide --a perf/p80/R0_dense/scores.json --b perf/p80/R0_gathered/scores.json
"""
import argparse
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())

SPEC = pathlib.Path("scripts/rfd3_port/p78_envelope_spec.json")
CKPT = "/home/ttuser/.boltz/rfd3/weights"
AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def seed_list(s):
    """'1-144' or '1,2,3' or '1-8,1001-1008'."""
    out = []
    for part in s.split(","):
        if "-" in part.strip("-"):
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def rung_fixture(spec, rung):
    for r in spec["size_ladder"]:
        if r["rung"] == rung:
            return pathlib.Path(r["fixture"]), r["designed"]
    raise SystemExit("unknown rung %r; the ladder is %s"
                     % (rung, [r["rung"] for r in spec["size_ladder"]]))


# ---------------------------------------------------------------------------------- design stage
def stage_design(args, spec):
    from tt_bio.rfd3 import design as rfd3_design
    from tt_bio.rfd3 import model as M

    fixture, designed = rung_fixture(spec, args.rung)
    specs = json.loads(fixture.read_text())
    # The arm is the softmax and nothing else. `dense_b` is `dense` on a disjoint seed block --
    # the null calibration -- so it sets the same flag.
    M.set_gathered_softmax(args.arm == "gathered")
    work = pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    print("[p80] design rung=%s arm=%s gathered=%s steps=%d n=%d -> %s"
          % (args.rung, args.arm, M._GATHERED_SOFTMAX, args.steps,
             len(seed_list(args.seeds)), work), flush=True)

    rows = []
    manifest = work / "designs.json"
    done = {r["seed"]: r for r in json.loads(manifest.read_text())["rows"]} if manifest.exists() else {}
    for seed in seed_list(args.seeds):
        if seed in done and pathlib.Path(done[seed]["cif"]).exists():
            rows.append(done[seed])                      # resumable across passes
            continue
        out_dir = work / ("seed_%05d" % seed)
        t0 = time.perf_counter()
        rfd3_design.run_design(specs, str(out_dir), checkpoint_dir=CKPT, from_pdb=True,
                               num_timesteps=args.steps, seed=seed, num_designs=1,
                               batch_size=1, verbose=False)
        cifs = sorted(out_dir.glob("*.cif"))
        row = {"seed": seed, "arm": args.arm, "rung": args.rung,
               "cif": str(cifs[0]) if cifs else None,
               "wall_s": round(time.perf_counter() - t0, 3)}
        rows.append(row)
        print("[p80] seed %5d  %7.2f s  %s" % (seed, row["wall_s"], row["cif"]), flush=True)
        manifest.write_text(json.dumps(
            {"rung": args.rung, "arm": args.arm, "steps": args.steps,
             "designed_residues": designed, "fixture": str(fixture), "rows": rows}, indent=2) + "\n")
    print("[p80] wrote", manifest)


# ----------------------------------------------------------------------------------- score stage
def designed_chain(cif, n_designed):
    """(chain name, {seqid: CA position}, one-letter sequence) for the designed binder.

    Identified by residue count: every rung designs exactly n_designed residues and the target is
    a different length at every rung. If two chains tie, the LAST one is the design -- the contig
    puts the designed segment after the target -- and the tie is printed so it is never silent.
    """
    import gemmi
    st = gemmi.read_structure(str(cif))
    st.remove_alternative_conformations()
    hits = []
    for chain in st[0]:
        res = [r for r in chain if r.find_atom("CA", "*") is not None]
        if len(res) == n_designed:
            hits.append((chain.name, res))
    if not hits:
        return None, None, None
    if len(hits) > 1:
        print("[p80] WARNING %s has %d chains of %d residues, taking the last"
              % (cif, len(hits), n_designed), flush=True)
    name, res = hits[-1]
    ca = {r.seqid.num: r.find_atom("CA", "*").pos for r in res}
    seq = "".join(AA3TO1.get(r.name, "X") for r in res)
    return name, ca, seq


def refold(seqs, work, model, accelerator):
    """One esmfold2 call per design, written to a per-design fasta.

    The refold runs on the shipped dense default in EVERY arm (p78: only the design step
    differs), so this must not inherit RFD3_GATHERED_SOFTMAX from the caller's environment.
    """
    fa_dir = work / "refold_in"
    out_dir = work / "refold_out"
    fa_dir.mkdir(parents=True, exist_ok=True)
    for seed, seq in seqs:
        (fa_dir / ("seed_%05d.fasta" % seed)).write_text(">seed_%05d\n%s\n" % (seed, seq))
    env = dict(os.environ)
    env.pop("RFD3_GATHERED_SOFTMAX", None)
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(fa_dir),
           "--model", model, "--out_dir", str(out_dir), "--output_format", "cif",
           "--accelerator", accelerator, "--single_sequence"]
    print("[p80] refold:", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        print("[p80] refold rc=%d -- inspect %s" % (rc, out_dir), flush=True)
    return out_dir


def stage_score(args, spec):
    import gemmi
    work = pathlib.Path(args.work)
    man = json.loads((work / "designs.json").read_text())
    n_designed = man["designed_residues"]

    seqs, meta = [], {}
    for row in man["rows"]:
        if not row.get("cif"):
            continue
        name, ca, seq = designed_chain(pathlib.Path(row["cif"]), n_designed)
        if seq is None:
            print("[p80] seed %d: no %d-residue chain in %s" % (row["seed"], n_designed, row["cif"]))
            continue
        seqs.append((row["seed"], seq))
        meta[row["seed"]] = {"chain": name, "ca": ca, "seq": seq}

    out_dir = refold(seqs, work, args.refold_model, args.accelerator)

    rows = []
    for seed, seq in seqs:
        hits = sorted(out_dir.rglob("*seed_%05d*.cif" % seed))
        if not hits:
            print("[p80] seed %d: no refold output" % seed, flush=True)
            continue
        st = gemmi.read_structure(str(hits[0]))
        st.remove_alternative_conformations()
        ref = [r.find_atom("CA", "*") for ch in st[0] for r in ch
               if r.find_atom("CA", "*") is not None]
        des = [meta[seed]["ca"][k] for k in sorted(meta[seed]["ca"])]
        n = min(len(ref), len(des))
        if n != len(des):
            print("[p80] seed %d: refold has %d CA against %d designed, aligning the first %d"
                  % (seed, len(ref), len(des), n), flush=True)
        sup = gemmi.superpose_positions([a.pos for a in ref[:n]], des[:n])
        # ESMFold2 writes per-atom pLDDT into the B-factor column.
        plddt = sum(a.b_iso for a in ref[:n]) / n
        rows.append({"seed": seed, "scrmsd": round(sup.rmsd, 4), "plddt": round(plddt, 3),
                     "n_ca": n, "seq": seq, "refold": str(hits[0])})
        print("[p80] seed %5d  scRMSD %7.3f A  pLDDT %6.2f  n_ca %d"
              % (seed, sup.rmsd, plddt, n), flush=True)

    out = work / "scores.json"
    out.write_text(json.dumps({"rung": man["rung"], "arm": man["arm"],
                               "refold_model": args.refold_model,
                               "designed_residues": n_designed, "rows": rows}, indent=2) + "\n")
    print("[p80] wrote %s (%d scored of %d designed)" % (out, len(rows), len(man["rows"])))


# ---------------------------------------------------------------------------------- decide stage
def boot_diff(a, b, stat, resamples, rng):
    """Bootstrap distribution of stat(b) - stat(a). UNPAIRED, per p78: the arms share seeds but a
    one-ULP difference in step 1 diverges the trajectory, so same-seed structures are unrelated."""
    out = []
    for _ in range(resamples):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        out.append(stat(rb) - stat(ra))
    out.sort()
    return out


def stage_decide(args, spec):
    a = json.loads(pathlib.Path(args.a).read_text())
    b = json.loads(pathlib.Path(args.b).read_text())
    st = spec["statistics"]
    # "10 000 resamples" -- take the space-grouped number in front of the word, not every digit
    # in the sentence (a plain digit scrape also swallows the "seed 0" and asks for 100 000).
    m = re.search(r"([\d][\d\s]*\d)\s*resamples", st["decision_statistic"])
    resamples = int(m.group(1).replace(" ", "")) if m else 10000
    rng = random.Random(0)

    def col(d, k):
        return [r[k] for r in d["rows"]]

    def median(v):
        v = sorted(v)
        n = len(v)
        return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])

    strict = 2.0        # p78 scRMSD_protocol: the field's strict designability threshold
    rate = lambda v: sum(1 for x in v if x <= strict) / len(v)
    mean = lambda v: sum(v) / len(v)

    print("arm A = %s %s (n=%d)   arm B = %s %s (n=%d)   %d resamples, unpaired"
          % (a["rung"], a["arm"], len(a["rows"]), b["rung"], b["arm"], len(b["rows"]), resamples))
    if len(a["rows"]) < spec["n"]["per_arm_per_rung"] or \
       len(b["rows"]) < spec["n"]["per_arm_per_rung"]:
        print("INTERIM -- below the pre-registered n of %d per arm. p78 forbids reading a verdict "
              "here; more designs, never a wider margin." % spec["n"]["per_arm_per_rung"])

    verdicts = []
    for key, field, stat, side in (
        ("primary", "scrmsd", rate, "lower"),
        ("secondary", "scrmsd", median, "upper"),
        ("tertiary", "plddt", mean, "lower"),
    ):
        m = spec["metric"][key]
        va, vb = col(a, field), col(b, field)
        obs = stat(vb) - stat(va)
        dist = boot_diff(va, vb, stat, resamples, rng)
        lo, hi = dist[int(0.05 * resamples)], dist[int(0.95 * resamples) - 1]
        bound = lo if side == "lower" else hi
        print("\n%-10s %-24s  A %.4f   B %.4f   diff %+.4f   one-sided 95%% %s bound %+.4f"
              % (key, m["name"], stat(va), stat(vb), obs, side, bound))
        print("           margin: %s" % m["margin"])
        verdicts.append((key, m["name"], stat(va), stat(vb), obs, bound, side))

    print("\nThe three margins are in p78_envelope_spec.json and are applied by reading the bound "
          "above against the margin printed beside it. This harness does not restate them as "
          "numbers, on purpose: a bar that lives in the harness is a bar that can be edited after "
          "seeing the data.")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({
            "a": {"rung": a["rung"], "arm": a["arm"], "n": len(a["rows"])},
            "b": {"rung": b["rung"], "arm": b["arm"], "n": len(b["rows"])},
            "resamples": resamples, "paired": False,
            "n_required": spec["n"]["per_arm_per_rung"],
            "interim": min(len(a["rows"]), len(b["rows"])) < spec["n"]["per_arm_per_rung"],
            "metrics": [{"key": k, "name": nm, "a": av, "b": bv, "diff": d,
                         "bound": bd, "side": sd}
                        for k, nm, av, bv, d, bd, sd in verdicts],
        }, indent=2) + "\n")
        print("wrote", args.out)



# ----------------------------------------------------------------------------------- power stage
def stage_power(args, spec):
    """How often does the NULL clear the pre-registered bars at a given n?

    p78 derives n = 144 from the primary margin: at the worst-case rate p = 0.5 the one-sided
    95 % bound of a difference of two independent proportions is 1.645 * sqrt(2 p (1-p) / n),
    which sits inside 10 points from n >= 135. That arithmetic assumes the OBSERVED difference
    lands at zero. It does not: the point estimate itself moves by about one SE, so the bound is
    (observed - half-width) and both terms carry noise. This stage measures the null pass rate by
    simulation instead of assuming it, which is the same question the spec's own null calibration
    asks of the card -- and answering it on the host first is free.

    Nothing here changes a bar. If the null pass rate is poor the remedy p78 already fixes is more
    designs, never a wider margin.
    """
    rng = random.Random(args.power_seed)
    strict = 2.0
    rate = lambda v: sum(1 for x in v if x <= strict) / len(v)
    # A scRMSD population with the pre-registered worst case for the primary metric: rate 0.5.
    # Heavy-tailed on purpose, since that is what the spec says these distributions look like.
    def draw(n):
        out = []
        for _ in range(n):
            out.append(rng.gauss(1.6, 0.5) if rng.random() < 0.5 else rng.gauss(6.0, 4.0))
        return [max(0.1, x) for x in out]

    print("[p80] null power at margin -%.2f on the primary rate, %d trials, %d bootstrap "
          "resamples each" % (args.margin, args.trials, args.resamples))
    print("%6s %10s %10s %12s" % ("n", "pass rate", "mean bound", "mean |diff|"))
    for n in args.power_n:
        passes, bounds, diffs = 0, [], []
        for _ in range(args.trials):
            a, b = draw(n), draw(n)
            obs = rate(b) - rate(a)
            dist = boot_diff(a, b, rate, args.resamples, rng)
            lo = dist[int(0.05 * args.resamples)]
            bounds.append(lo)
            diffs.append(abs(obs))
            passes += lo >= -args.margin
        print("%6d %9.1f%% %10.4f %12.4f"
              % (n, 100.0 * passes / args.trials, sum(bounds) / len(bounds),
                 sum(diffs) / len(diffs)))
    print("\nA null that fails is the metric failing to resolve the margin at that n, not the "
          "lever failing. p78's rule is that n rises until the null passes.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("design", "score", "decide", "power"))
    ap.add_argument("--rung", default="R4")
    ap.add_argument("--arm", default="dense", choices=("dense", "gathered", "dense_b"))
    ap.add_argument("--seeds", default="1-144")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--work", default=None)
    ap.add_argument("--refold-model", default="esmfold2")
    ap.add_argument("--accelerator", default="tenstorrent")
    ap.add_argument("--a", default=None, help="decide: arm A scores.json")
    ap.add_argument("--b", default=None, help="decide: arm B scores.json")
    ap.add_argument("--out", default=None, help="decide: verdict JSON")
    ap.add_argument("--power-n", type=int, nargs="+", default=[144, 288, 432, 576],
                    help="power: sample sizes to simulate")
    ap.add_argument("--margin", type=float, default=0.10, help="power: the primary margin")
    ap.add_argument("--trials", type=int, default=200, help="power: simulated experiments")
    ap.add_argument("--resamples", type=int, default=2000, help="power: bootstrap draws per trial")
    ap.add_argument("--power-seed", type=int, default=0)
    args = ap.parse_args()
    spec = json.loads(SPEC.read_text())
    if args.stage == "power":
        stage_power(args, spec)
    elif args.stage == "design":
        stage_design(args, spec)
    elif args.stage == "score":
        stage_score(args, spec)
    else:
        stage_decide(args, spec)


if __name__ == "__main__":
    main()
