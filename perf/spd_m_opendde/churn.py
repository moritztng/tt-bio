"""Shape churn: what a long-lived OpenDDE worker pays the first time it folds a new shape.

Served 474-aa jobs on one Galaxy read ~75, ~84 and ~92-100 s for the same residue count, in steps
of ~9 s. Same-length binders differ in two shapes the token bucket does not pin: the refiner's
structural-token axis (Ns = 2 * n_res - n_gly, bucketed to 32) and the atom axis. This loads the
model once and folds, in one process, the fixture and three same-length variants twice each:

  same   L -> I everywhere: same atom count, same Ns. New sequence, no new shape (control).
  atom   40 L -> K: +40 atoms, same Ns.
  ns     30 A -> G: Ns -30 (one refiner bucket down), -30 atoms.

first - second per variant is the cost of meeting that shape. Every fold is the full predict_one
(featurize + fold + CIF write), the worker's own boundary.
"""
import argparse, hashlib, json, re, sys, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tree = a.tree.resolve()
    sys.path.insert(0, str(tree))
    sys.path.insert(0, str(tree / "scripts" / "gpu_vs_tt"))
    import tt_baseline as B
    from tt_bio.main import _resolve_sampling_steps
    B.SAMPLING_STEPS = _resolve_sampling_steps(None, "opendde")

    fix = tree / "perf" / "size512" / "fixtures"
    yml, a3m = (fix / f"cdk2x2_{a.size}.yaml").read_text(), (fix / f"cdk2x2_{a.size}.a3m").read_text()
    seq = re.search(r"sequence: (\w+)", yml).group(1)

    def swap(s, src, dst, n=None):
        out, k = list(s), 0
        for i, c in enumerate(out):
            if c == src and (n is None or k < n):
                out[i], k = dst, k + 1
        assert n is None or k == n, (src, k)
        return "".join(out)

    variants = {"base": seq, "same": swap(seq, "L", "I"), "atom": swap(seq, "L", "K", 40),
                "ns": swap(seq, "A", "G", 30)}
    inp = a.out.parent / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, s in variants.items():
        p, q = inp / f"churn_{name}.yaml", inp / f"churn_{name}.a3m"
        p.write_text(yml.replace(seq, s))
        rows = a3m.split("\n")
        rows[1] = s
        q.write_text("\n".join(rows))
        paths[name] = (p, q)

    msa_dir = a.out.parent / "msa"
    one_fold, meta, state = B.build_fold("opendde", msa_dir, *paths["base"], fast=a.fast)
    for p, q in paths.values():
        B.seed_msa_cache(p, q, msa_dir)
    job_cfg = meta["job_cfg"]

    res = {"tree": str(tree), "size": a.size, "fast": a.fast, "folds": [],
           "variants": {n: {"n_gly": s.count("G"), "ns": 2 * len(s) - s.count("G"),
                            "ns_bucket": -(-(2 * len(s) - s.count("G")) // 32) * 32}
                        for n, s in variants.items()}}
    order = ["base", "base", "base", "same", "same", "atom", "atom", "ns", "ns", "base"]
    for i, name in enumerate(order):
        t0 = time.perf_counter()
        m, _best, _f = state.predict_one(paths[name][0], job_cfg)
        dt = time.perf_counter() - t0
        cif = sorted(Path(job_cfg["struct_dir"]).glob(f"churn_{name}*.cif"))
        res["folds"].append({"i": i, "variant": name, "fold_s": round(dt, 3), "n_atoms": m.get("n_atoms"),
                             "n_tokens": m.get("n_tokens"), "plddt": m.get("plddt") or m.get("complex_plddt"),
                             "cif": hashlib.sha256(cif[0].read_bytes()).hexdigest()[:16] if cif else None,
                             "loadavg": open("/proc/loadavg").read().split()[:3]})
        print(f"[churn] {i} {name} {dt:.2f}s atoms={m.get('n_atoms')}", flush=True)
        a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
