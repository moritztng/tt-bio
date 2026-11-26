"""Grade the D34 hybrid (device Newton-Schulz for min dim >= 64, upstream host NS for the rest) on upstream's REAL
NS inputs: 3 independent fresh-step snapshots (dump_momentum.py), bar B1'' unchanged and stricter (D34):
every device matrix passes (a'') in all 3 snapshots (median e_dev/e_emu <= 1.05 and each <= 1.15), (b) passes
(B1' CPU grade, unchanged), (c) e_dev <= 1.5 e_host for every device matrix, and both broken arms (x1.05 on one
matrix, 4 NS iterations) FAIL. Host-side matrices run upstream's own code, so they are identical by construction.
usage: b1h_grade.py --snaps s0.pt s1.pt s2.pt --devs d0.json d1.json d2.json --broken b.json --broken-iter i.json
       --b1prime b1prime.json --out out.json      (broken arms ran on snapshot 0)
"""
import argparse, json, statistics, sys, os
import torch
sys.path.insert(0, os.path.dirname(__file__))
from b1prime import errs, wide


def grade(snap, d, min_dim):
    groups = torch.load(snap)["groups"]
    per = {(gi, k): (ed, eh) for s, k, ed, eh, gi in d["per"]}
    rat, fa, fc, n_host, seen = [], [], [], 0, 0
    for gi, g in enumerate(groups):
        if min(g.shape[1:]) < min_dim:
            n_host += g.shape[0]; continue
        w = wide(g.float())
        for i in range(0, w.shape[0], 8):
            ee, _ = errs(w[i:i + 8])
            for j in range(ee.shape[0]):
                k = i + j; ed, eh = per[(gi, k)]; seen += 1
                q = ed / float(ee[j]); rat.append(q)
                if q > 1.15: fa.append([list(w.shape[1:]), gi, k, round(q, 4)])
                if ed > 1.5 * eh: fc.append([list(w.shape[1:]), gi, k, ed, eh])
    assert seen == len(per), (seen, len(per))
    med = statistics.median(rat)
    return dict(n_device=len(rat), n_host=n_host, median=round(med, 4), max=round(max(rat), 4), n_a_fail=len(fa),
                a_fail=fa[:20], n_c_fail=len(fc), c_fail=fc[:20], fails=bool(med > 1.05 or fa or fc),
                sha=d["sha"], digest=d["digest"], aiclk=d["aiclk"], step_s_min=d["step_s_min"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snaps", nargs=3, required=True); ap.add_argument("--devs", nargs=3, required=True)
    ap.add_argument("--broken", required=True); ap.add_argument("--broken-iter", required=True)
    ap.add_argument("--b1prime", required=True); ap.add_argument("--min-dim", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    devs = [json.load(open(p)) for p in a.devs]
    assert all(d["iters"] == 5 and not d["broken"] and d["min_dim"] == a.min_dim for d in devs)
    snaps = [grade(s, d, a.min_dim) for s, d in zip(a.snaps, devs)]
    bk_d, bi_d = json.load(open(a.broken)), json.load(open(a.broken_iter))
    assert bk_d["broken"] and bi_d["iters"] == 4
    bk, bi = grade(a.snaps[0], bk_d, a.min_dim), grade(a.snaps[0], bi_d, a.min_dim)
    b = json.load(open(a.b1prime))["b"]; fb = [x for x in b if x["median"] > 1.10]
    ok = not any(s["fails"] for s in snaps) and not fb and bk["fails"] and bi["fails"]
    res = dict(bar="B1'' on real momentum, hybrid min dim >= 64 on device (D34)", snapshots=snaps, b=b, b_fail=fb,
               broken_x105=bk, broken_iter4=bi, verdict="PASS" if ok else "FAIL")
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(dict(verdict=res["verdict"], snaps=[(s["median"], s["max"], s["n_a_fail"], s["n_c_fail"], s["n_device"], s["n_host"]) for s in snaps],
                          b_fail=len(fb), x105=(bk["fails"], bk["max"]), iter4=(bi["fails"], bi["median"]))))


if __name__ == "__main__":
    main()
