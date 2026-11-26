"""Grade device Muon Newton-Schulz against B1'' (state/r4t/DECISIONS.md D33, orchestrator ruling, POST-DATA).

(a'') over all matrices, median(e_dev / e_emu) <= 1.05 and every matrix e_dev / e_emu <= 1.15;
(b)   unchanged from B1' (CPU only: median(e_emu / e_host) <= 1.10 per shape over 200 random matrices), read from
      the B1' grade (b1prime.py, same emulation, same seed-1 draws) instead of re-running 8 minutes of CPU;
(c)   unchanged: e_dev <= 1.5 e_host for every matrix with min dim >= 64 (e_host = upstream bf16 order vs float64);
(d)   BOTH broken device arms must FAIL B1'' whole: x1.05 on one matrix, and one Newton-Schulz iteration dropped.
"""
import argparse, json, statistics, sys
import torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
from b1prime import errs, wide


def grade(d, shapes):
    per = {(tuple(s), k): (ed, eh) for s, k, ed, eh in d["per"]}
    gen = torch.Generator().manual_seed(0)
    rat, fa, fc = [], [], []
    for (r, c), cnt in shapes:
        w = wide(torch.randn(cnt, r, c, generator=gen))
        ee, _ = errs(w)
        for k in range(cnt):
            ed, eh = per[(tuple(w.shape[1:]), k)]
            q = ed / float(ee[k]); rat.append(q)
            if q > 1.15:
                fa.append([[r, c], k, round(q, 4)])
            if min(r, c) >= 64 and ed > 1.5 * eh:
                fc.append([[r, c], k, ed, eh])
    med = statistics.median(rat)
    return dict(n=len(rat), median=round(med, 4), max=round(max(rat), 4), a_fail_median=med > 1.05,
                a_fail_matrices=fa[:20], n_a_fail=len(fa), c_fail=fc[:20], n_c_fail=len(fc),
                fails=bool(med > 1.05 or fa or fc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True); ap.add_argument("--dev", required=True)
    ap.add_argument("--broken", required=True); ap.add_argument("--broken-iter", required=True)
    ap.add_argument("--b1prime", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    shapes = json.loads(open(a.shapes).read().splitlines()[2])
    dev = json.load(open(a.dev))
    g = grade(dev, shapes)
    bk = grade(json.load(open(a.broken)), shapes)
    bi_d = json.load(open(a.broken_iter)); bi = grade(bi_d, shapes)
    assert bi_d.get("iters") == 4, "algorithmic broken arm must be the 4-iteration run"
    b = json.load(open(a.b1prime))["b"]; fb = [x for x in b if x["median"] > 1.10]
    ok = not g["fails"] and not fb and bk["fails"] and bi["fails"]
    res = dict(bar="B1'' (D33, post-data)", dev_sha=dev["sha"], dev_digest=dev["digest"], aiclk=dev["aiclk"],
               step_s_min=dev["step_s_min"], dev=g, b=b, b_fail=fb, broken_x105=bk, broken_iter4=bi,
               broken_iter4_sha=bi_d["sha"], broken_iter4_aiclk=bi_d["aiclk"], verdict="PASS" if ok else "FAIL")
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(dict(verdict=res["verdict"], dev={k: g[k] for k in ("median", "max", "n_a_fail", "n_c_fail")},
                          b_fail=len(fb), x105_fails=bk["fails"], x105=(bk["median"], bk["max"]),
                          iter4_fails=bi["fails"], iter4=(bi["median"], bi["max"], bi["n_c_fail"]))))


if __name__ == "__main__":
    main()
