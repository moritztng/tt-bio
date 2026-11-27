"""Grade device Muon Newton-Schulz against B1' (state/r4t/DECISIONS.md D31, POST-DATA amendment of B1).

(a) every matrix |e_dev - e_emu| / e_emu <= 0.05, e_emu = CPU emulation of the device op order (ns_device.py `gram`
    arm: gram rounded to bf16 once, p fp32 rounded once, a x + p x fp32 rounded once) vs float64;
(b) per shape over >= 200 random matrices, median(e_emu / e_host) <= 1.10 (e_host = upstream's bf16 host order);
(c) e_dev <= 1.5 e_host for every matrix with min dim >= 64;
(d) the x1.05 broken device arm must FAIL.
CPU only. Reads the device per-matrix errors from ns_device.py's `per` field; regenerates the same seed-0 matrices.
"""
import argparse, json, sys, time
import torch

A, B, C, EPS = 3.4445, -4.775, 2.0315, 1e-7
bf = lambda t: t.to(torch.bfloat16).double()


def ns_ref(x, dt):
    x = x.to(dt); x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=EPS)
    for _ in range(5):
        g = torch.bmm(x, x.transpose(-1, -2)); x = torch.baddbmm(x, torch.baddbmm(g, g, g, beta=B, alpha=C), x, beta=A)
    return x.double()


def ns_emu(x):
    x = bf(x.double()); x = bf(x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=EPS))
    for _ in range(5):
        g = bf(x @ x.transpose(-1, -2)); p = bf(g * B + (g @ g) * C); x = bf(x * A + p @ x)
    return x


def errs(w):
    r = ns_ref(w, torch.float64); n = r.flatten(1).norm(dim=1)
    e = lambda o: ((o - r).flatten(1).norm(dim=1) / n)
    return e(ns_emu(w)), e(ns_ref(w, torch.bfloat16))


def wide(w):
    return w.transpose(-1, -2).contiguous() if w.shape[-2] > w.shape[-1] else w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--broken", required=True)
    ap.add_argument("--nrand", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    shapes = json.loads(open(a.shapes).read().splitlines()[2])
    dev, brk = json.load(open(a.dev)), json.load(open(a.broken))

    def grade_ac(d):
        per = {(tuple(s), k): (ed, eh) for s, k, ed, eh in d["per"]}
        gen = torch.Generator().manual_seed(0)
        fa, fc, worst_a = [], [], 0.0
        for (r, c), cnt in shapes:
            w = wide(torch.randn(cnt, r, c, generator=gen))
            ee, _ = errs(w)
            for k in range(cnt):
                ed, eh = per[(tuple(w.shape[1:]), k)]
                da = abs(ed - float(ee[k])) / float(ee[k]); worst_a = max(worst_a, da)
                if da > 0.05:
                    fa.append([[r, c], k, ed, float(ee[k])])
                if min(r, c) >= 64 and ed > 1.5 * eh:
                    fc.append([[r, c], k, ed, eh])
        return fa, fc, worst_a

    t0 = time.time()
    fa, fc, worst_a = grade_ac(dev)
    bfa, bfc, _ = grade_ac(brk)
    print("a/c done", round(time.time() - t0), "s", flush=True)
    gen = torch.Generator().manual_seed(1)
    b = []
    for (r, c), _cnt in shapes:
        rat = []
        for i in range(0, a.nrand, 8):  # chunks bound memory on the 6144x768 shape
            ee, eh = errs(wide(torch.randn(min(8, a.nrand - i), r, c, generator=gen)))
            rat.append(ee / eh)
        rat = torch.cat(rat)
        b.append(dict(shape=[r, c], n=len(rat), median=round(float(rat.median()), 4), max=round(float(rat.max()), 3)))
        print(b[-1], round(time.time() - t0), "s", flush=True)
    fb = [x for x in b if x["median"] > 1.10]
    res = dict(bar="B1' (D31, post-data)", dev_sha=dev["sha"], dev_digest=dev["digest"], step_s_min=dev["step_s_min"],
               aiclk=dev["aiclk"], a_fail=fa, a_worst=round(worst_a, 4), b=b, b_fail=fb, c_fail=fc,
               d_broken_fails=bool(bfa or bfc), d_broken_flags=bfa + bfc,
               verdict="PASS" if not (fa or fb or fc) and (bfa or bfc) else "FAIL")
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("verdict", "a_worst", "a_fail", "b_fail", "c_fail", "d_broken_fails")}))


if __name__ == "__main__":
    main()
