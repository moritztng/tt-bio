import sys, json, statistics, torch
sys.path.insert(0, "/home/ttuser/.coworker/wt/r4t-perf/perf/r4t_perf")
import b1prime as G
A, B, C, EPS = G.A, G.B, G.C, G.EPS
def bft(t):  # fp32 -> bf16 by truncation (round toward zero)
    i = t.float().contiguous().view(torch.int32) & ~0xFFFF
    return i.view(torch.float32).double()
def emu(x, rnd, rnd_in=None):
    rnd_in = rnd_in or rnd
    x = G.bf(x.double()); x = rnd_in(x / x.norm(dim=(-2, -1), keepdim=True))
    for _ in range(5):
        g = rnd((x @ x.transpose(-1, -2)).float()); p = rnd((g * B + (g @ g) * C).float()); x = rnd((x * A + p @ x).float())
    return x
shapes = json.loads(open("/home/ttuser/.coworker/wt/r4t-perf/perf/r4t_perf/rfd4_denoiser_2d_shapes.txt").read().splitlines()[2])
dev = json.load(open("/home/ttuser/r4t/perf/b1p/dev.json"))
per = {(tuple(s), k): ed for s, k, ed, eh in dev["per"]}
gen = torch.Generator().manual_seed(0)
for (r, c), cnt in shapes[:3]:
    w = G.wide(torch.randn(cnt, r, c, generator=gen))
    ref = G.ns_ref(w, torch.float64); n = ref.flatten(1).norm(dim=1)
    e = lambda o: ((o - ref).flatten(1).norm(dim=1) / n)
    er, et = e(emu(w, G.bf)), e(emu(w, bft, G.bf))
    ed = torch.tensor([per[(tuple(w.shape[1:]), k)] for k in range(cnt)], dtype=torch.float64)
    print([r, c], "dev/rne med %.4f" % (ed / er).median(), "dev/trunc med %.4f min %.3f max %.3f" % ((ed / et).median(), (ed / et).min(), (ed / et).max()),
          "trunc/rne %.4f" % (et / er).median())
print("--- site variants")
def tf32(t):  # keep 10 explicit mantissa bits, truncated
    i = t.float().contiguous().view(torch.int32) & ~0x1FFF
    return i.view(torch.float32).double()
def emu2(x, site):
    T = lambda s, v: (bft(v) if site == s else G.bf(v))
    x = G.bf(x.double()); x = G.bf(x / x.norm(dim=(-2, -1), keepdim=True))
    for _ in range(5):
        g = T("g", (x @ x.transpose(-1, -2)).float())
        gg = (g @ g); gg = tf32(gg) if site == "tf32" else gg
        p = T("p", (g * B + gg * C).float())
        px = p @ x; px = tf32(px) if site == "tf32" else px
        x = T("x", (x * A + px).float())
    return x
gen = torch.Generator().manual_seed(0)
(r, c), cnt = shapes[0]
w = G.wide(torch.randn(cnt, r, c, generator=gen))
ref = G.ns_ref(w, torch.float64); n = ref.flatten(1).norm(dim=1)
e = lambda o: ((o - ref).flatten(1).norm(dim=1) / n)
er = e(emu(w, G.bf)); ed = torch.tensor([per[(tuple(w.shape[1:]), k)] for k in range(cnt)], dtype=torch.float64)
for site in ("g", "p", "x", "tf32"):
    es = e(emu2(w, site)); q = ed / es
    print(site, "variant/rne %.4f" % (es / er).median(), "dev/variant med %.4f min %.3f max %.3f" % (q.median(), q.min(), q.max()))
