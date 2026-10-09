"""spd-msa: OPM's full-depth a/b layout permutes (`contiguous_ab`) as tile transposes.

    TT_VISIBLE_DEVICES=<chip> python perf/spd_msa/opm_abperm.py OUT [DEPTH=9984] [TOKENS=736] [REPS=5]

a [S, I, C] -> permute (1, 2, 0) -> [I, C, S]   vs  HC transpose then WH transpose (as PWA's head output).
b [S, J, D] -> permute (2, 1, 0) -> [D, J, S]   vs  WH, HC, WH transposes.
Pure data movement: each route must be torch.equal to the permute. AICLK is read before and after each arm.
"""
import json, statistics, sys, time
from pathlib import Path

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
S_ = int(sys.argv[2]) if len(sys.argv) > 2 else 9984
T_ = int(sys.argv[3]) if len(sys.argv) > 3 else 736
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 5
LOG = open(OUT / "abperm.jsonl", "a")


def log(**kw):
    kw["t_unix"] = time.time(); line = json.dumps(kw, default=str)
    LOG.write(line + "\n"); LOG.flush(); print(line, flush=True)


def aiclk():
    out = {}
    for p in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            v = int((p / "tt_aiclk").read_text().split()[0])
            if v < 5000:
                out[p.name.split("!")[1]] = v
        except Exception:
            pass
    return out


from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
import tt_bio.tenstorrent as T

torch.manual_seed(0)
dev = T.get_device()
C = 32
log(ev="start", depth=S_, tokens=T_, reps=REPS, arch=str(dev.arch()), aiclk=aiclk())
x = ttnn.from_torch(torch.randn(S_, T_, C).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                    dtype=ttnn.bfloat16)


def a_perm():
    return ttnn.permute(x, (1, 2, 0))


def a_tchain():
    t = ttnn.transpose(ttnn.reshape(x, (1, S_, T_, C)), 1, 2)     # [1, I, S, C]
    o = ttnn.transpose(t, 2, 3)                                   # [1, I, C, S]
    ttnn.deallocate(t)
    return ttnn.reshape(o, (T_, C, S_))


def b_perm():
    return ttnn.permute(x, (2, 1, 0))


def b_tchain():
    t = ttnn.transpose(ttnn.reshape(x, (1, S_, T_, C)), 2, 3)     # [1, S, D, J]
    u = ttnn.transpose(t, 1, 2)                                   # [1, D, S, J]
    ttnn.deallocate(t)
    o = ttnn.transpose(u, 2, 3)                                   # [1, D, J, S]
    ttnn.deallocate(u)
    return ttnn.reshape(o, (C, T_, S_))


res = {}
for name, fn in (("a_perm", a_perm), ("a_tchain", a_tchain), ("b_perm", b_perm), ("b_tchain", b_tchain),
                 ("a_perm", a_perm), ("a_tchain", a_tchain)):
    ts, c0 = [], aiclk()
    for rep in range(REPS + 1):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        y = fn()
        ttnn.synchronize_device(dev)
        if rep:
            ts.append((time.perf_counter() - t0) * 1e3)
        if name not in res and rep == REPS:
            res[name] = ttnn.to_torch(y)
        ttnn.deallocate(y)
    log(ev="arm", arm=name, ms_median=statistics.median(ts), ms_min=min(ts), ms_max=max(ts), n=len(ts),
        aiclk_before=c0, aiclk_after=aiclk())
for t, b in (("a_tchain", "a_perm"), ("b_tchain", "b_perm")):
    log(ev="ab", arm=t, base=b, equal=bool(torch.equal(res[t], res[b])), shape=list(res[t].shape))
log(ev="end")
