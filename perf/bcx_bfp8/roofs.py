"""bcx-bfp8 roof probe: what one byte per element actually buys, per op class.

`bcx-plan` prices bfp8 off a NOMINAL byte ratio. A bfp8_b tile is 1088 B against bf16's 2048 B
(1024 mantissa bytes + 64 exponent bytes per 32x32 tile), so the nominal ratio is 0.5313 and the
nominal ceiling on a perfectly byte-bound op is 1.882x. This measures what the shipped primitives
actually deliver on the shapes `bcx-realcensus` ranked in the real Evoformer block backward, in
one process at one clock, arms interleaved.

Blackhole idles at 800 MHz and a sync-per-rep loop never leaves the floor, so every case ramps the
arbiter with a persistent matmul stream first and reads `tt_aiclk` synchronously on both sides of
its own timed loop. A reading taken at the idle floor is an artifact, not a slow op.
"""
import argparse, json, os, time, threading
from pathlib import Path

import torch
import ttnn

OUT = Path(__file__).resolve().parent


def sysfs_node(visible=None):
    visible = visible or os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    root = "/sys/class/tenstorrent"
    nodes = sorted(os.listdir(root),
                   key=lambda n: os.path.basename(os.path.realpath(f"{root}/{n}/device")))
    node = nodes[int(visible)]
    return f"{root}/{node}", os.path.basename(os.path.realpath(f"{root}/{node}/device"))


NODE, PCI = sysfs_node()
AICLK = f"{NODE}/tt_aiclk"


def aiclk():
    try:
        return int(open(AICLK).read().split()[0])
    except Exception:
        return None


class Sampler:
    """Fine-grained backstop for the two synchronous reads: a 0.1 ms op window catches no
    100 ms tick, which is how the first take of this probe reported no clock at all."""

    def __init__(self, dt=0.01):
        self.dt, self.samples, self._stop = dt, [], threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self._stop.is_set():
            c = aiclk()
            if c:
                self.samples.append((time.time(), c))
            self._stop.wait(self.dt)

    def stop(self):
        self._stop.set()

    def window(self, a, b):
        xs = sorted(c for t, c in self.samples if a <= t <= b)
        return {"n": 0} if not xs else {"n": len(xs), "min": xs[0],
                                        "median": xs[len(xs) // 2], "max": xs[-1]}


BPE = {ttnn.bfloat16: 2.0, ttnn.bfloat8_b: 1088.0 / 1024.0, ttnn.float32: 4.0}
NAME = {ttnn.bfloat16: "bf16", ttnn.bfloat8_b: "bfp8_b", ttnn.float32: "fp32"}
DTS = [ttnn.bfloat16, ttnn.bfloat8_b]


def mk(shape, dt):
    return ttnn.from_torch(torch.randn(*shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=DEV)


def prod(s):
    n = 1
    for d in s:
        n *= d
    return n


def ramp(burn=0.35):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < burn:
        for _ in range(12):
            ttnn.deallocate(ttnn.matmul(RAMP, RAMP))
        ttnn.synchronize_device(DEV)


def bench(fn, reps, warm=3):
    for _ in range(warm):                      # program-cache miss and tilize out of the median
        ttnn.deallocate(fn())
    ttnn.synchronize_device(DEV)
    ramp()
    c0 = aiclk()
    ttnn.synchronize_device(DEV)
    ts, t_start = [], time.perf_counter()
    w0 = time.time()
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(DEV)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(r)
    w1, t_end = time.time(), time.perf_counter()
    c1 = aiclk()
    ts.sort()
    return {"median": ts[len(ts) // 2], "p10": ts[max(0, int(0.1 * len(ts)))],
            "p90": ts[min(len(ts) - 1, int(0.9 * len(ts)))], "reps": reps,
            "wall": t_end - t_start,
            "aiclk": {"before": c0, "after": c1, **SAMP.window(w0, w1)}}


RES = []


def run(name, cls, dt, build, nbytes, reps, note=None):
    tens = []
    try:
        fn = build(dt, tens)
        r = bench(fn, reps)
    except Exception as e:
        RES.append({"op": name, "class": cls, "dtype": NAME[dt], "error": repr(e)[:240]})
        print(f"  {name:42s} {NAME[dt]:7s} UNSUPPORTED {repr(e)[:70]}", flush=True)
        for t in tens:
            try:
                ttnn.deallocate(t)
            except Exception:
                pass
        return
    b = nbytes(dt)
    r.update(op=name, dtype=NAME[dt], bytes=b, gbps=b / r["median"] / 1e9)
    r["class"] = cls
    if note:
        r["note"] = note
    RES.append(r)
    a = r["aiclk"]
    print(f"  {name:42s} {NAME[dt]:7s} {r['median']*1e3:8.3f} ms  {r['gbps']:7.1f} GB/s  "
          f"clk {a['before']}/{a.get('median')}/{a['after']}", flush=True)
    for t in tens:
        try:
            ttnn.deallocate(t)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out", default="roofs.json")
    a = ap.parse_args()

    global DEV, RAMP, SAMP
    DEV = ttnn.open_device(device_id=0)
    RAMP = ttnn.from_torch(torch.randn(4096, 4096), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=DEV)
    SAMP = Sampler()
    print(f"idle aiclk {aiclk()} on {PCI}; ramping", flush=True)
    ramp(2.0)
    print(f"ramped aiclk {aiclk()}", flush=True)

    # ---- copy / layout, on the shapes realcensus ranked in the block backward
    for shape in [(1, 256, 256, 128), (81, 4, 256, 256), (1, 64, 256, 256), (256, 4, 256, 32)]:
        n = prod(shape)
        print(f"clone {shape}", flush=True)
        for dt in DTS:
            run(f"clone{shape}", "copy", dt,
                lambda d, T, s=shape: (T.append(mk(s, d)), (lambda: ttnn.clone(T[0])))[1],
                lambda d, n=n: 2 * n * BPE[d], a.reps)

    # ---- eltwise
    for shape in [(1, 256, 256, 128), (81, 4, 256, 256), (1, 256, 256, 256), (256, 4, 256, 32)]:
        n = prod(shape)
        print(f"add {shape}", flush=True)
        for dt in DTS:
            run(f"add{shape}", "eltwise", dt,
                lambda d, T, s=shape: (T.extend([mk(s, d), mk(s, d)]),
                                       (lambda: ttnn.add(T[0], T[1])))[1],
                lambda d, n=n: 3 * n * BPE[d], a.reps)

    # ---- permute, the block's two big ones (last two axes)
    for shape in [(81, 4, 256, 256), (1, 64, 256, 256), (1, 256, 256, 128)]:
        n = prod(shape)
        print(f"permute {shape} (0,1,3,2)", flush=True)
        for dt in DTS:
            run(f"permute{shape}", "copy", dt,
                lambda d, T, s=shape: (T.append(mk(s, d)),
                                       (lambda: ttnn.permute(T[0], (0, 1, 3, 2))))[1],
                lambda d, n=n: 2 * n * BPE[d], a.reps)

    # ---- matmul: realcensus rows 1, 5, 18, plus a big square for the FLOP roof
    MM = [((256, 8, 32, 32), (256, 8, 32, 32)),
          ((81, 4, 256, 256), (81, 4, 256, 32)),
          ((1, 64, 256, 256), (1, 64, 256, 256)),
          ((1, 1, 4096, 4096), (1, 1, 4096, 4096))]
    for sa, sb in MM:
        na, nb = prod(sa), prod(sb)
        nc = prod(sa[:-1]) * sb[-1]
        print(f"matmul {sa} x {sb}", flush=True)
        for dt in DTS:
            run(f"matmul{sa}x{sb}", "matmul", dt,
                lambda d, T, A=sa, B=sb: (T.extend([mk(A, d), mk(B, d)]),
                                          (lambda: ttnn.matmul(T[0], T[1])))[1],
                lambda d, na=na, nb=nb, nc=nc: (na + nb + nc) * BPE[d],
                8 if sa[-1] == 4096 else a.reps, note=f"flops={2 * na * sb[-1]}")

    # ---- typecast: the bf16<->fp32 conversions realcensus puts at 13.7 % of the block,
    #      plus the two crossings a bfp8 arm has to pay at its own boundaries
    for shape in [(1, 256, 256, 128), (81, 4, 256, 256)]:
        n = prod(shape)
        print(f"typecast {shape}", flush=True)
        for src, dst in [(ttnn.bfloat16, ttnn.float32), (ttnn.bfloat8_b, ttnn.float32),
                         (ttnn.bfloat16, ttnn.bfloat8_b), (ttnn.bfloat8_b, ttnn.bfloat16)]:
            run(f"typecast{shape}->{NAME[dst]}", "copy", src,
                lambda d, T, s=shape, D=dst: (T.append(mk(s, d)),
                                              (lambda: ttnn.typecast(T[0], D)))[1],
                lambda d, n=n, D=dst: n * BPE[d] + n * BPE[D], a.reps,
                note=f"dst={NAME[dst]}")

    SAMP.stop()
    blob = {"host": os.uname().nodename, "pci": PCI, "aiclk_node": AICLK,
            "visible": os.environ.get("TT_VISIBLE_DEVICES"),
            "loadavg": os.getloadavg(), "torch_threads": torch.get_num_threads(),
            "commit": os.popen(f"git -C {OUT} rev-parse HEAD").read().strip(),
            "bpe": {NAME[k]: v for k, v in BPE.items()},
            "nominal_bfp8_over_bf16": BPE[ttnn.bfloat8_b] / BPE[ttnn.bfloat16],
            "cases": RES}
    (OUT / a.out).write_text(json.dumps(blob, indent=1, default=str))
    print("wrote", OUT / a.out, flush=True)
    ttnn.close_device(DEV)


if __name__ == "__main__":
    main()
