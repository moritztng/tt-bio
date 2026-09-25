#!/usr/bin/env python3
"""Attribute the taped backward's host-RSS growth to a per-node and per-block RATE.

`of3t-stepqb2` measured an exactness-ON crop-384 step and found the backward's RSS floor going
10.15 -> 15.014 GiB, +4.86 GiB, monotonic, with no new phase entered and no new model object
built. That is a phase boundary, not a slope, and a phase boundary names nothing. This samples
INSIDE the walk so the growth gets a denominator.

Two denominators, because the tape has two:

  PER NODE     `_retire(t)` runs once per tape node the walk retires, at the end of that node's
               own turn, so wrapping it counts the walk's progress exactly. RSS is read there.
  PER BLOCK    `checkpoint`'s `_recompute` calls `backward` again, so a nested `backward` IS a
               checkpointed segment's recompute. Wrapping `backward` with a depth counter gives
               entry/exit RSS per block for free and needs no edit to a contested file.

Both hooks are module-attribute rebinds on `tt_bio.autograd`: `_backward` looks `_retire` up as
a global and `_recompute` looks `backward` up as a global, so a rebind is seen. Nothing in
`tt_bio/` is edited to measure.

THE CENSUS is what tells a retained TAPE from a retained BUFFER. At a recompute boundary it
walks `gc.get_objects()` once and reports, deduplicated by storage pointer: live host torch
bytes, how many of those bytes are float64, and the number of live `Tensor` and `_Node` objects.
`host_f64_softmax`'s `bw` closes over `y64`, which at [384,4,384,384] is 1.688 GiB, so if inner
tapes survive their recompute the float64 line is where it shows.

THE `gc` ARM is the test of `checkpoint._recompute`'s own claim. Its last line is
`del y, inner, roots` with a comment saying the refcount frees the inner tape there, replacing a
`gc.collect()` per recompute that cost 0.16 s. `--arm gc` puts that collect back, outside the
engine, and reports what it reclaims. If the slope flattens, the comment is the defect and the
inner tape is cyclic garbage; if it does not, the inner tapes do free and the growth is on the
outer tape.

SAFETY. pc has 30 GB and no swap, and the kernel OOM-killer picks by score, not by cause -- a
sibling row's pytest suite is a plausible victim. The sampler hard-exits this process the moment
MemAvailable falls under --floor-gib, and every record is flushed as it is written, so a breached
run still yields its slope. A breached peak is a LOWER BOUND and the artifact says so.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

PAGE = os.sysconf("SC_PAGE_SIZE")
GIB = 1024.0 ** 3

_PHASE = ["import"]
_EV = None
_STOP = threading.Event()
_FLOOR = 2.0
_N = {"retire": 0, "bw_depth": 0, "recompute": 0, "sm_fwd": 0, "sm_bwd": 0, "gc_freed": 0,
      "gc_gib": 0.0, "trim_gib": 0.0}


_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


class _MallInfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in
                ("arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks",
                 "uordblks", "fordblks", "keepcost")]


_LIBC.mallinfo2.restype = _MallInfo2
M_TRIM_THRESHOLD, M_MMAP_THRESHOLD = -1, -3


def _mallinfo() -> dict:
    """glibc's own view, and it reports the MAIN ARENA only -- which is the point rather than a
    limitation. `arena` is heap the allocator holds, `uordblks` is what the program has asked
    for and not returned, `hblkhd` is what went to mmap. RSS minus `uordblks` minus `hblkhd`
    is memory no Python object and no live malloc block accounts for."""
    mi = _LIBC.mallinfo2()
    return {"arena_gib": round(mi.arena / GIB, 4), "hblkhd_gib": round(mi.hblkhd / GIB, 4),
            "uordblks_gib": round(mi.uordblks / GIB, 4),
            "fordblks_gib": round(mi.fordblks / GIB, 4),
            "keepcost_gib": round(mi.keepcost / GIB, 4), "hblks": mi.hblks}


def _rss() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * PAGE


def _vmhwm() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    return 0


def _avail() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


_T0 = time.perf_counter()


def _emit(rec: dict) -> None:
    rec["t"] = round(time.perf_counter() - _T0, 3)
    rec["rss_gib"] = round(_rss() / GIB, 4)
    _EV.write(json.dumps(rec) + "\n")
    _EV.flush()


def _sampler(period: float):
    while not _STOP.is_set():
        avail = _avail()
        rec = {"ev": "s", "t": round(time.perf_counter() - _T0, 3), "phase": _PHASE[-1],
               "rss_gib": round(_rss() / GIB, 4), "avail_gib": round(avail / GIB, 4),
               "n_retire": _N["retire"], "n_recompute": _N["recompute"]}
        if avail < _FLOOR * GIB:
            rec["BREACH"] = (f"MemAvailable {rec['avail_gib']} GiB under the {_FLOOR} GiB floor. "
                             f"Exiting before the kernel OOM-killer picks a victim by score. "
                             f"Every peak in this profile is a LOWER BOUND.")
            _EV.write(json.dumps(rec) + "\n")
            _EV.flush()
            os.fsync(_EV.fileno())
            os._exit(37)
        _EV.write(json.dumps(rec) + "\n")
        _EV.flush()
        time.sleep(period)


def _census(ag):
    """One pass over the heap: live host torch bytes (deduped by storage), and tape objects.

    Deduplicated by `data_ptr`, because a view and its base report the same storage and summing
    both would price one allocation twice -- exactly the error that would make a tape of views
    look like retained memory.
    """
    import torch
    t0 = time.perf_counter()
    seen: dict = {}
    n_t = n_node = n_torch = 0
    Tensor, Node = ag.Tensor, ag._Node
    for o in gc.get_objects():
        ty = type(o)
        if ty is Tensor:
            n_t += 1
        elif ty is Node:
            n_node += 1
        elif ty is torch.Tensor or ty is torch.nn.Parameter:
            try:
                if o.device.type != "cpu":
                    continue
                st = o.untyped_storage()
                seen[st.data_ptr()] = (st.nbytes(), o.dtype is torch.float64)
            except Exception:                                        # noqa: BLE001
                continue
            n_torch += 1
    tot = sum(v[0] for v in seen.values())
    f64 = sum(v[0] for v in seen.values() if v[1])
    return {"host_gib": round(tot / GIB, 4), "host_f64_gib": round(f64 / GIB, 4),
            "storages": len(seen), "torch_objs": n_torch, "tape_tensors": n_t,
            "tape_nodes": n_node, "census_s": round(time.perf_counter() - t0, 3)}


def _phased(fn, name):
    def wrapper(*a, **k):
        _PHASE.append(name)
        try:
            return fn(*a, **k)
        finally:
            _PHASE.pop()
    wrapper.__name__ = getattr(fn, "__name__", name)
    return wrapper


def main() -> int:
    global _EV, _FLOOR, _T0
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--exact", default="on", choices=("on", "off"))
    ap.add_argument("--arm", default="base", choices=("base", "gc", "trim", "probe", "mmap"),
                    help="gc: gc.collect() per recompute (tests _recompute's refcount claim). "
                         "trim: malloc_trim(0) per recompute (tests allocator retention). "
                         "probe: both, measured separately in one run, so the two are attributed "
                         "against the SAME walk rather than across two runs of a loaded box. "
                         "mmap: mallopt(M_MMAP_THRESHOLD) once at startup and nothing per "
                         "recompute, so the arena never takes the block in the first place.")
    ap.add_argument("--mmap-threshold-mib", type=float, default=2.0,
                    help="arm mmap only: blocks at or above this go straight to mmap and are "
                         "unmapped on free. Setting it also disables glibc's DYNAMIC threshold, "
                         "which is what raises the cap to 32 MiB after the first large free.")
    ap.add_argument("--trim-flag", default="", choices=("", "on", "off"),
                    help="set tt_bio.autograd.TRIM_HOST_HEAP, the shipped lever. Empty leaves "
                         "the engine default, which is the only honest control once the fix "
                         "has landed on this tree.")
    ap.add_argument("--grad-dump", default="",
                    help="sha256 per declared weight gradient after the backward -- the full "
                         "tensor set, not a sample, so an allocator change is proven not to be "
                         "a gradient change")
    ap.add_argument("--node-stride", type=int, default=10, help="RSS event every N retires")
    ap.add_argument("--census-stride", type=int, default=6, help="heap census every N recomputes")
    ap.add_argument("--floor-gib", type=float, default=3.0)
    ap.add_argument("--period", type=float, default=0.2)
    ap.add_argument("--with-optimizer", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    _FLOOR = a.floor_gib
    mmap_thr = None
    if a.arm == "mmap":
        # Before the engine is imported, so every host allocation the run makes is under it.
        mmap_thr = int(a.mmap_threshold_mib * 1024 * 1024)
        if _LIBC.mallopt(M_MMAP_THRESHOLD, mmap_thr) != 1:
            raise RuntimeError("mallopt(M_MMAP_THRESHOLD) refused")

    out_dir = REPO / "perf" / "of3t_tapemem" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{a.tokens}_{a.arm}"
    ev_path = out_dir / f"tape_{tag}.jsonl"
    step_json = out_dir / f"step_{tag}.json"
    _EV = open(ev_path, "w", buffering=1)
    _T0 = time.perf_counter()
    _EV.write(json.dumps({
        "header": True, "tag": tag, "argv": sys.argv[1:], "arm": a.arm, "tokens": a.tokens,
        "cycles": a.cycles, "samples": a.samples, "exact": a.exact,
        "floor_gib": _FLOOR, "period_s": a.period, "node_stride": a.node_stride,
        "census_stride": a.census_stride, "page_bytes": PAGE,
        "mem_total_gib": round(os.sysconf("SC_PHYS_PAGES") * PAGE / GIB, 3),
        "avail_at_start_gib": round(_avail() / GIB, 3),
        "mallinfo_at_start": _mallinfo(),
        "glibc": __import__("platform").libc_ver()[1], "python": sys.version.split()[0],
        "mmap_threshold_bytes": mmap_thr,
        "trim_flag_arg": a.trim_flag,
        "loadavg_at_start": open("/proc/loadavg").read().split()[:3],
        "host": os.uname().nodename,
        "visible_devices": os.environ.get("TT_VISIBLE_DEVICES", ""),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid()}) + "\n")

    threading.Thread(target=_sampler, args=(a.period,), daemon=True).start()

    _PHASE.append("import_engine")
    from tt_bio import autograd as ag
    from perf.of3t_stepfloor import fullstep as F
    _PHASE.pop()

    if a.trim_flag:
        ag.TRIM_HOST_HEAP = (a.trim_flag == "on")
    _N["trim_calls"] = 0
    _N["trim_s"] = 0.0
    _real_trim = ag._trim_host_heap

    def trim_timed():
        t0 = time.perf_counter()
        r = _real_trim()
        _N["trim_calls"] += 1
        _N["trim_s"] += time.perf_counter() - t0
        return r

    ag._trim_host_heap = trim_timed

    # --- the two hooks -------------------------------------------------------------------
    _real_retire = ag._retire

    def retire(t):
        _N["retire"] += 1
        n = _N["retire"]
        if n % a.node_stride == 0:
            _emit({"ev": "node", "i": n, "d": _N["bw_depth"], "rc": _N["recompute"]})
        return _real_retire(t)

    ag._retire = retire

    _real_backward = ag.backward

    def backward(roots, seeds=None):
        d = _N["bw_depth"]
        _N["bw_depth"] = d + 1
        if d > 0:
            _N["recompute"] += 1
        rc = _N["recompute"]
        _PHASE.append("backward" if d == 0 else "recompute")
        _emit({"ev": "bw_enter", "d": d, "rc": rc, "i": _N["retire"],
               "pins": len(ag._CKPT_PINS)})
        try:
            return _real_backward(roots, seeds)
        finally:
            _PHASE.pop()
            _N["bw_depth"] = d
            rec = {"ev": "bw_exit", "d": d, "rc": rc, "i": _N["retire"],
                   "pins": len(ag._CKPT_PINS), "sm_bwd": _N["sm_bwd"]}
            if d > 0 and a.arm in ("gc", "trim", "probe"):
                r0 = _rss()
                rec["mi_before"] = _mallinfo()
                if a.arm in ("gc", "probe"):
                    freed = gc.collect()
                    r1 = _rss()
                    _N["gc_freed"] += freed
                    _N["gc_gib"] += (r0 - r1) / GIB
                    rec["gc_objs"] = freed
                    rec["gc_gib"] = round((r0 - r1) / GIB, 4)
                else:
                    r1 = r0
                if a.arm in ("trim", "probe"):
                    _LIBC.malloc_trim(0)
                    r2 = _rss()
                    _N["trim_gib"] += (r1 - r2) / GIB
                    rec["trim_gib"] = round((r1 - r2) / GIB, 4)
                    rec["mi_after"] = _mallinfo()
            if d > 0 and rc % a.census_stride == 0:
                rec["census"] = _census(ag)
            _emit(rec)

    ag.backward = backward

    _real_smv = ag.host_f64_softmax_values

    def smv(v, dim=-1):
        _N["sm_bwd" if _N["bw_depth"] else "sm_fwd"] += 1
        return _real_smv(v, dim)

    ag.host_f64_softmax_values = smv

    _PARAMSET = {}
    if a.grad_dump:
        import hashlib

        import torch
        import ttnn

        _real_declare = F.declare_all

        def declare_all(trunk, sampler, out):
            """`declare_all` returns the NAMED weight set, and a digest keyed on `id()` is
            worthless across two processes. Capturing it here is what makes the comparison a
            per-parameter one rather than a count."""
            ps = _real_declare(trunk, sampler, out)
            _PARAMSET.update({n: t for n, t in ps.items()})
            return ps

        F.declare_all = declare_all
        _real_release = ag.release_pins
        _dumped = {"done": False}

        def release_pins():
            """Digest every declared weight's gradient before the pins go, over its BYTES.

            An allocator change cannot move a float, but "cannot" is an argument and this row
            was told to bring a proof. The whole declared set, no sampling, no norms.
            """
            if not _dumped["done"] and _PARAMSET:
                _dumped["done"] = True
                rows = {}
                for name, t in sorted(_PARAMSET.items()):
                    g = getattr(t, "grad", None)
                    if g is None:
                        rows[name] = None
                        continue
                    try:
                        h = ttnn.to_torch(g).contiguous()
                        # `numpy()` refuses bfloat16 and 1,217 of 3,152 gradients ARE bfloat16,
                        # so the first version of this digest recorded an error for 39 % of the
                        # set and two arms "agreed" by both failing. The uint8 view is the raw
                        # bytes for every dtype, which is what bit-identical means.
                        raw = h.view(torch.uint8).numpy().tobytes()
                        rows[name] = {"sha256": hashlib.sha256(raw).hexdigest(),
                                      "bytes": len(raw),
                                      "shape": list(h.shape), "dtype": str(h.dtype)}
                    except Exception as exc:                          # noqa: BLE001
                        rows[name] = {"error": repr(exc)[:160]}
                got = sum(1 for v in rows.values() if isinstance(v, dict) and "sha256" in v)
                Path(a.grad_dump).write_text(json.dumps(
                    {"tag": tag, "arm": a.arm, "tokens": a.tokens, "cycles": a.cycles,
                     "samples": a.samples, "declared": len(rows), "with_grad": got,
                     "grads": rows}, indent=1, sort_keys=True))
                print(f"[tapeprof] grad dump: {got} of {len(rows)} declared weights "
                      f"-> {a.grad_dump}", flush=True)
            return _real_release()

        ag.release_pins = release_pins

    # --- phase tags, same set `of3t-restep`'s rssprofile.py uses ---------------------------
    _real_trunk = F.trunk_forward

    def trunk_tagged(trunk, held, cycles, taped):
        _PHASE.append("trunk_taped" if taped else "trunk_nograd")
        try:
            return _real_trunk(trunk, held, cycles, taped)
        finally:
            _PHASE.pop()

    F.trunk_forward = trunk_tagged
    F.diffusion_train = _phased(F.diffusion_train, "diffusion")
    F.host_losses = _phased(F.host_losses, "losses")
    F.declare_all = _phased(F.declare_all, "declare_weights")
    from perf.of3t_perf import step as S
    import tt_baseline as B
    B.build_fold = _phased(B.build_fold, "capture_build_fold")
    S.capture = _phased(S.capture, "capture_prep")

    argv = ["fullstep.py", "--tokens", str(a.tokens), "--cycles", str(a.cycles),
            "--samples", str(a.samples), "--reps", str(a.reps), "--out", str(step_json)]
    if not a.with_optimizer:
        # The optimizer is `of3t-optorder`'s term, measured and fixed there at 4.263 GiB. Leaving
        # it in would add a 6 GiB step AFTER the walk this row is measuring and buy nothing.
        argv.append("--no-optimizer")
    sys.argv = argv
    _PHASE.append("setup")
    t0 = time.perf_counter()
    if a.exact == "on":
        rc = F.main()
    else:
        with ag.exact_training(False):
            rc = F.main()
    wall = round(time.perf_counter() - t0, 3)
    _PHASE.pop()
    _STOP.set()
    time.sleep(a.period * 2)
    _EV.write(json.dumps({"footer": True, "rc": rc, "wall_s": wall,
                          "vmhwm_gib": round(_vmhwm() / GIB, 4),
                          "avail_at_end_gib": round(_avail() / GIB, 3),
                          "counts": dict(_N), "mallinfo_at_end": _mallinfo(),
                          "trim_host_heap_effective": bool(ag.TRIM_HOST_HEAP),
                          "exact_training_ops": list(ag.exact_training_ops()),
                          "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime())}) + "\n")
    _EV.close()
    print(f"[tapeprof] rc={rc} wall={wall}s vmhwm={_vmhwm()/GIB:.3f} GiB "
          f"retires={_N['retire']} recomputes={_N['recompute']} -> {ev_path}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
