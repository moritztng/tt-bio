#!/usr/bin/env python3
"""One 512 aa fold cell, measured at a clock this process pins and samples while folding.

Why this exists rather than reusing perf/other512/fold_ab_multi.py: every Boltz-2 and Protenix
cell this campaign is scored against was taken before the AICLK governor was understood, and the
governor alone swings a 512 aa fold 1.27-1.41x. A fold blocks on hundreds of host syncs, the chip
idles in the gaps, and the ARC governor drops it to its 800 MHz floor, so an unpinned number
measures the box mood rather than the tree. This harness pins the clock through tt-kmd's ARC
message queue, keeps it pinned with a watchdog, and records the clock DURING each fold, per fold.

It is deliberately self-contained (no import from another perf/ directory) because it is also run
against an OLD tree, extracted from git at the commit a published cell came from. One file copies.

Not doing less of the model's own work: recycles, sampling steps, sample count and seed all come
from the tree being measured, and are recorded per run.
"""
from __future__ import annotations

import argparse
import atexit
import fcntl
import hashlib
import json
import os
import signal
import statistics as st
import struct
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "gpu_vs_tt"))

# ---- ARC FORCE_AICLK -----------------------------------------------------------------------
# tt-kmd owns the ARC message queue and multiplexes it over every open fd, so this rides
# alongside UMD instead of racing it. Layout is UMD's: message[0] type, [1..7] arguments.
_IOCTL_SMC_MSG = (0xFA << 8) | 17
_POST, _POLL = 1 << 0, 1 << 1
_FORCE_AICLK = 0x33
_LAYOUT = "=IIII8I"
_SAG_MHZ = 50


def _smc(fd, msg_type, *args):
    msg = [msg_type] + list(args) + [0] * (7 - len(args))
    fcntl.ioctl(fd, _IOCTL_SMC_MSG, struct.pack(_LAYOUT, 48, _POST, 0, 0, *msg))
    deadline = time.time() + 2.0
    while time.time() < deadline:
        buf = bytearray(struct.pack(_LAYOUT, 48, _POLL, 0, 0, *([0] * 8)))
        try:
            fcntl.ioctl(fd, _IOCTL_SMC_MSG, buf, True)
        except OSError as e:
            if e.errno == 11:
                time.sleep(0.005)
                continue
            raise
        resp = struct.unpack(_LAYOUT, bytes(buf))[4:]
        return resp[0] & 0xFF, resp[0] >> 16
    raise TimeoutError("no ARC response")


def _aiclk(node):
    return int(Path(f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk").read_text())


def _own_nodes():
    """Which /dev/tenstorrent/N this process holds.

    TT_VISIBLE_DEVICES is a UMD logical id and is NOT the device node, so the force is aimed off
    our own fd table, never off the environment.
    """
    nodes = set()
    for fd in Path("/proc/self/fd").iterdir():
        try:
            t = os.readlink(fd)
        except OSError:
            continue
        if t.startswith("/dev/tenstorrent/"):
            tail = t.rsplit("/", 1)[1]
            if tail.isdigit():
                nodes.add(int(tail))
    return sorted(nodes)


class Clock:
    """A forced AICLK on every chip this process has open, plus the thread that keeps it there.

    The force is chip state, not fd state: it outlives the fd and it outlives the process, so
    release is wired to atexit and to the fatal signals as well as the normal path. A run that
    died without releasing would leave the card at burst and ~40 W above idle indefinitely.
    """

    def __init__(self, target):
        self.target = target
        self.fds = {}
        self.reasserts = 0
        self.stop = threading.Event()
        self.thread = None

    def acquire(self):
        for n in _own_nodes():
            fd = os.open(f"/dev/tenstorrent/{n}", os.O_RDWR | os.O_APPEND)
            status, _ = _smc(fd, _FORCE_AICLK, self.target)
            if status == 0xFF:
                os.close(fd)
                raise RuntimeError(f"firmware refused FORCE_AICLK on node {n}")
            self.fds[n] = fd
        atexit.register(self.release)
        for s in (signal.SIGINT, signal.SIGTERM):
            prev = signal.getsignal(s)

            def handler(sig, frm, p=prev):
                self.release()
                if callable(p):
                    p(sig, frm)
                else:
                    os._exit(130)

            signal.signal(s, handler)
        self.thread = threading.Thread(target=self._watch, daemon=True)
        self.thread.start()
        time.sleep(1.0)
        return {n: _aiclk(n) for n in self.fds}

    def _watch(self):
        while not self.stop.wait(0.25):
            for n, fd in list(self.fds.items()):
                try:
                    if _aiclk(n) < self.target - _SAG_MHZ:
                        _smc(fd, _FORCE_AICLK, self.target)
                        self.reasserts += 1
                except Exception:      # noqa: BLE001 -- a watchdog must never kill a fold
                    pass

    def release(self):
        self.stop.set()
        for n, fd in list(self.fds.items()):
            try:
                _smc(fd, _FORCE_AICLK, 0)
            except Exception:          # noqa: BLE001
                pass
            os.close(fd)
            self.fds.pop(n, None)


class Sampler(threading.Thread):
    """AICLK and card power at 4 Hz while a fold runs.

    Reads sysfs only, so the probe cannot be the contention it measures.
    """

    def __init__(self, nodes):
        super().__init__(daemon=True)
        self.nodes = nodes
        self.stop = threading.Event()
        self.a = []
        self.w = []

    def run(self):
        clk = [Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk") for n in self.nodes]
        pw = [next(iter(Path(f"/sys/class/tenstorrent/tenstorrent!{n}").glob(
            "device/hwmon/hwmon*/power1_input")), None) for n in self.nodes]
        while not self.stop.wait(0.25):
            for p in clk:
                try:
                    v = int(p.read_text())
                    if v < 3000:
                        self.a.append(v)
                except (OSError, ValueError):
                    pass
            for p in pw:
                if p is not None:
                    try:
                        self.w.append(int(p.read_text()))
                    except (OSError, ValueError):
                        pass

    def take(self):
        a, w = self.a[:], self.w[:]
        self.a.clear()
        self.w.clear()
        if not a:
            return {}
        return {"aiclk_mean": round(sum(a) / len(a), 1), "aiclk_min": min(a),
                "aiclk_max": max(a), "aiclk_n": len(a),
                "power_w_mean": round(sum(w) / len(w) / 1e6, 1) if w else None}


def foreign_holders():
    """Every OTHER process holding a /dev/tenstorrent fd, recorded per fold.

    This is the quiet-box check. benchlock takes a lock at one instant and cannot see a co-tenant
    already mid-run without it, which is how a 10-fold session was lost to a board partner on
    2026-09-16.
    """
    me, out = os.getpid(), []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == me:
            continue
        try:
            for fd in (p / "fd").iterdir():
                if os.readlink(fd).startswith("/dev/tenstorrent/"):
                    cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                        "utf8", "replace")[:120]
                    out.append({"pid": int(p.name), "cmd": cmd})
                    break
        except (OSError, PermissionError):
            continue
    return out


WALL = defaultdict(lambda: {"n": 0, "s": 0.0})
DEV = {"d": None}


def timed(key, fn, *a, **kw):
    import ttnn
    ttnn.synchronize_device(DEV["d"])
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    ttnn.synchronize_device(DEV["d"])
    w = WALL[key]
    w["n"] += 1
    w["s"] += time.perf_counter() - t0
    return out


TIMER_NAMES = [("Pairformer", "stage"), ("MSAModule", "stage"), ("TemplateModule", "stage"),
               ("PairformerLayer", "block"), ("MSALayer", "block"),
               ("TriangleMultiplication", "body"), ("TriangleAttention", "body"),
               ("AttentionPairBias", "body"), ("PairWeightedAveraging", "body"),
               ("Transition", "body"), ("OuterProductMean", "body"),
               ("DiffusionTransformer", "stage"), ("AtomAttentionEncoder", "stage"),
               ("AtomAttentionDecoder", "stage"), ("ConfidenceHead", "stage")]


def install_timers():
    """Block-level timers, PROBED rather than assumed.

    A class this tree does not have would crash the fold, and a class it has under another name
    would silently go uncounted, so every installed key is recorded and reported.
    """
    import tt_bio.tenstorrent as T
    installed = []
    mods = [T]
    for extra in ("protenix", "boltz2", "opendde"):
        try:
            mods.append(__import__(f"tt_bio.{extra}", fromlist=["x"]))
        except Exception:              # noqa: BLE001
            pass
    for mod in mods:
        short = mod.__name__.rsplit(".", 1)[-1]
        for nm, kind in TIMER_NAMES:
            cls = getattr(mod, nm, None)
            if cls is None or not callable(getattr(cls, "__call__", None)):
                continue
            key = f"{kind}:{nm}" if mod is T else f"{kind}:{short}.{nm}"
            if key in installed:
                continue
            f = cls.__call__
            cls.__call__ = (lambda g, k: lambda self, *x, **kw: timed(k, g, self, *x, **kw))(f, key)
            installed.append(key)
    return installed


def patch_boltz2_cfg():
    """Inject the Boltz-2 hyperparameters build_fold's cfg does not carry.

    Without this, load_model raises KeyError('conf_kwargs'). Exactly what tt_bio.main builds,
    process-local.
    """
    from tt_bio import worker as _W
    import tt_baseline as B

    _diffusion = {"step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003,
                  "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0, "sigma_data": 16.0,
                  "P_mean": -1.2, "P_std": 1.5, "coordinate_augmentation": True,
                  "alignment_reverse_diff": True, "synchronize_sigmas": True}
    _pairformer = {"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True}
    _msa = {"subsample_msa": True, "num_subsampled_msa": 1024, "use_paired_feature": True,
            "msa_s": 64, "msa_blocks": 4, "msa_dropout": 0.15, "z_dropout": 0.25,
            "pairwise_head_width": 32, "pairwise_num_heads": 4,
            "activation_checkpointing": True}
    _steering = {"fk_steering": False, "physical_guidance_update": False,
                 "contact_guidance_update": True, "num_particles": 3, "fk_lambda": 4.0,
                 "fk_resampling_interval": 3, "num_gd_steps": 20}
    _conf = dict(predict_args={"recycling_steps": B.RECYCLING_STEPS,
                               "sampling_steps": B.SAMPLING_STEPS,
                               "diffusion_samples": B.DIFFUSION_SAMPLES,
                               "max_parallel_samples": None},
                 diffusion_process_args=_diffusion, pairformer_args=_pairformer, msa_args=_msa,
                 steering_args=_steering, use_kernels=True, use_tenstorrent=True, trace=False,
                 diffusion_trace=False)
    _aff = dict(predict_args={"recycling_steps": 5, "sampling_steps": 200, "diffusion_samples": 5,
                              "max_parallel_samples": 1},
                diffusion_process_args=_diffusion, pairformer_args=_pairformer, msa_args=_msa,
                steering_args=dict(_steering, contact_guidance_update=False),
                affinity_mw_correction=False, use_tenstorrent=True, trace=False,
                diffusion_trace=False)
    _orig = _W._WorkerState.load_model

    def _load(self, cfg):
        cfg.setdefault("conf_kwargs", _conf)
        cfg.setdefault("aff_kwargs", _aff)
        cfg.setdefault("use_potentials", False)
        return _orig(self, cfg)

    _W._WorkerState.load_model = _load


def sha_dir(d):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
            for p in sorted(Path(d).glob("*")) if p.is_file()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=6, help="timed folds after the cold one")
    ap.add_argument("--clock", type=int, default=1350, help="MHz to pin; 0 leaves the governor")
    ap.add_argument("--timers", action="store_true", help="block timers; inflates fold_s")
    ap.add_argument("--fixdir", type=Path, default=ROOT / "perf" / "size512" / "fixtures")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import tt_bio.tenstorrent as T
    import tt_baseline as B
    import importlib.metadata as im

    assert Path(T.__file__).resolve().is_relative_to(ROOT), \
        f"tt_bio resolved to {T.__file__}, outside {ROOT}: set PYTHONPATH to the tree under test"

    # p300 boxes need the mesh descriptor for a bare single-chip open. Guarded, because the old
    # tree may not carry the helper.
    try:
        from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
        if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
            mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
            if mgd:
                os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    except Exception:                  # noqa: BLE001
        pass

    # Each model at ITS OWN shipped recycling/sampling default, read off the tree under test.
    try:
        from tt_bio.main import _resolve_recycling_steps, _resolve_sampling_steps
        B.RECYCLING_STEPS = _resolve_recycling_steps(None, a.model)
        B.SAMPLING_STEPS = _resolve_sampling_steps(None, a.model)
    except Exception as e:             # noqa: BLE001
        print(f"note: tree has no _resolve_*_steps ({e}); using tt_baseline defaults "
              f"rec={B.RECYCLING_STEPS} steps={B.SAMPLING_STEPS}", flush=True)
    if a.model == "boltz2":
        patch_boltz2_cfg()

    installed = install_timers() if a.timers else []

    res = {"tag": a.tag, "model": a.model, "size": a.size, "host": os.uname().nodename,
           "card_env": os.environ.get("TT_VISIBLE_DEVICES"), "ttnn": im.version("ttnn"),
           "tree": str(ROOT), "clock_target": a.clock, "timers": installed,
           "recycling_steps": B.RECYCLING_STEPS, "sampling_steps": B.SAMPLING_STEPS,
           "diffusion_samples": B.DIFFUSION_SAMPLES, "seed": getattr(B, "SEED", None),
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "loadavg_start": open("/proc/loadavg").read().split()[:3],
           "foreign_at_start": foreign_holders(), "runs": []}

    def dump():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))

    tgt = a.fixdir / f"cdk2x2_{a.size}.yaml"
    a3m = a.fixdir / f"cdk2x2_{a.size}.a3m"
    res["fixture_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (tgt, a3m)}
    built = B.build_fold(a.model, ROOT / f".msa_pvx_{a.size}", tgt, a3m)
    one_fold, meta = built[0], built[1]
    DEV["d"] = T.get_device()
    g = DEV["d"].compute_with_storage_grid_size()
    res["grid"] = [g.x, g.y]
    res["device_nodes"] = _own_nodes()
    struct_dir = Path(meta["struct_dir"])

    clock = None
    if a.clock:
        clock = Clock(a.clock)
        res["clock_after_force"] = clock.acquire()
        print(f"forced AICLK -> {res['clock_after_force']}", flush=True)
    samp = Sampler(res["device_nodes"])
    samp.start()
    try:
        for i in range(a.reps + 1):
            fh = foreign_holders()
            samp.take()
            fold_s, m = one_fold()
            rec = {"i": i, "cold": i == 0, "fold_s": round(fold_s, 3),
                   "plddt": m.get("plddt"), "n_tokens": m.get("n_tokens"),
                   "cif_sha256": sha_dir(struct_dir),
                   "foreign_tt": fh,
                   "loadavg1": float(open("/proc/loadavg").read().split()[0]),
                   **samp.take()}
            if a.timers:
                rec["wall_ms"] = {k: {"calls": v["n"], "ms": round(v["s"] * 1e3, 2)}
                                  for k, v in sorted(WALL.items(), key=lambda kv: -kv[1]["s"])}
                WALL.clear()
            res["runs"].append(rec)
            dump()
            tag = "cold" if i == 0 else "    "
            print(f"  [{i}] {tag} {fold_s:8.3f}s  aiclk {rec.get('aiclk_mean')} "
                  f"(min {rec.get('aiclk_min')}) pw {rec.get('power_w_mean')}W  "
                  f"plddt {m.get('plddt')}  foreign {len(fh)}", flush=True)
    finally:
        samp.stop.set()
        if clock is not None:
            res["clock_reasserts"] = clock.reasserts
            clock.release()

    warm = [r for r in res["runs"] if not r["cold"] and "fold_s" in r]
    if warm:
        f = sorted(r["fold_s"] for r in warm)
        clks = [r["aiclk_mean"] for r in warm if r.get("aiclk_mean")]
        res["summary"] = {
            "n": len(f), "fold_s_sorted": f, "median_fold_s": round(st.median(f), 3),
            "min_fold_s": f[0], "max_fold_s": f[-1],
            "aa_floor_s": round(f[-1] - f[0], 3),
            "aa_floor_pct": round(100 * (f[-1] - f[0]) / st.median(f), 2),
            "aiclk_mean_over_timed": round(st.mean(clks), 1) if clks else None,
            "aiclk_min_over_timed": min([r["aiclk_min"] for r in warm if r.get("aiclk_min")],
                                        default=None),
            "digests": sorted({d for r in warm for d in r["cif_sha256"].values()}),
            "plddts": sorted({r["plddt"] for r in warm if r["plddt"] is not None}),
            "cotenanted_folds": sum(1 for r in warm if r["foreign_tt"]),
            "clean_session": all(not r["foreign_tt"] for r in res["runs"]),
        }
        print(json.dumps(res["summary"], indent=1), flush=True)
    dump()
    print("wrote", a.out, flush=True)
    T.cleanup()


if __name__ == "__main__":
    main()
