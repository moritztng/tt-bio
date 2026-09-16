#!/usr/bin/env python3
"""What the Boltz-2 512 aa cell costs on `main` today, measured rather than summed.

One arm: the shipped default of whatever tree this runs from. No flags are set, no levers are
pinned, so the number is what a user gets. The campaign has a published cell and a state doc that
both quote seconds measured on older trees; this re-takes the second so neither has to be carried
forward by arithmetic.

Protocol, fixture and cfg come from perf/b2x-flag-levers/ab_flag_levers.py, the fold this campaign
times everywhere: 200 sampling steps, 3 recycles, full MSA, 1 sample, seed 0, templates off.
Warm folds only in the summary; the cold ones are kept in the record and excluded from the median.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, shutil, socket, statistics as st
import sys, tempfile, threading, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "_b2x_flaglev", REPO / "perf" / "b2x-flag-levers" / "ab_flag_levers.py")
LEV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(LEV)

FIX = REPO / "perf" / "size512" / "fixtures"
OUT: dict = {}
OUT_PATH: Path | None = None


class ClockSampler(threading.Thread):
    """The card ARC clock, sampled from sysfs at 5 Hz while a fold runs.

    This box reads the same fixture at 21.5 s and at 15.2 s on the same card, the same tree and
    the same protocol, and the campaign attributed that swing to a host co-tenant. The ARC
    governor is the better suspect: a 200-step fold blocks on ~200 host syncs, so the chip idles
    between them and AICLK falls back to its 800 MHz base. Nothing here opens the device, so the
    probe cannot be the contention it measures.
    """

    ROOT = Path("/sys/class/tenstorrent")

    def __init__(self, card):
        super().__init__(daemon=True)
        self.card, self.stop = card, threading.Event()
        self.aiclk, self.power = [], []

    def run(self):
        clk = self.ROOT / self.card / "tt_aiclk"
        pw = next(iter((self.ROOT / self.card).glob("device/hwmon/hwmon*/power1_input")), None)
        while not self.stop.wait(0.2):
            try:
                v = int(clk.read_text().strip())
                if v < 3000:
                    self.aiclk.append(v)
            except (OSError, ValueError):
                pass
            if pw is not None:
                try:
                    self.power.append(int(pw.read_text().strip()))
                except (OSError, ValueError):
                    pass

    def take(self):
        a, w = self.aiclk[:], self.power[:]
        self.aiclk.clear()
        self.power.clear()
        if not a:
            return {}
        return {"aiclk_mean": round(sum(a) / len(a), 1), "aiclk_max": max(a),
                "aiclk_frac_boost": round(sum(1 for x in a if x > 900) / len(a), 3),
                "aiclk_n": len(a),
                "power_w_mean": round(sum(w) / len(w) / 1e6, 1) if w else None}


def dump():
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def tt_holders(exclude_pid: int) -> list:
    """Every other process on this host holding a /dev/tenstorrent fd, with its CPU share.

    benchlock serialises timed runs that use it, and is blind to one that does not. This box
    folds 512 aa in ~14.7 s idle and ~20 s with one co-tenant fold running, so a foreign holder
    is not noise to average out, it is a different measurement. Recorded per fold, not once.
    """
    out = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == exclude_pid:
            continue
        try:
            fds = list((proc / "fd").iterdir())
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
        n = 0
        for fd in fds:
            try:
                if "tenstorrent" in os.readlink(fd):
                    n += 1
            except OSError:
                pass
        if n:
            try:
                cmd = (proc / "cmdline").read_bytes().decode(errors="replace").replace("\0", " ")
            except OSError:
                cmd = "?"
            out.append({"pid": int(proc.name), "fds": n, "cmd": cmd[:120]})
    return out



def flag_snapshot() -> dict:
    """Resolved values of the levers this campaign merged, read off the modules, not off the env.

    A lever that does not exist in the tree being measured reports "not in tree" rather than False,
    so an older tree can be measured with the same harness without reading as a lever turned off.
    """
    import tt_bio.tenstorrent as TT
    import tt_bio.triatt_qkv as TQ
    import tt_bio.boltz2 as B2

    def val(fn):
        try:
            v = fn()
        except AttributeError:
            return "not in tree"
        return bool(v)

    return {
        "TT_BIO_DEVICE_CONDITIONING": val(lambda: B2._device_conditioning()),
        "TT_BIO_DEVICE_ZINIT": val(lambda: B2._device_zinit()),
        "TT_BIO_DEVICE_CONFIDENCE": val(lambda: B2._device_confidence()),
        "TT_BIO_DEVICE_CONF_HEADS": val(lambda: B2._device_conf_heads()),
        "TT_BIO_TRIATT_FUSED_QKVG": val(lambda: TQ._QKVG_ENABLED),
        "TT_BIO_TRIATT_FUSED_QKVGB": val(lambda: TQ._QKVGB_ENABLED),
        "TT_BIO_TRIMUL_FUSED_GOUT": val(lambda: TT._TRIMUL_FUSED_GOUT),
        "TT_BIO_SDPA_GRID_Q_CHUNK": val(lambda: TT._SDPA_GRID_Q_CHUNK),
        "TT_BIO_ATOM_SHIFT_GATHER": val(lambda: not TT._ATOM_SHIFT_GATHER_OFF),
        "TT_BIO_ATOM_L1": val(lambda: TT._ATOM_L1),
        "TT_BIO_UNFUSED_SILU": val(lambda: TT._UNFUSED_SILU),
        "TT_BIO_DIT_COND_HOIST": val(lambda: TT._B2_DIT_COND_HOIST),
        "TT_BIO_TRANSITION_L1_ROWS": val(lambda: TT._TRANSITION_L1_ROWS),
        "TT_BIO_TRIMUL_GP_BANK_SPLIT": val(lambda: TT._TRIMUL_GP_BANK_SPLIT),
    }


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=10, help="timed folds")
    ap.add_argument("--warm", type=int, default=2, help="discarded folds before timing")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--fixture", default="cdk2x2_512")
    ap.add_argument("--clock", default=None,
                    help="card directory under /sys/class/tenstorrent whose AICLK is sampled "
                         "while folding; nothing here opens the device")
    ap.add_argument("--allow-cotenant", action="store_true",
                    help="measure anyway with a foreign device holder on the box; the record "
                         "keeps the holder list per fold and the number is not publishable")
    args = ap.parse_args()
    OUT_PATH = args.out
    LEV.SAMPLING_STEPS, LEV.RECYCLING_STEPS = args.steps, args.recycles

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not this tree")
    leaked = sorted(k for k in os.environ if k.startswith("TT_BIO_")
                    and k not in ("TT_BIO_LEASE_CARDS", "TT_BIO_LEASE_HOLDER"))
    assert not leaked, f"a lever is pinned in the environment, this is not the default: {leaked}"

    holders = tt_holders(os.getpid())
    if holders and not args.allow_cotenant:
        print(f"REFUSING: another process holds a Tenstorrent device: {holders}", flush=True)
        return 3
    dev = get_device()
    OUT["env"] = {
        "host": socket.gethostname(), "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "tt_bio_file": _TB.__file__, "fixture": args.fixture,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(), "steps": args.steps, "recycles": args.recycles,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "ttnn": getattr(ttnn, "__version__", "?"),
    }
    OUT["flags"] = flag_snapshot()
    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-cellrecheck-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{args.fixture}.yaml", (FIX / f"{args.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    LEV._ensure_local_artifacts = _ensure_local_artifacts
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-wave2-cell-recheck", cfg)

    clk = ClockSampler(args.clock) if args.clock else None
    if clk:
        clk.start()

    def fold(i: int, warm: bool) -> dict:
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(FIX / f"{args.fixture}.yaml", cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        clock = clk.take() if clk else {}
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        return {"i": i, "warmup": warm, "fold_s": round(wall, 3),
                "foreign_tt": tt_holders(os.getpid()),
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
                "loadavg1": round(os.getloadavg()[0], 2), **clock}

    runs = []
    for i in range(args.warm + args.reps):
        warm = i < args.warm
        r = fold(i, warm)
        runs.append(r)
        tag = "warm" if warm else "fold"
        print("  {} {:<3d} {:7.3f}s plddt={} load={} aiclk={} boost={} cif {}".format(
            tag, i, r["fold_s"], r["plddt"], r["loadavg1"], r.get("aiclk_mean"),
            r.get("aiclk_frac_boost"), r["cif_sha256"][:16]), flush=True)
        OUT["runs"] = runs
        dump()

    timed = [r["fold_s"] for r in runs if not r["warmup"]]
    shas = sorted({r["cif_sha256"][:16] for r in runs if not r["warmup"]})
    plddts = sorted({r["plddt"] for r in runs if not r["warmup"]})
    med = st.median(timed)
    OUT["n"] = len(timed)
    OUT["fold_s_sorted"] = sorted(timed)
    OUT["median_fold_s"] = round(med, 3)
    OUT["min_fold_s"], OUT["max_fold_s"] = min(timed), max(timed)
    OUT["spread_pct"] = round(100.0 * (max(timed) - min(timed)) / min(timed), 2)
    OUT["cif_sha256_16"] = shas
    OUT["one_digest"] = len(shas) == 1
    OUT["plddt"] = plddts
    clocks = [r["aiclk_mean"] for r in runs if not r["warmup"] and "aiclk_mean" in r]
    if clocks:
        OUT["aiclk_mean_over_timed_folds"] = round(sum(clocks) / len(clocks), 1)
        OUT["aiclk_mean_per_fold"] = clocks
    OUT["cotenanted_folds"] = sum(1 for r in runs if not r["warmup"] and r["foreign_tt"])
    OUT["clean_session"] = OUT["cotenanted_folds"] == 0
    OUT["vs"] = {
        "closing_md_20.113": round(20.113 / med, 5),
        "perf_page_17.340": round(17.340 / med, 5),
        "h200_7.538_whole_fold": round(med / 7.538, 4),
    }
    dump()
    print(json.dumps({k: OUT[k] for k in
                      ("n", "median_fold_s", "min_fold_s", "max_fold_s", "spread_pct",
                       "clean_session", "cotenanted_folds",
                       "cif_sha256_16", "plddt", "vs", "flags")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
