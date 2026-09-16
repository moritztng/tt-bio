#!/usr/bin/env python3
"""Does forcing the ARC clock to burst make the Boltz-2 512 aa fold faster on a p300c?

`b2z2-wave2-cell-recheck` fitted `fold_s = 2.901 + 15355/AICLK_MHz` on this fixture and found the
governor, not the code, sets the second. That predicts 14.27 s if the clock could be held at the
1350 MHz burst instead of sagging to 850-1124 MHz across a fold, because a fold blocks on ~200 host
syncs and the chip idles in every gap.

The knob is ARC message `FORCE_AICLK` (0x33), sent through tt-kmd's SMC message queue, which any
process can reach without touching UMD or the engine. Argument is the target in MHz; 0 hands the
clock back to the governor, which is the shipped default and so is the control arm.

Two things this does NOT use, both measured dead first: `TENSTORRENT_IOCTL_SET_POWER_STATE` with
`TT_POWER_FLAG_MAX_AI_CLK` (firmware accepts the message and the clock does not move), and
`AICLK_GO_BUSY`, which UMD already sends at every device open.

The arms alternate fold by fold in one process, one device open, one model load, so governor drift,
host load and thermal state cannot line up with an arm. Every fold records its own AICLK trace,
board power and ASIC temperature: a forced fold that does not actually hold 1350 MHz is the check
that would catch a false win.
"""
from __future__ import annotations

import argparse, fcntl, hashlib, importlib.util, json, os, shutil, socket
import statistics as st, struct, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location(
    "_cell_now", REPO / "perf" / "b2z2_cell_recheck" / "cell_now.py")
CN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CN)
LEV, FIX = CN.LEV, CN.FIX

OUT: dict = {}
OUT_PATH: Path | None = None


class AiClkForce:
    """ARC `FORCE_AICLK` over tt-kmd's SMC message queue.

    The driver owns the queue and multiplexes it across every open fd, so this rides alongside
    whatever UMD is doing on the same chip instead of racing it. Layout is UMD's: message[0] is
    the type, message[1..7] the arguments. A response status of 0xFF means the firmware does not
    implement the message.
    """

    IOCTL = (0xFA << 8) | 17          # _IO(TENSTORRENT_IOCTL_MAGIC, 17)
    POST, POLL = 1 << 0, 1 << 1
    FORCE_AICLK = 0x33

    def __init__(self, node: int):
        self.fd = os.open(f"/dev/tenstorrent/{node}", os.O_RDWR | os.O_APPEND)
        self.mhz = 0

    def _smc(self, msg_type: int, *args: int) -> tuple:
        msg = [msg_type] + list(args) + [0] * (7 - len(args))
        fcntl.ioctl(self.fd, self.IOCTL, struct.pack("=IIII8I", 48, self.POST, 0, 0, *msg))
        deadline = time.time() + 2.0
        while time.time() < deadline:
            buf = bytearray(struct.pack("=IIII8I", 48, self.POLL, 0, 0, *([0] * 8)))
            try:
                fcntl.ioctl(self.fd, self.IOCTL, buf, True)
            except OSError as e:
                if e.errno == 11:                 # EAGAIN: response not back yet
                    time.sleep(0.002)
                    continue
                raise
            resp = struct.unpack("=IIII8I", bytes(buf))[4:]
            return resp[0] & 0xFF, resp[0] >> 16
        raise TimeoutError(f"no ARC response to message 0x{msg_type:02X}")

    def set(self, mhz: int) -> None:
        """Force the clock to `mhz`, or hand it back to the governor with 0."""
        status, _ = self._smc(self.FORCE_AICLK, mhz)
        assert status == 0, f"FORCE_AICLK({mhz}) rejected by firmware, status 0x{status:02X}"
        self.mhz = mhz

    def close(self) -> None:
        self.set(0)
        os.close(self.fd)


class Sampler(CN.ClockSampler):
    """cell_now's 5 Hz AICLK probe, plus the numbers this campaign is actually judged on."""

    def __init__(self, card: str, node: int):
        super().__init__(card)
        self._temp = next(Path(f"/sys/class/tenstorrent/tenstorrent!{node}/device/hwmon")
                          .glob("hwmon*/temp1_input"))
        self.temps: list = []

    def run(self) -> None:
        import threading
        threading.Thread(target=self._temp_loop, daemon=True).start()
        super().run()

    def _temp_loop(self) -> None:
        while not self.stop.wait(0.2):
            try:
                self.temps.append(int(self._temp.read_text()) / 1e3)
            except (OSError, ValueError):
                pass

    def take(self) -> dict:
        a, t = self.aiclk[:], self.temps[:]
        self.temps.clear()
        d = super().take()
        if a:
            s = sorted(a)
            d["aiclk_min"] = s[0]
            d["aiclk_p10"] = s[int(0.1 * (len(s) - 1))]
            d["aiclk_frac_burst"] = round(sum(1 for x in a if x >= 1300) / len(a), 3)
        if t:
            d["temp_c_max"] = round(max(t), 1)
        return d


def holder_nodes(pid: int) -> set:
    nodes = set()
    try:
        fds = list(Path(f"/proc/{pid}/fd").iterdir())
    except OSError:
        return nodes
    for fd in fds:
        try:
            t = os.readlink(fd)
        except OSError:
            continue
        if t.startswith("/dev/tenstorrent/"):
            nodes.add(int(t.rsplit("/", 1)[1]))
    return nodes


def own_device_node() -> int:
    """Which /dev/tenstorrent/N ttnn actually opened.

    TT_VISIBLE_DEVICES is a UMD logical id and is not the device node; the force has to be applied
    to the chip that is folding, so read it off our own fd table instead of assuming they match.
    """
    nodes = holder_nodes(os.getpid())
    assert len(nodes) == 1, f"expected exactly one open Tenstorrent chip, got {sorted(nodes)}"
    return nodes.pop()


def board_of(node: int) -> str:
    return Path(f"/sys/class/tenstorrent/tenstorrent!{node}/tt_serial").read_text().strip()


def dump() -> None:
    if OUT_PATH:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(OUT, indent=1))


def arm_stats(runs: list, arm: str) -> dict:
    r = [x for x in runs if not x["warmup"] and x["arm"] == arm]
    t = sorted(x["fold_s"] for x in r)
    pick = lambda k: [x[k] for x in r if x.get(k) is not None]
    clk, pw, burst = pick("aiclk_mean"), pick("power_w_mean"), pick("aiclk_frac_burst")
    return {
        "n": len(t), "fold_s_sorted": t,
        "median_fold_s": round(st.median(t), 3), "mean_fold_s": round(st.fmean(t), 3),
        "min_fold_s": t[0], "max_fold_s": t[-1],
        "aiclk_mean": round(st.fmean(clk), 1) if clk else None,
        "aiclk_min": min(pick("aiclk_min"), default=None),
        "aiclk_frac_burst": round(st.fmean(burst), 3) if burst else None,
        "power_w_mean": round(st.fmean(pw), 1) if pw else None,
        "temp_c_max": max(pick("temp_c_max"), default=None),
        "digests": sorted({x["cif_sha256"][:16] for x in r}),
        "plddt": sorted({round(x["plddt"], 6) for x in r}),
    }


def main() -> int:
    global OUT_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=10, help="timed folds PER ARM")
    ap.add_argument("--warm", type=int, default=1, help="discarded folds before timing")
    ap.add_argument("--mhz", type=int, default=1350, help="forced AICLK target for the on arm")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--recycles", type=int, default=3)
    ap.add_argument("--fixture", default="cdk2x2_512")
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

    dev = get_device()
    node = own_device_node()
    board = board_of(node)
    hwmon = Path(f"/sys/class/tenstorrent/tenstorrent!{node}/device/hwmon")

    OUT["env"] = {
        "host": socket.gethostname(), "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "device_node": node, "board_serial": board, "fixture": args.fixture,
        "forced_mhz": args.mhz, "tt_bio_file": _TB.__file__,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": os.getloadavg(), "steps": args.steps, "recycles": args.recycles,
        "commit": os.popen(f"git -C {REPO} rev-parse HEAD").read().strip(),
        "ttnn": getattr(ttnn, "__version__", "?"),
        "kmd": Path("/sys/module/tenstorrent/version").read_text().strip(),
        "fw_bundle": Path(f"/sys/class/tenstorrent/tenstorrent!{node}/tt_fw_bundle_ver").read_text().strip(),
        "power1_max_w": int(next(hwmon.glob("hwmon*/power1_max")).read_text()) / 1e6,
        # A holder on the other chip of our own board shares our power and thermal envelope. The
        # arms alternate fold by fold, so it cannot line up with one of them, but it is recorded
        # per fold and counted in the summary rather than assumed away.
        "same_board_holders_at_start": [
            h for h in CN.tt_holders(os.getpid())
            if any(board_of(n) == board for n in holder_nodes(h["pid"]))],
    }
    OUT["flags"] = CN.flag_snapshot()
    dump()

    work = Path(tempfile.mkdtemp(prefix="b2z2-aiclkforce-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    LEV._seed_msa(FIX / f"{args.fixture}.yaml", (FIX / f"{args.fixture}.a3m").read_text(), msa_dir)
    cfg = LEV.build_cfg(msa_dir, struct_dir)
    LEV._ensure_local_artifacts = _ensure_local_artifacts
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z2-aiclk-burst-pin", cfg)

    force = AiClkForce(node)
    clk = Sampler(f"tenstorrent!{node}", node)
    clk.start()

    def fold(i: int, warm: bool, arm: str) -> dict:
        force.set(args.mhz if arm == "force" else 0)
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        clk.take()                      # drop whatever was sampled between folds
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(FIX / f"{args.fixture}.yaml", cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        return {"i": i, "warmup": warm, "arm": arm, "fold_s": round(wall, 3),
                "foreign_tt": CN.tt_holders(os.getpid()),
                "plddt": metrics.get("complex_plddt", metrics.get("plddt")),
                "cif_sha256": hashlib.sha256(cifs[0].read_bytes()).hexdigest(),
                "loadavg1": round(os.getloadavg()[0], 2), **clk.take()}

    runs = []
    try:
        for i in range(args.warm + 2 * args.reps):
            warm = i < args.warm
            arm = "force" if (i - args.warm) % 2 else "base"
            r = fold(i, warm, arm)
            runs.append(r)
            print("  {:<5s} {:<3d} {:7.3f}s aiclk={} min={} burst={} {}W {}C plddt={} cif {}".format(
                "warm" if warm else arm, i, r["fold_s"], r.get("aiclk_mean"), r.get("aiclk_min"),
                r.get("aiclk_frac_burst"), r.get("power_w_mean"), r.get("temp_c_max"),
                round(r["plddt"], 6), r["cif_sha256"][:12]), flush=True)
            OUT["runs"] = runs
            dump()
    finally:
        clk.stop.set()
        force.close()

    base, forced = arm_stats(runs, "base"), arm_stats(runs, "force")
    OUT["base"], OUT["force"] = base, forced
    OUT["ratio_median"] = round(base["median_fold_s"] / forced["median_fold_s"], 4)
    OUT["ratio_mean"] = round(base["mean_fold_s"] / forced["mean_fold_s"], 4)
    OUT["aiclk_delta_mhz"] = round((forced["aiclk_mean"] or 0) - (base["aiclk_mean"] or 0), 1)
    OUT["power_delta_w"] = round((forced["power_w_mean"] or 0) - (base["power_w_mean"] or 0), 1)
    OUT["one_digest_across_arms"] = len(set(base["digests"]) | set(forced["digests"])) == 1
    OUT["base_arm_self_bit_exact"] = len(base["digests"]) == 1
    OUT["cotenanted_folds"] = sum(1 for r in runs if not r["warmup"] and r["foreign_tt"])
    dump()

    row = ("{:<5s} {n:2d} folds median {median_fold_s}s  aiclk {aiclk_mean} (min {aiclk_min})  "
           "burst {aiclk_frac_burst}  {power_w_mean}W  {temp_c_max}C")
    print("\n" + row.format("base", **base), flush=True)
    print(row.format("force", **forced), flush=True)
    print("ratio base/force = {} (mean {})  aiclk +{} MHz  power +{} W  "
          "one digest across arms {}  base arm self-bit-exact {}".format(
              OUT["ratio_median"], OUT["ratio_mean"], OUT["aiclk_delta_mhz"],
              OUT["power_delta_w"], OUT["one_digest_across_arms"],
              OUT["base_arm_self_bit_exact"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
