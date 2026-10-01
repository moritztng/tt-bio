#!/usr/bin/env python3
"""One rung of the BindCraft 2 size ladder: a real target, a real binder length, one card.

`perf/bcx_p10_campaign/campaign_run.py` runs a whole campaign and always on PD-L1. This
runs the same user-facing entry -- `bindcraft2.run_campaign`, every design stage, the
compile lock live -- on any target, and can stop after K gradient rounds so a rung can be
declared healthy without paying for a campaign at every size.

What it records per rung: the token axis, the device resident peak and what was free at it,
seconds per gradient round with the AICLK sampled DURING the round, what the shipped
auto-interleave default chose, and, when the size does not fit, the verbatim text the user
is shown. That last one is the point of the large rungs: a clean refusal that names the
size is a supported outcome and a traceback is not.

The footprint leg is a SEPARATE run from the timed leg, and that is not fastidiousness:
`tenstorrent.dram_peak` warns in its own docstring that `get_memory_view` drains the
pipeline (12.0 s -> 28.8 s on a 117 aa fold), so a run with `--footprint` reports no
timing and says so in `timing_valid`.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "perf" / "bcx_round"))
sys.path.insert(0, str(ROOT / "perf" / "bcx_predictor"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import meter as M                                                       # noqa: E402
import bc2_state as B                                                   # noqa: E402
from tt_bio import bindcraft2, duotraj                                  # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings   # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings               # noqa: E402

TARGETS = json.loads((HERE / "targets.json").read_text())


def resolve(path: str) -> str:
    """A manifest path, either shipped with BindCraft 2 or checked in beside this file."""
    if path.startswith("BC2:"):
        return os.path.join(B.BC2, path[4:])
    return str(ROOT / path)


def _safe(fn):
    """An instrument never kills a rung: the rung is the measurement, this is the label."""
    try:
        return fn()
    except BaseException as exc:
        return {"error": repr(exc)}


def host_memory() -> dict:
    out = {}
    for line in open("/proc/self/status"):
        if line.startswith(("VmRSS:", "VmHWM:")):
            out[line.split(":")[0].lower()] = int(line.split()[1]) * 1024
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            out["mem_available"] = int(line.split()[1]) * 1024
    return out


class Rounds:
    """A record per gradient round, written to disk as each one lands.

    On disk at every round because the large rungs are expected to die, and a rung that is
    OOM-killed still has to say how far it got and on how much memory.
    """

    def __init__(self, path, dram_total):
        self.path, self.rows, self.dram_total = path, [], dram_total
        self.free_min = None
        import threading
        self.lock = threading.Lock()

    def mark(self, slot):
        free = duotraj.free_device_bytes()
        with self.lock:
            if free and (self.free_min is None or free < self.free_min):
                self.free_min = free
            self.rows.append({"t": time.time(), "slot": slot,
                              "round": sum(1 for r in self.rows if r["slot"] == slot) + 1,
                              "device_free": free, "load1": round(os.getloadavg()[0], 2),
                              # Cumulative, so consecutive boundaries subtract to a per-round
                              # reach. `declined` rising is the fused triangle attention
                              # falling through to the composed path, which is a footprint
                              # and a speed event and shows up in neither as an error.
                              "triatt": M.reach(), **host_memory()})
            self.dump()

    def dump(self):
        tmp = f"{self.path}.partial"
        pathlib.Path(tmp).write_text(json.dumps(self.rows))
        os.replace(tmp, self.path)

    def per_round(self, clock):
        """Wall seconds per round per slot, each with the clock sampled inside that round.

        The last round of a run has no successor boundary to close it, so it is dropped
        rather than closed against the run's end: a round whose wall includes teardown is
        not a round time.
        """
        out = []
        for slot in sorted({r["slot"] for r in self.rows}):
            rs = [r for r in self.rows if r["slot"] == slot]
            for a, b in zip(rs, rs[1:]):
                out.append({"slot": slot, "round": a["round"],
                            "seconds": round(b["t"] - a["t"], 3),
                            **clock.window(a["t"], b["t"])})
        return out


class DevicePeak:
    """The device high-water mark, sampled at tape nodes rather than at round boundaries.

    `tenstorrent.dram_peak` only fires where tt-bio's own code is tagged, and the AF2 design
    backward runs through `autograd`'s tape, which carries no tags: at 192 tokens it reported
    a 1.941 GiB peak off a forward tag while the backward is where the footprint is made. So
    this samples where `bcx-bigtarget`'s census sampled, at every Nth `_retire` -- the point
    the backward walk releases an intermediate, which every tape node passes through.

    Every Nth and not every one because `get_memory_view` drains the pipeline. This leg
    reports no timing for exactly that reason.
    """

    def __init__(self, every: int):
        self.every, self.calls, self.samples = every, 0, 0
        self.used_max = 0
        self.free_min = None
        self.largest_free_at_peak = None
        self.total = 0

    def sample(self) -> None:
        import ttnn
        from tt_bio import tenstorrent
        if tenstorrent._device is None:
            return
        mv = ttnn.get_memory_view(tenstorrent._device, ttnn.BufferType.DRAM)
        banks = int(mv.num_banks)
        self.total = int(mv.total_bytes_per_bank) * banks
        free = int(mv.total_bytes_free_per_bank) * banks
        used = self.total - free
        self.samples += 1
        if self.free_min is None or free < self.free_min:
            self.free_min = free
        if used > self.used_max:
            self.used_max = used
            lcf = mv.largest_contiguous_bytes_free_per_bank
            if isinstance(lcf, (list, tuple)):
                lcf = min(lcf)
            self.largest_free_at_peak = int(lcf)

    def install(self) -> None:
        """Wrap `autograd._retire`. A module-level rebind is enough: the backward walk looks
        it up in the module globals, which is also why the tree stays clean -- nothing on
        disk changes, so the timed leg sharing this worktree measures an unpatched tree."""
        from tt_bio import autograd
        real = autograd._retire

        def retire(t):
            self.calls += 1
            if self.calls % self.every == 0:
                try:
                    self.sample()
                except Exception:
                    pass
            return real(t)
        autograd._retire = retire

    def report(self) -> dict:
        """Both units, spelled out. `bcx-bigtarget`'s curve is in DECIMAL GB -- it calls the
        card 34.226 GB, which is 34.226e9 bytes and 31.875 GiB -- so a GiB figure compared
        against it reads 7 % light. Reporting `_gb` as decimal and `_gib` as binary beside
        it makes a rung comparable with that curve without anyone doing the conversion in
        their head and getting it wrong."""
        if not self.samples:
            return {"node_probe_samples": 0}
        free_at_peak = self.total - self.used_max
        return {"node_probe_samples": self.samples, "node_probe_calls": self.calls,
                "resident_peak_gb": round(self.used_max / 1e9, 4),
                "resident_peak_gib": round(self.used_max / 2**30, 4),
                "free_at_peak_gb": round(free_at_peak / 1e9, 4),
                "free_at_peak_gib": round(free_at_peak / 2**30, 4),
                "device_total_gb": round(self.total / 1e9, 4),
                "largest_free_block_at_peak_mb": None if self.largest_free_at_peak is None
                else round(self.largest_free_at_peak / 2**20, 1)}


def peak_from_probe(path: pathlib.Path) -> dict:
    """The device high-water mark, read out of `tenstorrent.dram_peak`'s own census file."""
    if not path.exists():
        return {}
    best, line = 0.0, ""
    for ln in path.read_text().splitlines():
        if "GiB used" not in ln:
            continue
        try:
            used = float(ln.split(":")[1].strip().split()[0])
        except (IndexError, ValueError):
            continue
        if used > best:
            best, line = used, ln.strip()
    return {"tagged_peak_gib": best, "tagged_peak_line": line} if best else {}


AXIS: dict = {}


def observe_axis():
    """Record the token axis the Evoformer SEAM actually gets, not the one arithmetic predicts.

    `tokens` above is `_pad32(target_residues + binder)` off `targets.json`, and for a
    single-chain target that is exact. It is NOT exact for a multi-chain one: the fused-arm
    note printed 544 on a rung this harness labelled 512 (hIL2R, 2 chains) and 608 on one it
    labelled 576 (hTNFa, 3 chains), both one bucket high, while hPDL1 on one chain matched.
    BindCraft 2 hands the splice whatever complex it built, `_pad` buckets THAT, and the
    difference is real tokens -- so a table keyed on the arithmetic mislabels exactly the rungs
    that carry a boundary.

    `_pad` is a staticmethod taking no self, so the wrap is a plain function and the class keeps
    working for a concurrent leg in another process.
    """
    inner = bindcraft2.EvoformerOnDevice._pad

    def pad(m, z, mask, pair_mask):
        out = inner(m, z, mask, pair_mask)
        n = int(out[4])
        AXIS.setdefault("complex_residues", n)
        AXIS.setdefault("evoformer_axis", int(bindcraft2._pad32(n)))
        AXIS["axis_seen"] = sorted(set(AXIS.get("axis_seen", [])) |
                                   {int(bindcraft2._pad32(n))})
        return out

    bindcraft2.EvoformerOnDevice._pad = staticmethod(pad)
    return AXIS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=sorted(TARGETS))
    ap.add_argument("--binder", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=0,
                    help="stop after this many gradient rounds; 0 runs the whole campaign")
    ap.add_argument("--trajectories", default="auto",
                    help='a count, or "auto" to leave the argument off and take the shipped '
                         "default's own resolution")
    ap.add_argument("--max-trajectories", dest="max_trajectories", type=int, default=2)
    ap.add_argument("--final-designs", dest="final_designs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--validation", default="device", choices=("jax", "device"))
    ap.add_argument("--footprint", action="store_true",
                    help="arm tenstorrent.dram_peak. Reports no timing: the probe drains the "
                         "pipeline and a run with it on measures the probe")
    ap.add_argument("--probe-every", dest="probe_every", type=int, default=8,
                    help="sample device DRAM at every Nth tape node, with --footprint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--memory", default="auto",
                    help="bindcraft2.predictor(memory=...): auto, fast, lean or offload")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="K=V")
    args = ap.parse_args()

    spec = TARGETS[args.target]
    tokens = bindcraft2._pad32(spec["residues"] + args.binder)
    axis = observe_axis()
    project = args.out
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)

    probe = pathlib.Path(project) / "dram_peak.txt"
    peak = DevicePeak(max(1, args.probe_every))
    if args.footprint:
        os.environ["TT_BIO_DRAM_PEAK"] = str(probe)
        peak.install()

    # The settings a user would write: the shipped PD-L1 campaign with its target swapped.
    # Inline `targets` rather than a `settings/target/<name>.json`, so nothing is written
    # into the BindCraft 2 install and the rung is reproducible from this repo alone.
    request = {"modality": "binder", "campaign_name": f"{args.target}_{args.binder}",
               "number_of_final_designs": args.final_designs,
               "targets": [{"name": args.target, "target_path": resolve(spec["path"]),
                            "chains": spec["chains"], "hotspots": spec["hotspots"]}]}
    req_path = pathlib.Path(project) / "settings_request.json"
    req_path.write_text(json.dumps(request, indent=1))

    budget = args.max_trajectories
    overrides = [f"campaign_seed={args.seed}", f"max_trajectories={budget}",
                 f"project_folder={project}",
                 f"binder_lengths=[{','.join([str(args.binder)] * budget)}]",
                 "compile_next_length=0"] + args.sets
    settings = cleaned_campaign_settings(
        read_settings(req_path, parse_setting_overrides(overrides)))

    M.CLOCK = M.Clock(1.0)
    M.CLOCK.start()

    from tt_bio import tenstorrent
    rounds = Rounds(os.path.join(project, "rounds.json"), 0)

    n = None if args.trajectories == "auto" else int(args.trajectories)

    # What the shipped default would pick at THIS rung, asked the way `run_campaign` asks it.
    # Twice, because the two answers are different numbers with different failure modes: before
    # a card is open `auto_trajectories` prices the card from a model, and with one open it reads
    # `free_device_bytes()` and is self-correcting (`state/bwx-perf.md`). A ladder that records
    # only the second cannot see a pre-open default that errs high.
    design_axis = bindcraft2.design_tokens(settings)

    def auto_now() -> dict:
        try:
            count, why = duotraj.auto_trajectories(design_axis)
            return {"count": count, "why": why, "tokens_asked": design_axis}
        except BaseException as exc:                                   # noqa: BLE001
            return {"error": repr(exc), "tokens_asked": design_axis}

    def auto_would_choose():
        a = auto_now()
        return None if "error" in a else [a["count"], a["why"]]

    cls = bindcraft2.design_model_class()
    real_sequence_gradients = cls.sequence_gradients

    def sequence_gradients(self, *a, **kw):
        if not kw.get("compile_only"):
            rounds.mark(duotraj.slot())
            if args.rounds and len(rounds.rows) > args.rounds:
                raise M.StopAfterRounds(f"{args.rounds} rounds collected")
        return real_sequence_gradients(self, *a, **kw)
    cls.sequence_gradients = sequence_gradients

    stamp = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
             "target": args.target, "target_residues": spec["residues"],
             "target_chains": spec["n_chains"], "target_fold": spec["fold"],
             "binder": args.binder, "tokens": tokens,
             "rounds_requested": args.rounds, "footprint_probe": args.footprint,
             "timing_valid": not args.footprint,
             "trajectories_arg": args.trajectories,
             # `design_tokens` is BindCraft 2's own `design_residue_count`, which is what prices
             # the default. It is NOT necessarily the axis the Evoformer seam runs -- the complex
             # BC2 builds is larger than target + binder -- so both are recorded and `close`
             # compares them. Pricing a trajectory a bucket low is a default that errs high.
             "design_tokens": design_axis,
             "auto_before_open": auto_now(),
             # Same answer as `auto_before_open`, in the flat two-element shape four readers of
             # this JSON already parse (bgx_size/table.py, bgx_size/camp576_watch.sh,
             # bwx_scale/report.py, bcx_default/campaign_report.py). Priced on `design_axis` and
             # not on the arithmetic `tokens` for the reason above, and it inherits `auto_now`'s
             # guard, so a rung still records itself when the default raises.
             "auto_would_choose": auto_would_choose(),
             "max_trajectories": budget, "validation": args.validation, "seed": args.seed,
             "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
             "dirty": bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                          capture_output=True, text=True).stdout.strip()),
             "compile_cache": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
             "pci": M.CLOCK.pci, "sysfs": M.CLOCK.path, "nproc": os.cpu_count(),
             "loadavg_start": os.getloadavg(), "mem_start": host_memory(),
             "started_utc": time.strftime("%FT%TZ", time.gmtime())}
    print(json.dumps(stamp, indent=1), flush=True)

    t0 = time.time()
    trajectories = None

    def close(exc=None, tb=None):
        """Write the rung out BEFORE the predictor's teardown, which can abort the process
        on `close_device` and take the record with it."""
        M.CLOCK.stop()
        rounds.dump()
        stamp.update({"wall_seconds": round(time.time() - t0, 2),
                      "trajectories_returned": trajectories,
                      "error": exc, "traceback": tb,
                      "dram_total_bytes": tenstorrent._dram_total_bytes()
                      if tenstorrent._device is not None else None,
                      "complex_residues": axis.get("complex_residues"),
                      "evoformer_axis": axis.get("evoformer_axis"),
                      "axis_seen": axis.get("axis_seen"),
                      "axis_matches_arithmetic": (axis.get("evoformer_axis") == tokens
                                                  if axis.get("evoformer_axis") else None),
                      # The default prices a trajectory at `design_tokens`. Where the seam runs a
                      # larger axis than that, every auto count on this board was priced low.
                      "auto_priced_axis_short_by": (axis["evoformer_axis"] - design_axis
                                                    if axis.get("evoformer_axis") and design_axis
                                                    else None),
                      "device_free_min": rounds.free_min,
                      "rounds_done": len(rounds.rows),
                      "rounds_per_slot": {s: sum(1 for r in rounds.rows if r["slot"] == s)
                                          for s in sorted({r["slot"] for r in rounds.rows})},
                      "per_round": rounds.per_round(M.CLOCK),
                      # Which levers the ENGINE ran, read off the modules that own the flags,
                      # and each one's serve counter. A rung whose footprint jumps has to say
                      # whether a lever went inert rather than leave it to arithmetic.
                      "levers": _safe(M.levers), "lever_stats": _safe(M.lever_stats),
                      "triatt_reach": _safe(M.reach),
                      "aiclk_run": M.CLOCK.window(t0, time.time()),
                      **peak_from_probe(probe), **peak.report(),
                      "mem_end": host_memory(), "loadavg_end": os.getloadavg(),
                      "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
        pathlib.Path(os.path.join(project, "rung.json")).write_text(json.dumps(stamp, indent=1))
        print(json.dumps({k: v for k, v in stamp.items() if k != "per_round"}, indent=1),
              flush=True)

    try:
        with bindcraft2.campaign_predictor(trunk="device", validation=args.validation,
                                           checkpoints=args.params, extra_msa=True,
                                           template=True, exact=False,
                                           memory=args.memory) as build:
            stamp["fast"] = build.fast
            stamp["memory_requested"] = args.memory
            stamp["memory_used"] = build.memory.used
            stamp["auto_card_open"] = auto_now()
            per_card = {} if n is None else {"trajectories_per_card": n}
            trajectories = bindcraft2.run_campaign(
                settings, project, **per_card, af2_weights=args.params,
                mpnn_weights=os.path.join(B.BC2, "bindcraft", "weights", "proteinmpnn",
                                          "weights_neutral"))
    except M.StopAfterRounds as stop:
        close(f"StopAfterRounds({stop})")
        return 0
    except BaseException as exc:
        import traceback
        close(repr(exc), traceback.format_exc())
        raise
    close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
