#!/usr/bin/env python3
"""Per-step wall clock for the ABodyBuilder3 training step, on one card.

PLAN.md §6's schedule gate: above 3.0 s/step at batch 64 and 230 tokens the base recipe does not
fit its 16-161 chip-hour envelope. So this number decides a schedule, and two things about how it
is taken matter as much as the number.

**The host has to be quiet, and quiet has to be MEASURED, not grepped.** `tt_atom`'s `host_quiet()`
searched `ps` output for the substring `tt_bio`, which every fleet worker's own task prompt
contains, so it matched itself and returned "not quiet" forever -- three benchmarks burned their
full budget measuring nothing. This reads `/proc/<pid>/fd` and reports who actually holds a
`/dev/tenstorrent/*` node open, which a process merely mentioning the device cannot trip. Processes
whose fd directory is unreadable are counted and named rather than skipped: an unreadable pid is an
unknown, and a quiet check that silently drops unknowns is blind exactly where it matters.

**And the node has to be OUR node, which is not our card number.** Measured on qb1:
`TT_VISIBLE_DEVICES=3` opens `/dev/tenstorrent/0`. So a check that compares cotenants against the
logical card number reads a cotenant on "card 3" as a conflict when it is not, and misses the one
that is. This asks the kernel which node this very process opened, after the device is up, and
splits cotenants into same-node and elsewhere-on-host. Both matter and they are not the same thing:
same-node is contention for the chip, elsewhere is contention for the host and the power rail.

**The clock is sampled DURING the run.** On Blackhole the AICLK sets the fold time -- 800 MHz and
1350 MHz are a 1.5x spread on the same work -- so a number without its during-run clock is not a
measurement. A sampler thread reads it while the steps run and the min/median/max are reported.

What this measures and what it does not, stated so the number cannot be quoted wrong: the device
forward and backward for `batch` samples at `tokens` tokens, as `batch / micro` accumulation steps
of `micro` samples each, which is upstream's own arithmetic (`stages/train.py:64-73` builds the
loader at 8 and accumulates to 64). It does NOT include the host geometry tail (torsion angles to
frames, atom14), the four losses, or the optimizer, none of which exist yet. It is therefore a
LOWER bound on the step and an upper bound on nothing.

Run: TT_VISIBLE_DEVICES=<card> TT_BIO_LEASE_CARDS=<card> PYTHONPATH=$PWD python3 \
        scripts/abb3_port/step_time.py [--steps 100] [--micro 8] [--tokens 256]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import threading
import time

import torch
import ttnn

from tt_bio.abodybuilder3 import DeviceABB3, to_device_fp32
from tt_bio.abodybuilder3_reference import (ABB3Config, ABB3StructureModule,
                                            single_and_pair_features)
from tt_bio.tenstorrent import get_device
from tt_bio.train import abodybuilder3_grad as grad


def our_nodes() -> set[str]:
    """The device nodes THIS process has open. Empty before the device is opened."""
    nodes = set()
    try:
        for name in os.listdir(f"/proc/{os.getpid()}/fd"):
            try:
                target = os.readlink(f"/proc/{os.getpid()}/fd/{name}")
            except OSError:
                continue
            if target.startswith("/dev/tenstorrent/"):
                nodes.add(target)
    except OSError:
        pass
    return nodes


def device_holders() -> tuple[list[tuple[int, str, str]], list[int]]:
    """Every process holding a `/dev/tenstorrent/*` node open, and the pids we could not read."""
    holders, blind = [], []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        fd_dir = f"/proc/{entry}/fd"
        try:
            names = os.listdir(fd_dir)
        except PermissionError:
            blind.append(pid)
            continue
        except (FileNotFoundError, NotADirectoryError):
            continue
        for name in names:
            try:
                target = os.readlink(f"{fd_dir}/{name}")
            except OSError:
                continue
            if target.startswith("/dev/tenstorrent/"):
                try:
                    with open(f"/proc/{entry}/cmdline", "rb") as fh:
                        cmd = fh.read().replace(b"\0", b" ").decode(errors="replace").strip()
                except OSError:
                    cmd = "?"
                holders.append((pid, target, cmd[:90]))
                break
    return holders, blind


def report_host(tag: str) -> tuple[int, int]:
    """Print who holds what, split by whether it is our own node. Returns the two counts."""
    mine = our_nodes()
    holders, blind = device_holders()
    # tt-smi opens a device node to read telemetry, so the clock sampler this script runs would
    # otherwise appear as a cotenant of its own measurement.
    holders = [h for h in holders if "tt-smi" not in h[2]]
    same = [h for h in holders if h[0] != os.getpid() and h[1] in mine]
    other = [h for h in holders if h[0] != os.getpid() and h[1] not in mine]
    print(f"host [{tag}]: our node(s) {sorted(mine) or 'not open yet'}; "
          f"{len(same)} cotenant(s) on our node, {len(other)} elsewhere, "
          f"{len(blind)} pid(s) unreadable")
    for pid, node, cmd in same:
        print(f"  SAME NODE  pid {pid} -> {node}  {cmd}")
    for pid, node, cmd in other:
        print(f"  elsewhere  pid {pid} -> {node}  {cmd}")
    if blind:
        print(f"  unreadable pids: {blind[:12]}{' ...' if len(blind) > 12 else ''}")
    return len(same), len(other)


class ClockSampler(threading.Thread):
    """Read every card's AICLK while the run is in flight."""

    def __init__(self, period: float = 2.0):
        super().__init__(daemon=True)
        self.period = period
        self.samples: dict[int, list[int]] = {}
        self.failures = 0
        self._stop = threading.Event()

    def run(self):
        smi = os.path.expanduser("~/.local/bin/tt-smi")
        while not self._stop.is_set():
            try:
                raw = subprocess.run([smi, "-s"], capture_output=True, timeout=30).stdout
                # Regex and not `json.loads`: tt-smi writes its snapshot with a banner around the
                # JSON on some versions, so a strict parse throws and the sampler silently records
                # nothing -- which is what happened on the first run of this script, and a clock
                # line reading "no samples" is indistinguishable from a clock that was never
                # sampled. The field is hex, e.g. "AICLK": "0x546".
                found = re.findall(rb'"AICLK":\s*"0x([0-9a-fA-F]+)"', raw)
                for i, hexval in enumerate(found):
                    self.samples.setdefault(i, []).append(int(hexval, 16))
                if not found:
                    self.failures += 1
            except Exception:
                self.failures += 1
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()

    def summary(self) -> str:
        if not self.samples:
            return f"AICLK: NO SAMPLES ({self.failures} failed reads) -- this run carries no clock"
        parts = []
        for card, vals in sorted(self.samples.items()):
            parts.append(f"card {card} min {min(vals)} median {int(statistics.median(vals))} "
                         f"max {max(vals)} MHz over {len(vals)}")
        return ("AICLK during the run: " + "; ".join(parts)
                + (f"; {self.failures} failed reads" if self.failures else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    args = ap.parse_args()
    assert args.batch % args.micro == 0
    accum = args.batch // args.micro

    report_host("before opening the device")

    cfg = ABB3Config(use_plddt=True, no_blocks=args.blocks)
    torch.manual_seed(0)
    ref = ABB3StructureModule(cfg)
    with torch.no_grad():
        for p in ref.parameters():
            p.normal_(0.0, 0.05)
    g = torch.Generator().manual_seed(1)
    aatype = torch.randint(0, 21, (args.micro, args.tokens), generator=g)
    is_heavy = (torch.arange(args.tokens) < args.tokens // 2).expand(args.micro,
                                                                     args.tokens).clone()
    ri = torch.arange(args.tokens).expand(args.micro, args.tokens).clone()
    single, pair = single_and_pair_features(aatype, is_heavy, ri)
    mask = torch.ones(args.micro, args.tokens)

    dev = get_device()
    sampler = ClockSampler()
    try:
        grad.install()
        model = DeviceABB3(ref.state_dict(), cfg,
                           to_device=lambda x: grad.param(to_device_fp32(x)))
        square = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).unsqueeze(1)
        sq_dev = to_device_fp32(square)
        bias_dev = to_device_fp32(cfg.inf * (square - 1.0))
        sd_v, zd_v = to_device_fp32(single), to_device_fp32(pair)
        seed_states = to_device_fp32(torch.randn(args.micro, args.tokens, cfg.embed_dim,
                                                 generator=g))
        params = [p for p in _parameters(model)]
        print(f"taped parameters on device: {len(params)}")

        def one_micro_batch():
            for p in params:
                p.grad = None
            out = model(grad.param(sd_v), grad.param(zd_v), sq_dev, bias_dev)
            grad.backward([out["states"][-1]], [seed_states])

        for _ in range(args.warmup):
            one_micro_batch()
        ttnn.synchronize_device(dev)
        same, other = report_host("device open, before timing")
        sampler.start()

        step_times, micro_times = [], []
        for step in range(args.steps):
            t0 = time.perf_counter()
            for _ in range(accum):
                m0 = time.perf_counter()
                one_micro_batch()
                ttnn.synchronize_device(dev)
                micro_times.append(time.perf_counter() - m0)
            step_times.append(time.perf_counter() - t0)
            if step == 0 or (step + 1) % 10 == 0:
                print(f"  step {step + 1:>4}  {step_times[-1]:.3f} s"
                      f"  (median so far {statistics.median(step_times):.3f} s)")
        sampler.stop()
        med = statistics.median(step_times)
        print(f"\nSTEP: median {med:.3f} s over {len(step_times)} steps at batch {args.batch} "
              f"({accum} x {args.micro}) and {args.tokens} tokens, "
              f"micro-batch median {statistics.median(micro_times):.3f} s, "
              f"min step {min(step_times):.3f} s, max {max(step_times):.3f} s")
        print(f"CLOCK: {sampler.summary()}")
        print(f"EXCLUDES: host geometry tail, the four losses, the optimizer")
        if same or other:
            print(f"CONTENDED: {same} cotenant(s) on our node and {other} elsewhere on the host "
                  f"during timing, so this median is an UPPER bound and not a kill-line reading")
        return 0
    finally:
        sampler.stop()
        grad.uninstall()
        ttnn.close_device(dev)


def _parameters(model):
    """Every taped leaf the model holds, walked without a registry.

    The device module keeps its weights in plain attributes and lists rather than a module tree,
    which is deliberate -- there is no nn.Module here -- so the parameter set is discovered rather
    than declared. A registry would be another thing to keep in sync with the layout.
    """
    seen, out = set(), []
    def walk(obj, depth=0):
        if depth > 4 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, grad.Tensor):
            out.append(obj)
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                walk(x, depth + 1)
        elif hasattr(obj, "__dict__"):
            for x in vars(obj).values():
                walk(x, depth + 1)
    walk(model)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
