#!/usr/bin/env python3
"""When each of AdamW's host buffers is first touched, and what each one costs resident.

`of3t-restep` measured `AdamW.__init__` at **+6.00 GiB** on a crop-384 exactness-ON step and
attributed it to "three host fp32 numpy copies per weight -- master, exp_avg, exp_avg_sq".
`tt_bio/train/optim.py` allocates **five** dicts, not three, and two of them are not moments.
This measures each one, separately, at the step's real parameter census.

CARD-FREE BY CONSTRUCTION. `tensors.to_host` returns a numpy array unchanged
(`tt_bio/train/tensors.py:28-29`), so the real `AdamW.__init__` runs to completion over
host-backed parameters with no ttnn import and no device open. Nothing here touches a card.

The census is the step's own: **3152 parameters, 381,302,188 elements**, read off
`perf/of3t_restep/out/step_exact_on_384.json` rather than guessed. `--scale` shrinks the
element count for a box that cannot hold the full set; every per-element figure is scale-free
and the scale is recorded in the artifact.

Two instruments, because one of them can lie:

  * **construct**: RSS delta across the real `AdamW(params)` call. This is what `of3t-restep`
    measured and it is a TOTAL -- it cannot attribute.
  * **drop**: RSS delta from deleting one dict at a time and collecting. numpy hands a buffer
    over ~128 KB to mmap and frees it with munmap, so a resident dict shows up here at its
    full size and a dict that was never faulted in shows up at ~0. That is the whole question
    for `np.zeros_like`, which is `empty_like` + `copyto(0)` and therefore writes every page.

`avail_at_start_gib` is recorded because pc's available memory swings with its agent
population and a peak quoted without it is not comparable.
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

# `of3t-restep`'s sampler, imported rather than re-written.
from perf.of3t_restep.rssprofile import (  # noqa: E402
    GIB, _mem_available_bytes, _rss_bytes,
)

# The crop-384 step's own census (`perf/of3t_restep/out/step_exact_on_384.json`).
PARAMS = 3152
ELEMENTS = 381_302_188

# Every dict `AdamW.__init__` builds, in the order it builds them, with the line that does it
# and the only thing that reads it. `optim.py` line numbers at the commit stamped in the
# artifact.
BUFFERS = ["master", "init_master", "init_device", "init", "exp_avg", "exp_avg_sq"]


_LIBC = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")


def _trim() -> None:
    """Hand glibc's freed arenas back to the kernel before reading RSS.

    Without it the drop instrument reads ~0 on a realistic census: OF3's 3152 tensors have a
    MEDIAN of a few thousand elements, well under numpy's 128 KB mmap threshold, so most of
    the buffer set is heap and `free()` alone leaves it in the arena and in RSS.
    """
    gc.collect()
    try:
        _LIBC.malloc_trim(ctypes.c_size_t(0))
    except AttributeError:  # not glibc
        pass


def load_adamw(rev: str | None, work: Path, path_override: str | None = None):
    """`AdamW` from the working tree, or from `tt_bio/train/optim.py` as it stands at `rev`.

    One arm per process: measuring both in one would read the first arm's freed heap as the
    second one's floor. `rev` is stamped into the artifact next to the working tree's commit,
    so the pair is an A/B with a named before.
    """
    if not rev and not path_override:
        from tt_bio.train.optim import AdamW
        return AdamW, "working tree"
    work.mkdir(parents=True, exist_ok=True)
    if path_override:
        # A box with the sources but no git checkout -- the full-census arm runs on qb2's
        # CPU from an rsynced tree, and the before-arm's optim.py travels with it.
        path = Path(path_override)
    else:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{rev}:tt_bio/train/optim.py"],
                             capture_output=True, text=True, check=True).stdout
        path = work / f"optim_at_{rev.replace('/', '_')}.py"
        path.write_text(src)
    import importlib.util
    name = "tt_bio.train._optim_arm"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if path_override:
        return mod.AdamW, f"file:{path}"
    return mod.AdamW, subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", rev],
        capture_output=True, text=True, check=True).stdout.strip()


class HostTensor:
    """The two attributes `AdamW` uses of a parameter. No device, no tape."""

    __slots__ = ("value", "grad")

    def __init__(self, value):
        self.value = value
        self.grad = None


def _bf16(a: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 precision, in place, round-to-nearest-even.

    Keeps the synthetic weights exactly representable in the dtype the device copy uses, so a
    master built from them is the same kind of array the real one is.
    """
    u = a.view(np.uint32)
    u += 0x7FFF + ((u >> 16) & 1)
    u &= 0xFFFF0000
    return a


def build_params(n_params: int, elements: int, seed: int = 0):
    """`n_params` host tensors holding `elements` float32 values in total.

    Sizes follow a lognormal spread rather than a uniform one: OF3's census is 3152 tensors
    over 381.3 M elements, a mean of 121 k, and the real spread straddles numpy's 128 KB mmap
    threshold in both directions. A uniform split would put every array on the same side of it
    and make the drop instrument read cleaner than it is.
    """
    rng = np.random.default_rng(seed)
    w = rng.lognormal(0.0, 1.6, n_params)
    counts = np.maximum(1, np.floor(w / w.sum() * elements)).astype(np.int64)
    counts[-1] += elements - int(counts.sum())
    if counts[-1] < 1:  # the tail absorbed too much; give it back proportionally
        counts[-1] = 1
    params = {}
    for i, c in enumerate(counts):
        a = rng.standard_normal(int(c), dtype=np.float32)
        params[f"w{i}"] = HostTensor(_bf16(a))
    return params, int(counts.sum()), int(counts.max()), int(np.median(counts))


def measure(scale: float, seed: int, jsonl, AdamW, rev_label: str):
    n_params = PARAMS
    elements = max(n_params, int(round(ELEMENTS * scale)))
    rec = {"scale": scale, "params": n_params, "elements_requested": elements,
           "optim_from": rev_label}

    def rss():
        return _rss_bytes()

    def mark(tag, before):
        d = rss() - before
        row = {"phase": tag, "delta_gib": round(d / GIB, 4),
               "rss_gib": round(rss() / GIB, 4),
               "avail_gib": round(_mem_available_bytes() / GIB, 4)}
        jsonl.write(json.dumps(row) + "\n")
        jsonl.flush()
        return d

    _trim()
    b0 = rss()
    rec["rss_baseline_gib"] = round(b0 / GIB, 4)
    params, got, biggest, med = build_params(n_params, elements, seed)
    rec["elements_built"] = got
    rec["largest_param_elements"] = biggest
    rec["median_param_elements"] = med
    rec["params_bytes_gib"] = round(got * 4 / GIB, 4)
    rec["build_params_gib"] = round(mark("build_params", b0) / GIB, 4)

    _trim()
    b1 = rss()
    t0 = time.perf_counter()
    opt = AdamW(params, lr=3e-4)
    rec["construct_s"] = round(time.perf_counter() - t0, 3)
    rec["construct_gib"] = round(mark("AdamW.__init__", b1) / GIB, 4)

    # What the FIRST STEP will allocate that construction did not. On the deferred arm the
    # moments appear here; on the eager arm they were already resident and this reads ~0.
    _trim()
    b2 = rss()
    for n in opt.master:
        opt.exp_avg[n]
        opt.exp_avg_sq[n]
    rec["first_step_moments_gib"] = round(mark("first_step_moments", b2) / GIB, 4)

    # Per-buffer attribution: drop one dict, collect, read the RSS back.
    # REVERSE allocation order. `malloc_trim` returns free pages from the top of the heap
    # down, so freeing the oldest buffer first leaves a hole under everything newer and reads
    # ~0 while the bytes are genuinely gone. Newest-first attributes each one cleanly; the
    # construct and first-step deltas above are the figures that do not depend on this at all.
    present = [b for b in reversed(BUFFERS) if hasattr(opt, b)]
    rec["buffers_present"] = present
    rec["buffers_absent"] = [b for b in BUFFERS if b not in present]
    rec["drop_order"] = "newest allocation first"
    per = {}
    for name in present:
        _trim()
        b = rss()
        setattr(opt, name, {})
        _trim()
        per[name] = round((b - rss()) / GIB, 4)
        jsonl.write(json.dumps({"phase": f"drop:{name}", "freed_gib": per[name],
                                "rss_gib": round(rss() / GIB, 4)}) + "\n")
        jsonl.flush()
    rec["drop_gib"] = per
    rec["drop_total_gib"] = round(sum(per.values()), 4)
    del opt
    _trim()
    rec["rss_after_drop_gib"] = round(rss() / GIB, 4)

    # Scale-free: bytes per element per buffer, and what the full census would cost.
    e = float(got)
    rec["bytes_per_element"] = {k: round(v * GIB / e, 3) for k, v in per.items()}
    rec["full_census_gib"] = {k: round(v * GIB / e * ELEMENTS / GIB, 3) for k, v in per.items()}
    rec["full_census_total_gib"] = round(sum(rec["full_census_gib"].values()), 3)
    rec["construct_full_census_gib"] = round(
        rec["construct_gib"] * GIB / e * ELEMENTS / GIB, 3)
    rec["first_step_moments_full_census_gib"] = round(
        rec["first_step_moments_gib"] * GIB / e * ELEMENTS / GIB, 3)
    rec["resident_after_first_step_full_census_gib"] = round(
        rec["construct_full_census_gib"] + rec["first_step_moments_full_census_gib"], 3)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="fraction of the crop-384 step's 381,302,188 elements")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optim-rev", default=None,
                    help="git rev to take tt_bio/train/optim.py from; default working tree")
    ap.add_argument("--optim-file", default=None,
                    help="path to an optim.py to load instead; for a box with no git tree")
    ap.add_argument("--work", default="/tmp/of3t/of3t-optorder")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "out"))
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"s{a.scale:g}"
    jsonl_path = out_dir / f"optmem_{tag}.jsonl"
    out_path = out_dir / f"optmem_{tag}.json"

    AdamW, rev_label = load_adamw(a.optim_rev, Path(a.work), a.optim_file)
    head = os.popen(f"git -C {REPO} rev-parse HEAD").read().strip()
    branch = os.popen(f"git -C {REPO} rev-parse --abbrev-ref HEAD").read().strip()
    out = {"doc": __doc__.split("\n\n")[0], "argv": sys.argv[1:], "env": {
        "host": socket.gethostname(),
        "commit": head, "branch": branch,
        "optim_rev": a.optim_rev, "optim_from": rev_label,
        "optim_dirty": bool(os.popen(
            f"git -C {REPO} status --porcelain tt_bio/train/optim.py").read().strip()),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg_start": os.getloadavg(),
        "mem_total_gib": round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / GIB, 3),
        "avail_at_start_gib": round(_mem_available_bytes() / GIB, 3),
        "card_opened": False,
        "ttnn_imported": "ttnn" in sys.modules}}

    with open(jsonl_path, "w", buffering=1) as jsonl:
        out["run"] = measure(a.scale, a.seed, jsonl, AdamW, rev_label)

    out["env"]["ttnn_imported_after"] = "ttnn" in sys.modules
    out["env"]["avail_at_end_gib"] = round(_mem_available_bytes() / GIB, 3)
    out_path.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out["run"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
