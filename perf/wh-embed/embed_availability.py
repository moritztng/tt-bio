#!/usr/bin/env python3
"""Does this model load and embed on this chip at all, in this precision?

Lever 0's kill gate. esmc-6b is 6.29 B params = 12.6 GB in bf16 against a Wormhole chip's
~12 GB of DRAM, so the bf16 load is PREDICTED to OOM and the block-fp8 (--fast) load to
succeed at ~6.3 GB. If bf16 loads, the arithmetic is wrong and main.py:2390's comment is
wrong with it -- that is the kill gate, and it is why this reports the failure rather than
raising it.

Also reports the chip's actual DRAM budget from the device itself, so the 12 GB in that
comment stops being a number nobody measured.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

UBIQUITIN = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTL"
             "LHLVLRLRGG")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--n-seqs", type=int, default=2)
    ap.add_argument("--residues", type=int, default=76)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    res = dict(model=args.model, fast=args.fast,
               arch=os.environ.get("PROBE_ARCH", "unknown"),
               visible_devices=os.environ.get("TT_VISIBLE_DEVICES", ""),
               n_seqs=args.n_seqs, residues=args.residues,
               loaded=False, embedded=False, error=None, error_type=None)

    reps = (args.residues // len(UBIQUITIN)) + 1
    seq = (UBIQUITIN * reps)[:args.residues]
    sequences = {f"seq{i}": seq for i in range(args.n_seqs)}

    # Probe each accessor on its own: a MeshDevice does not expose the same set a Device
    # does (num_dram_channels is absent on the mesh handle), and one missing name must not
    # cost the others. None of this decides the availability answer.
    try:
        from tt_bio.tenstorrent import get_device
        dev = get_device()
        for name in ("arch", "dram_grid_size", "num_dram_channels",
                     "dram_size_per_channel", "compute_with_storage_grid_size"):
            try:
                res[name] = str(getattr(dev, name)())
            except Exception as exc:
                res[name] = f"unavailable: {type(exc).__name__}"
        try:
            chans = dev.dram_grid_size().x * dev.dram_grid_size().y
            res["dram_total_GB"] = round(chans * dev.dram_size_per_channel() / 1e9, 2)
        except Exception as exc:
            res["dram_total_GB"] = f"unavailable: {type(exc).__name__}"
    except Exception as exc:
        res["dram_probe_error"] = f"{type(exc).__name__}: {exc}"

    try:
        if args.model.startswith("saprot"):
            from tt_bio.saprot import embed_sequences, load_saprot
            t0 = time.perf_counter()
            model = load_saprot(args.model, fast=args.fast)
            res["load_s"] = round(time.perf_counter() - t0, 1)
            res["loaded"] = True
            payload = {k: (v, "#" * len(v)) for k, v in sequences.items()}
        else:
            from tt_bio.esmc import embed_sequences, load_esmc
            t0 = time.perf_counter()
            model = load_esmc(args.model, fast=args.fast)
            res["load_s"] = round(time.perf_counter() - t0, 1)
            res["loaded"] = True
            payload = sequences

        t0 = time.perf_counter()
        out = embed_sequences(model, payload, batch_size=1)
        res["embed_s"] = round(time.perf_counter() - t0, 2)
        res["embedded"] = True
        res["d_model"] = int(out[0].pooled.shape[0])
        res["per_residue_shape"] = list(out[0].per_residue.shape)
    except Exception as exc:
        res["error_type"] = type(exc).__name__
        res["error"] = str(exc)[:1200]
        res["traceback_tail"] = traceback.format_exc().strip().splitlines()[-4:]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
