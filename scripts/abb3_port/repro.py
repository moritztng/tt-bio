#!/usr/bin/env python3
"""One rank of the ABodyBuilder3 reproduction run. Opens exactly one card.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:train-b3-train \
    PYTHONPATH=$PWD python3 scripts/abb3_port/repro.py --out runs/base --steps 193512

Data parallelism is one process per chip, launched by ``scripts/abb3_port/supervise.py``, and
the reason is not a preference: a ttnn process that can see four chips brings all four up, so
each replica must pin its own card and therefore cannot share a device context with the others.
``--rank``/``--world``/``--chips`` say which replica this is; the gradient exchange is on the
host over ``--rendezvous``.

``--kill-at N`` makes the process kill ITSELF with SIGKILL after step N, which is how the resume
is demonstrated instead of asserted. SIGKILL and not an exception on purpose: a watchdog reset
gives the process nothing, so a resume proven against a clean shutdown has not been proven
against the thing that will actually happen 9 to 47 times.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

import torch
import ttnn

from tt_bio.abodybuilder3_reference import ABB3Config, ABB3StructureModule  # noqa: E402
from tt_bio.tenstorrent import get_device  # noqa: E402
from tt_bio.train.abb3_dataset import dataset as make_dataset  # noqa: E402
from tt_bio.train.abb3_init import initialise_  # noqa: E402
from tt_bio.train.abb3_run import RunConfig, run  # noqa: E402
from tt_bio.train.abodybuilder3_step import TrainStep  # noqa: E402


def initial_state_dict(cfg: ABB3Config, seed: int) -> dict:
    """The starting weights, identical on every rank from one integer.

    Every rank building the model from the same seed is what makes the step-1 master hashes
    agree. Broadcasting rank 0's weights would work too and would be one more thing that can
    silently not happen; a seed cannot half-arrive.

    `initialise_` is what draws them. Constructing the module is not enough and used not to be
    known to be: `af2_reference.Linear` allocates zeros because every inference path loads a
    state dict over the allocation, so this function used to return an all-zero model and the
    seed above did nothing. The first `base-loss` leg ran 1,388 steps on it and trained nothing.
    """
    return initialise_(ABB3StructureModule(cfg), seed).state_dict()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=193_512)
    ap.add_argument("--global-batch", type=int, default=64)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--chips", default="0", help="comma-separated chip ids in rank order")
    ap.add_argument("--rendezvous", default="/dev/shm/abb3-dp")
    ap.add_argument("--checkpoint-minutes", type=float, default=30.0)
    ap.add_argument("--data", default="synthetic", help="synthetic | sabdab")
    ap.add_argument("--split-csv",
                    default="/home/ttuser/abb3_src/ABodyBuilder3/data/split.csv",
                    help="upstream's own split, as they ship it")
    ap.add_argument("--released-true",
                    default="/home/ttuser/abb3/base-loss/true",
                    help="their released ground truth, used to cross-check the split")
    ap.add_argument("--structures",
                    default="/home/ttuser/abb3_data/data/structures/structures")
    ap.add_argument("--split", default="train", choices=("train", "valid", "test"))
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--kill-at", type=int, default=0,
                    help="SIGKILL this process after this step, to demonstrate the resume")
    ap.add_argument("--max-seconds", type=float, default=0.0)
    ap.add_argument("--torch-threads", type=int, default=0,
                    help="cap torch's intra-op threads; 0 leaves its default")
    args = ap.parse_args()
    # Set after torch is imported, which is the only setting that reliably takes: OMP_NUM_THREADS
    # is read at import and the supervisor sets it too, so both paths are covered. This exists
    # because the step's serial term is host torch on an 8-core CPU, and N ranks each defaulting
    # to 8 threads oversubscribe the cores N-fold -- which is a different thing from a serial
    # cost and has a different fix.
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    print(f"[rank {args.rank}] torch intra-op threads: {torch.get_num_threads()}", flush=True)

    chips = tuple(int(c) for c in args.chips.split(","))
    if len(chips) != args.world:
        raise SystemExit(f"--chips {args.chips} names {len(chips)} chips for world "
                         f"{args.world}")
    cfg = RunConfig(out_dir=Path(args.out), steps=args.steps, global_batch=args.global_batch,
                    micro_batch=args.micro, tokens=args.tokens, seed=args.seed,
                    checkpoint_minutes=args.checkpoint_minutes, rank=args.rank,
                    world=args.world, chips=chips, rendezvous=Path(args.rendezvous),
                    max_seconds=args.max_seconds)
    mcfg = ABB3Config(use_plddt=False, no_blocks=args.blocks)
    dev = get_device()
    try:
        step = TrainStep(initial_state_dict(mcfg, args.seed), mcfg,
                         accumulate=cfg.accumulate, seed=args.seed)
        ids = None
        if args.data == "sabdab":
            from tt_bio.train.abb3_dataset import resolve_split
            # Cross-checked against their released predictions on every launch, not once at
            # build time: the split is the thing that decides whether the 2.714 A bar applies
            # to the result at all, and it costs a directory listing to re-establish.
            ids = resolve_split(args.split_csv, args.released_true)[args.split]
            print(f"[rank {args.rank}] split {args.split}: {len(ids)} structures, "
                  f"cross-checked against {args.released_true}", flush=True)
        data = make_dataset(args.data, cfg=mcfg, micro=args.micro, tokens=args.tokens,
                            device=dev, ids=ids, root=args.structures)

        def maybe_die(gs, row, _step):
            print(f"[rank {cfg.rank}] step {gs} loss {row.get('loss'):.6f} "
                  f"{row['wall']:.2f}s digest {row['digest'][:12]}", flush=True)
            if args.kill_at and gs >= args.kill_at:
                print(f"[rank {cfg.rank}] SIGKILL self after step {gs} -- standing in for a "
                      f"watchdog reset", flush=True)
                sys.stdout.flush()
                os.kill(os.getpid(), signal.SIGKILL)

        out = run(step, data, cfg, resume=not args.no_resume, on_step=maybe_die)
        print(f"[rank {cfg.rank}] done at step {out['steps_done']}")
        print(f"[rank {cfg.rank}] CLOCK: {out['provenance'].summary()}")
        print(f"[rank {cfg.rank}] COMM: {out['comm']}")
        (Path(args.out) / f"provenance-rank{args.rank}.json").write_text(
            json.dumps(out["provenance"].as_dict(), indent=2, default=str) + "\n")
        # The marker the supervisor reads to know the run finished rather than died.
        if out["steps_done"] >= args.steps:
            (Path(args.out) / f"COMPLETE-rank{args.rank}").write_text(
                f"{out['steps_done']}\n")
        return 0
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
