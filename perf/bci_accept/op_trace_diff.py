"""Do the taped and untaped arms run the same SEQUENCE of ops through one Evoformer block?

`op_shadow_diff.py` showed that with `TT_BIO_MM_LAYOUT=0` every op the taped block runs is
bit-identical to its shipped verb on the same operands, and the block output still differs by
0.0028 relative L2 on the pair track. If no op computes a different number, the two arms must be
running different ops: a branch on `ag.is_grad_enabled()`, a helper that composes on the tape
what it fuses off it, a different chunking.

This records the op trace of each arm and diffs the two. The recorder wraps `ttnn`'s own
callables BEFORE the first tape opens. The tape's shim caches `getattr(ttnn, name)` at first use,
so it captures the recorder and calls it as the shipped verb, and the untaped arm calls the same
recorder directly: one mechanism sees both arms. Each entry is (op, first tt-bio call site
outside the tape, operand shapes). Only calls with a tt-bio caller are kept.

  TT_VISIBLE_DEVICES=<card> python3 perf/bci_accept/op_trace_diff.py --card <card> \
      --af2-weights ~/bcx_e2e/af2_params --tokens 64
"""
from __future__ import annotations

import argparse
import collections
import difflib
import inspect
import os
import sys

import numpy as np

SKIP_FRAMES = ("/taped_ttnn.py", "/autograd.py", "/op_trace_diff.py")


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(b.ravel()))
    return float("nan") if denominator == 0.0 else float(
        np.linalg.norm((a - b).ravel()) / denominator)


def caller_site():
    frame = sys._getframe(2)
    while frame is not None:
        path = frame.f_code.co_filename
        if "/tt_bio/" in path and not path.endswith(SKIP_FRAMES):
            return f"{path.rsplit('/tt_bio/', 1)[1]}:{frame.f_lineno}"
        frame = frame.f_back
    return None


#: Ops whose operands are described in full rather than by shape. `batched_matmul`s tuned
#: branch is gated on dtype equality and on BOTH operands being DRAM interleaved
#: (`tenstorrent.py:3322`), so a shape-only trace cannot say why one arm qualifies for it and
#: the other does not.
DETAILED = {"matmul", "experimental.minimal_matmul", "linear"}


def describe(value):
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        shape = tuple(int(d) for d in shape)
    except Exception:
        return "?"
    try:
        memory = value.memory_config()
        where = f"{memory.buffer_type.name}/{memory.memory_layout.name}"
    except Exception:
        where = "?"
    return f"{shape}:{str(getattr(value, 'dtype', '?')).rsplit('.', 1)[-1]}:{where}"


def shapes(args, kwargs, name=""):
    detail = name in DETAILED
    out = []
    for value in list(args) + list(kwargs.values()):
        if detail:
            described = describe(value)
            if described is not None:
                out.append(described)
            continue
        shape = getattr(value, "shape", None)
        if shape is not None:
            try:
                out.append(tuple(int(d) for d in shape))
            except Exception:
                out.append("?")
    return tuple(out)


class Recorded:
    """A ttnn callable that appends to the current arm's trace, then calls through."""

    def __init__(self, name, fn, sink):
        self._name, self._fn, self._sink = name, fn, sink

    def __call__(self, *args, **kwargs):
        if self._sink.arm is not None:
            site = caller_site()
            if site is not None:
                self._sink.trace[self._sink.arm].append(
                    (self._name, site, shapes(args, kwargs, self._name)))
        return self._fn(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self._fn, attr)


class Sink:
    def __init__(self):
        self.arm = None
        self.trace = collections.defaultdict(list)


def install(ttnn, sink) -> int:
    count = 0
    for namespace, prefix in ((ttnn, ""), (getattr(ttnn, "transformer", None), "transformer."),
                              (getattr(ttnn, "experimental", None), "experimental.")):
        if namespace is None:
            continue
        for name in dir(namespace):
            if name.startswith("_"):
                continue
            fn = getattr(namespace, name, None)
            if fn is None or inspect.isclass(fn) or inspect.ismodule(fn) or not callable(fn):
                continue
            try:
                setattr(namespace, name, Recorded(prefix + name, fn, sink))
                count += 1
            except Exception:
                pass
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--af2-weights", required=True)
    parser.add_argument("--model", default="model_1_multimer_v3")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--n-seq", type=int, default=1)
    parser.add_argument("--c-m", type=int, default=256)
    parser.add_argument("--c-z", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--hunks", type=int, default=30, help="differing hunks printed")
    parser.add_argument("--dump", default=".bci/trace",
                        help="path prefix for the full per-arm traces. The hunks printed below "
                             "are an alignment, not the data; the tsv is what a later pass reads "
                             "to answer a question this run did not ask, with no card.")
    args = parser.parse_args()

    if "TT_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit("set TT_VISIBLE_DEVICES to the leased card before running this")
    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor(device=args.card)

    import ttnn
    from tt_bio import bindcraft2

    sink = Sink()
    print(f"recorder on {install(ttnn, sink)} ttnn callables, before any tape", flush=True)

    rng = np.random.default_rng(args.seed)
    n = args.tokens
    msa = (rng.standard_normal((args.n_seq, n, args.c_m)) * args.scale).astype(np.float32)
    pair = (rng.standard_normal((n, n, args.c_z)) * args.scale).astype(np.float32)
    mask = np.ones((args.n_seq, n), dtype=np.float32)
    pair_mask = np.ones((n, n), dtype=np.float32)

    pool = bindcraft2.TrunkPool(args.af2_weights)
    pool.require([args.model])
    if args.model in pool.absent:
        raise SystemExit(f"{args.model} not in {args.af2_weights}: {pool.absent[args.model]}")
    pool.use(args.model)

    with bindcraft2.refusals_unwrapped(), bindcraft2.fast_round():
        evo = bindcraft2.EvoformerOnDevice(pool, blocks=bindcraft2.EVOFORMER_BLOCKS,
                                           recompute=False, memory=bindcraft2._Memory("fast"))
        trunk = evo._trunk("")
        blocks = list(trunk.model.device_evoformer)
        try:
            carry_msa, carry_pair = msa, pair
            for index in range(args.warmup):
                trunk.model.device_evoformer = [blocks[index]]
                carry_msa, carry_pair = evo._primal("", carry_msa, carry_pair, mask, pair_mask)
            trunk.model.device_evoformer = [blocks[args.warmup]]
            # Each arm twice, so a sequence that varies between two runs of the SAME arm (a
            # cache warming, a plan retiring a rung) is seen as such and not blamed on the tape.
            for arm in ("untaped", "untaped_again", "taped", "taped_again"):
                sink.arm = arm
                if arm.startswith("untaped"):
                    out = evo._primal("", carry_msa, carry_pair, mask, pair_mask)
                else:
                    out = evo._taped("", carry_msa, carry_pair, mask, pair_mask)[:2]
                sink.arm = None
                if arm == "untaped":
                    reference = out
                from tt_bio import tenstorrent as _tt
                print(f"{arm:>14}: {len(sink.trace[arm])} ops, msa rel L2 "
                      f"{rel_l2(out[0], reference[0]):.6f}, pair rel L2 "
                      f"{rel_l2(out[1], reference[1]):.6f}, bmm rungs "
                      f"{dict(getattr(_tt, '_BMM_CFG_RUNG', {}))}, refused "
                      f"{len(getattr(_tt, '_BMM_CFG_REFUSED', ()))}", flush=True)
        finally:
            trunk.model.device_evoformer = blocks

    for arm, entries in sink.trace.items():
        path = f"{args.dump}_{arm}_t{args.tokens}.tsv"
        with open(path, "w") as handle:
            for index, (name, site, operands) in enumerate(entries):
                handle.write(f"{index}\t{name}\t{site}\t{operands}\n")
        print(f"wrote {len(entries)} ops to {path}", flush=True)

    def key(entry):
        return entry[0], entry[1]

    for left, right in (("untaped", "untaped_again"), ("taped", "taped_again"),
                        ("untaped", "taped")):
        a = [key(e) for e in sink.trace[left]]
        b = [key(e) for e in sink.trace[right]]
        matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        hunks = [op for op in matcher.get_opcodes() if op[0] != "equal"]
        shape_diffs = sum(1 for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag == "equal"
                          for x, y in zip(sink.trace[left][i1:i2], sink.trace[right][j1:j2])
                          if x[2] != y[2])
        print(f"\n=== {left} vs {right}: {len(a)} vs {len(b)} ops, {len(hunks)} differing "
              f"hunks, {shape_diffs} matched ops with different operand shapes ===", flush=True)
        for tag, i1, i2, j1, j2 in hunks[:args.hunks]:
            print(f"--- {tag} {left}[{i1}:{i2}] -> {right}[{j1}:{j2}]", flush=True)
            for entry in sink.trace[left][i1:i2][:8]:
                print(f"  - {entry[0]:<36} {entry[1]:<30} {entry[2]}", flush=True)
            for entry in sink.trace[right][j1:j2][:8]:
                print(f"  + {entry[0]:<36} {entry[1]:<30} {entry[2]}", flush=True)
        counts_a, counts_b = collections.Counter(a), collections.Counter(b)
        only = (counts_b - counts_a) + (counts_a - counts_b)
        if only:
            print(f"  multiset difference ({sum(only.values())} calls): "
                  f"{sorted(only.items())[:20]}", flush=True)


if __name__ == "__main__":
    main()
