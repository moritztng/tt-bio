"""p29: attribute EVERY millisecond of one RFD3 diffusion step to a named region.

p26/p28 profiled with ``profile_host_boundary.py --by_site --sync_first`` and got one
undifferentiated majority bucket: ``sync.drain`` = 57.8% of the step at 3359 atoms. That
bucket is device work enqueued *outside* every instrumented section, so no pass could name
it. This script makes it nameable, and the residual it leaves is the proof the attribution
is complete.

Method -- one rule, applied everywhere:

  Around every region, drain the device queue on ENTRY and on EXIT.

  * the ENTRY drain charges work enqueued *before* the region to ``dev.pre@<callsite>``,
    which names the un-sectioned inline glue that enqueued it;
  * the EXIT drain charges the region's own enqueued work to ``dev.<region>``;
  * wall clock between the two drains, minus any nested drains, is ``cpu.<region>``.

Because every drain subtracts itself from its parent's cpu time, the accounting is
EXCLUSIVE: a region's ``dev``/``cpu`` never include an inner region's. ``sum(dev.*)`` is
therefore total device-busy time and ``sum(cpu.*)`` total host time, both partitioned.

Draining serializes host dispatch against device execution, so the attributed total is
LONGER than the real step. ``--mode baseline`` measures the unperturbed ms/step with zero
instrumentation, so the report can state the inflation instead of hiding it.

Usage (shipped config -- do NOT export RFD3_TRACE_DECODER):
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:... PYTHONPATH=<worktree> \
  python3 scripts/rfd3_port/p29_attribute_step.py --contig "A1-10,230,A31-40" --batch 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PDB = ROOT / "scripts/rfd3_port/parity_artifacts/iai_protein/IAI_protein.pdb"
GOLDEN_DIR = Path("~/.coworker/artifacts/rfd3-goldens/capture").expanduser()

_t: dict[str, float] = defaultdict(float)
_n: dict[str, int] = defaultdict(int)
_b: dict[str, int] = defaultdict(int)
_on = False
_used = 0.0  # cumulative seconds already charged to SOME bucket (drains + region cpu)
_DEV = [None]
_TTNN = [None]
_FLOOR = [0.0]  # measured cost of synchronize_device on an already-idle device


def _acc(key, dt, nbytes=0):
    _t[key] += dt
    _n[key] += 1
    _b[key] += nbytes


def _flush(tag):
    """Drain the device queue; charge the drained device time to `tag`.

    Adds to `_used` so every enclosing region subtracts this drain from its own cpu
    time. Every charge to a bucket does the same, which is what makes the split
    EXCLUSIVE: a region's cpu time never contains a nested region's cpu or dev time.
    """
    global _used
    t0 = time.perf_counter()
    _TTNN[0].synchronize_device(_DEV[0])
    dt = time.perf_counter() - t0
    _used += dt
    _acc(f"dev.{tag}", dt)


def _charge(key, dt, nbytes=0):
    global _used
    _acc(key, dt, nbytes)
    _used += dt


def _measure_floor(ttnn, dev, n=400):
    """Cost of one synchronize_device on an idle device.

    This is the instrument's noise floor: ~100 drains per step times this is charged to
    `dev.*` even when the device did nothing, so a small `dev` bucket is meaningless
    below it. Measured, not assumed.
    """
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / n


def _callsite(depth=2):
    """`file:line` of the frame that called the wrapped function."""
    try:
        f = sys._getframe(depth)
        return f"{Path(f.f_code.co_filename).name}:{f.f_lineno}"
    except Exception:
        return "?"


def _section(obj, name, key, *, enqueues=True, tag=None):
    """Wrap obj.name as a sync-bracketed region reported as cpu.<key> / dev.<key>.

    `enqueues=False` marks a pure-host region: no exit drain (it enqueues nothing), so
    the region costs one drain instead of two and its cpu time stays uncontaminated.
    `tag` may be a callable(args) -> suffix, to split one function by argument (the two
    `_create_attention_indices` call sites use very different n_keys).
    """
    orig = getattr(obj, name)

    def wrapped(*a, **kw):
        if not _on:
            return orig(*a, **kw)
        k = f"{key}{tag(a) if tag else ''}"
        _flush(f"pre@{_callsite(2)}")
        u0, outer = _used, _REGION[0]
        _REGION[0] = k  # so --ops attributes each op to the region that issued it
        t0 = time.perf_counter()
        try:
            out = orig(*a, **kw)
            if enqueues:
                _flush(k)  # inside `wall` on purpose: `_used` subtracts it back out
        finally:
            _REGION[0] = outer
        wall = time.perf_counter() - t0
        _charge(f"cpu.{k}", wall - (_used - u0))
        return out

    setattr(obj, name, wrapped)


_REGION = ["step"]

# Every ttnn entry point the per-step path issues. Sync-bracketing each one turns the
# script into a per-op device profiler that measures the REAL in-loop op, not an isolated
# micro-benchmark (p25: an isolated sparse_qk read 21.5 ms and was 3.9% in the real loop).
_OPS = ("linear", "matmul", "rms_norm", "layer_norm", "softmax", "add", "multiply",
        "subtract", "sigmoid", "silu", "typecast", "reshape", "permute", "transpose",
        "concat", "pad", "scatter", "embedding", "to_layout", "unsqueeze", "squeeze",
        "clone", "repeat", "tilize", "untilize")


def _op_bytes(args, kw, out):
    """DRAM traffic an op must move at minimum: every ttnn input read once, output
    written once. A lower bound -- multi-pass kernels move more -- so an achieved
    bandwidth computed from it is a lower bound too."""
    n = 0
    for v in list(args) + list(kw.values()):
        if isinstance(v, _TTNN[0].Tensor):
            n += _tensor_bytes(v)
    if isinstance(out, _TTNN[0].Tensor):
        n += _tensor_bytes(out)
    return n


def _wrap_ops(ttnn):
    """Sync-bracket every ttnn op, keyed by enclosing region + op + output shape/dtype."""
    for name in _OPS:
        orig = getattr(ttnn, name, None)
        if orig is None:
            continue

        def make(orig=orig, name=name):
            def wrapped(*a, **kw):
                if not _on:
                    return orig(*a, **kw)
                u0 = _used
                t0 = time.perf_counter()
                out = orig(*a, **kw)
                k = f"op:{_REGION[0]}|{name}|{_shape_key(out)}"
                _flush(k)
                wall = time.perf_counter() - t0
                _charge(f"cpu.{k}", wall - (_used - u0), _op_bytes(a, kw, out))
                return out
            return wrapped

        setattr(ttnn, name, make())


def _shape_key(t):
    try:
        return f"{list(t.shape)}{str(t.dtype).split('.')[-1][:4]}"
    except Exception:
        return "?"


def _tensor_bytes(t):
    try:
        return int(t.numel()) * int(t.element_size())
    except Exception:
        try:  # ttnn tensor
            n = 1
            for d in t.shape:
                n *= int(d)
            return n * 2
        except Exception:
            return 0


def _wrap_transfers(ttnn):
    """Time every host<->device crossing, sync-bracketed on the read side.

    A read is the one operation that MUST drain, so it is the natural attribution point
    for inline glue that is not a function and cannot be `_section`-wrapped: the drain
    before a read names the code between the previous drain and that read.
    """
    to_torch, from_torch = ttnn.to_torch, ttnn.from_torch

    def rd(*a, **kw):
        if not _on:
            return to_torch(*a, **kw)
        site = _callsite(2)
        _flush(f"pre@{site}")
        u0 = _used
        t0 = time.perf_counter()
        out = to_torch(*a, **kw)
        wall = time.perf_counter() - t0
        _charge(f"cpu.to_torch@{site}", wall - (_used - u0), _tensor_bytes(out))
        return out

    def wr(*a, **kw):
        if not _on:
            return from_torch(*a, **kw)
        u0 = _used
        t0 = time.perf_counter()
        out = from_torch(*a, **kw)
        wall = time.perf_counter() - t0
        _charge(f"cpu.from_torch@{_callsite(2)}", wall - (_used - u0),
                _tensor_bytes(a[0]) if a else 0)
        return out

    ttnn.to_torch, ttnn.from_torch = rd, wr


def instrument(R, dm, ttnn, ops=False):
    """Sync-bracket every region of a step, derived from tt_bio/rfd3.py's call graph.

    Region list follows RFD3DiffusionModule.__call__ -> _forward_with_recycle ->
    _process_ in source order, so a reader can line the table up against the code.
    """
    _DEV[0], _TTNN[0] = dm.device, ttnn
    _wrap_transfers(ttnn)
    if ops:
        _wrap_ops(ttnn)

    # --- pure-host torch kernels (enqueue nothing) ---
    # attn_indices is called at two very different scales in one step: once over all
    # L atoms with 128 keys, then once per recycle over I tokens with 32.
    _section(R, "_create_attention_indices", "attn_indices", enqueues=False,
             tag=lambda a: f"(k={a[3]})")
    _section(R, "_scatter_mean", "scatter_mean", enqueues=False)
    _section(R, "_scaled_distogram_bins", "distogram_bins", enqueues=False)
    _section(R, "_dense_attention_mask", "dense_mask", enqueues=False)
    _section(R, "_sparse_qk_host", "sparse_qk_host", enqueues=False)

    # --- device regions that build attention inputs ---
    _section(R, "_sparse_qk_inputs", "sparse_qk_inputs")
    _section(R, "_pack_atoms_dev_core", "pack_atoms")

    # --- glue on the diffusion module itself ---
    for name, key in (("_downcast_c", "downcast_c"), ("_downcast_q", "downcast_q"),
                      ("_process_time", "process_time"), ("_grouping_buffers", "grouping_buffers")):
        _section(dm, name, key)

    # --- the four big components ---
    # decoder is wrapped on run_full_device, NOT __call__: the per-step path calls
    # run_full_device and keeps both outputs on the card, so every previous pass that
    # wrapped __call__ silently dropped the decoder from its table. Same trap the token
    # encoder hit (it is called as run_device).
    _section(type(dm.encoder), "__call__", "encoder")
    _section(type(dm.diffusion_token_encoder), "run_device", "token_encoder")
    _section(type(dm.diffusion_transformer), "__call__", "dit")
    _section(type(dm.decoder), "run_full_device", "decoder")
    _section(type(dm.sequence_head), "__call__", "sequence_head")

    # --- inside the decoder ---
    for attr, key in (("run_device", "dec.core_loop"), ("_pack_atoms_device", "dec.pack"),
                      ("_unpack_atoms_device", "dec.unpack"), ("_design_buffers", "dec.buffers")):
        _section(type(dm.decoder), attr, key)
    # GatedCrossAttention is shared by three instances; wrap each instance so they split.
    _section(dm.decoder.downcast, "run_device", "dec.downcast")
    _section(dm.downcast_c, "run_device", "gca.downcast_c")
    _section(dm.downcast_q, "run_device", "gca.downcast_q")

    # RFD3AtomBlock is shared by the 3-block encoder, the decoder's core loop and the
    # 18-block DiT. Tagging by sequence length is what separates the 3359-atom blocks from
    # the 250-token ones -- the whole size-dependence question lives in that split.
    # (a[0] is `self`: these wrap the unbound class function.)
    _section(R.RFD3AtomBlock, "__call__", "atom_block", tag=lambda a: f"(n={a[1].shape[1]})")
    _section(R.PairformerBlock, "__call__", "pairformer_block")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contig", default="A1-10,20,A31-40")
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--pdb", type=Path, default=PDB)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--timesteps", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2, help="steps excluded from the accounting")
    ap.add_argument("--mode", choices=("attributed", "baseline"), default="attributed",
                    help="baseline = zero instrumentation, for the perturbation figure")
    ap.add_argument("--ops", action="store_true",
                    help="also bracket every ttnn op: per-op device time and bytes, in the real loop")
    ap.add_argument("--json_out", type=Path)
    args = ap.parse_args()

    import ttnn
    import tt_bio.rfd3 as R
    from tt_bio.rfd3_featurize import featurize
    from tt_bio.rfd3_input import InputSpecification
    from tt_bio.rfd3_sampler import RFD3Sampler

    if args.spec:
        data = json.loads(args.spec.read_text())
        p = Path(data["input"])
        data["input"] = str((p if p.is_absolute() else args.spec.parent / p).resolve())
    else:
        data = {"input": str(args.pdb), "contig": args.contig}
    spec = InputSpecification.from_dict(data)
    spec.validate()
    f = featurize(data["input"], spec)
    f = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in f.items()}
    L = int(f["ref_pos"].shape[0])
    I = int(f["atom_to_token_map"].max().item()) + 1

    ti_w = torch.load(GOLDEN_DIR / "token_initializer.real_weights.pt", map_location="cpu", weights_only=True)
    dm_w = torch.load(GOLDEN_DIR / "diffusion_module.real_weights.pt", map_location="cpu", weights_only=True)
    ti = R.build_token_initializer(ti_w)
    dm = R.build_diffusion_module(dm_w)

    if args.mode == "attributed":
        instrument(R, dm, ttnn, ops=args.ops)

    coord0 = f["motif_pos"].float().unsqueeze(0)
    global _on
    with torch.no_grad():
        init = ti({k: (v.clone() if torch.is_tensor(v) else v) for k, v in f.items()})
        g = torch.Generator().manual_seed(42)
        # warmup: JIT, kernel cache and program cache all land here (p17: a cold read is 13x)
        RFD3Sampler(num_timesteps=args.warmup).sample(
            dm, args.batch, L, coord0, f, init, f["is_motif_atom_with_fixed_coord"], generator=g)
        if args.mode == "attributed":
            _FLOOR[0] = _measure_floor(ttnn, dm.device)
        _t.clear(), _n.clear(), _b.clear()
        _on = args.mode == "attributed"
        g = torch.Generator().manual_seed(42)
        t0 = time.perf_counter()
        RFD3Sampler(num_timesteps=args.timesteps).sample(
            dm, args.batch, L, coord0, f, init, f["is_motif_atom_with_fixed_coord"], generator=g)
        wall = time.perf_counter() - t0
        _on = False

    # RFD3Sampler walks CONSECUTIVE PAIRS of the schedule (zip(sched, sched[1:])), so
    # num_timesteps=N runs N-1 denoise steps. Dividing by N inflates every per-step figure
    # by N/(N-1) -- 20% at N=6.
    steps = args.timesteps - 1
    step_ms = wall / steps * 1e3
    print(f"\n=== p29 step attribution  L={L} atoms  I={I} tokens  batch={args.batch} "
          f"steps={steps}  mode={args.mode} ===")
    print(f"wall {wall * 1e3:.1f} ms total, {step_ms:.2f} ms/step")
    if args.mode == "baseline":
        print("(uninstrumented: this is the honest ms/step the attributed run inflates)")
        if args.json_out:
            args.json_out.write_text(json.dumps({"L": L, "I": I, "batch": args.batch,
                                                 "mode": "baseline", "step_ms": step_ms}, indent=1))
        return

    # Every drain costs `_FLOOR` even on an idle device, so subtract that from each dev
    # bucket: what remains is device work, not instrument overhead.
    floor_ms = _FLOOR[0] * 1e3
    ndrain = sum(v for k, v in _n.items() if k.startswith("dev.")) / steps
    dev = {k[4:]: max(0.0, v / steps * 1e3 - _n[k] / steps * floor_ms)
           for k, v in _t.items() if k.startswith("dev.")}
    dev_raw = sum(v / steps * 1e3 for k, v in _t.items() if k.startswith("dev."))
    cpu = {k[4:]: v / steps * 1e3 for k, v in _t.items() if k.startswith("cpu.")}
    dev_tot, cpu_tot = sum(dev.values()), sum(cpu.values())
    print(f"drain floor       {floor_ms * 1e3:7.1f} us/call x {ndrain:.0f} drains/step "
          f"= {ndrain * floor_ms:.2f} ms/step of pure instrument overhead")
    print(f"DEVICE-BUSY total {dev_tot:7.2f} ms/step   ({dev_tot / step_ms * 100:5.1f}% of attributed wall) "
          f"[raw {dev_raw:.2f} before floor subtraction]")
    print(f"HOST total        {cpu_tot:7.2f} ms/step   ({cpu_tot / step_ms * 100:5.1f}%)")
    print(f"unaccounted       {step_ms - dev_raw - cpu_tot:7.2f} ms/step   "
          f"({(step_ms - dev_raw - cpu_tot) / step_ms * 100:5.1f}%)\n")

    def table(keys, label, width, bw=False):
        print(f"{label:<{width}}{'dev ms':>9}{'dev %':>7}{'cpu ms':>9}{'cpu %':>7}{'calls':>7}"
              f"{'MB':>9}" + (f"{'GB/s':>8}" if bw else ""))
        for k in sorted(keys, key=lambda k: -(dev.get(k, 0) + cpu.get(k, 0))):
            d, c = dev.get(k, 0.0), cpu.get(k, 0.0)
            if d + c < 0.02:
                continue
            mb = _b[f"cpu.{k}"] / steps / 1e6
            line = (f"{k:<{width}}{d:>9.2f}{d / dev_tot * 100 if dev_tot else 0:>6.1f}%"
                    f"{c:>9.2f}{c / cpu_tot * 100 if cpu_tot else 0:>6.1f}%"
                    f"{_n.get(f'dev.{k}', _n.get(f'cpu.{k}', 0)) / steps:>7.1f}{mb:>9.2f}")
            if bw:
                line += f"{mb / 1e3 / (d / 1e3) if d > 0.01 else 0:>8.0f}"
            print(line)

    keys = set(dev) | set(cpu)
    table({k for k in keys if not k.startswith("op:")}, "region", 40)
    ops = {k for k in keys if k.startswith("op:")}
    if ops:
        # GB/s uses the lower-bound byte count (inputs once + output once), so it is a
        # LOWER bound on achieved bandwidth -- the compute-vs-bandwidth verdict has to
        # survive that direction of error.
        print(f"\nper-op, real in-loop, {len(ops)} distinct (region|op|out-shape):")
        table(ops, "op", 66, bw=True)

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"L": L, "I": I, "batch": args.batch, "mode": "attributed", "steps": steps,
             "step_ms": step_ms, "dev_total_ms": dev_tot, "dev_raw_ms": dev_raw,
             "cpu_total_ms": cpu_tot, "drain_floor_us": floor_ms * 1e3, "drains_per_step": ndrain,
             "dev": dev, "cpu": cpu,
             "calls": {k: v / steps for k, v in _n.items()},
             "bytes": {k: v / steps for k, v in _b.items()}}, indent=1))


if __name__ == "__main__":
    main()
