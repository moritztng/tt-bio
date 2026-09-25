#!/usr/bin/env python3
"""Count ttnn device ops and host dispatch cost per warm fold, at a given tt-bio tree.

Mirrors scripts/perf_regression.py::measure() (same fixture, same WARMUP/REPEAT,
same in-process _WorkerState protocol) and adds three things the gate does not
record:

  * ttnn.get_device_operation_id() delta per call -- the exact number of ttnn
    device-op enqueues the call issues. This is tt-metal's own per-enqueue
    counter, incremented once in enqueue_mesh_workload (ttnn device_operation.hpp),
    so it is a count, not an estimate.
  * host CPU time (time.process_time) per call, so wall/op and cpu/op can be read
    separately on a contended box.
  * AICLK sampled from sysfs DURING the timed region, plus loadavg either side.

Run it against the tree you want to measure via PYTHONPATH, one model per process.
"""
import argparse
import json
import os
import statistics
import tempfile
import threading
import time
from pathlib import Path

AICLK_NODES = sorted(Path("/sys/class/tenstorrent").glob("tenstorrent!*"))


class ClkSampler(threading.Thread):
    """Sample every card's AICLK while the timed region runs."""

    def __init__(self, period=0.05):
        super().__init__(daemon=True)
        self.period, self.stop_flag, self.samples = period, False, []

    def run(self):
        while not self.stop_flag:
            row = {}
            for n in AICLK_NODES:
                try:
                    row[n.name.split("!")[-1]] = int((n / "tt_aiclk").read_text().strip())
                except Exception:
                    pass
            if row:
                self.samples.append(row)
            time.sleep(self.period)

    def summary(self):
        out = {}
        for card in {k for s in self.samples for k in s}:
            vals = [s[card] for s in self.samples if card in s]
            if vals:
                out[card] = {"min": min(vals), "median": int(statistics.median(vals)),
                             "max": max(vals), "n": len(vals)}
        return out


def enqueue_probe(reps, shape):
    """Time the bare ttnn enqueue path: one program-cached tiny op, no host sync
    inside the loop. Separates host enqueue cost from device execution cost."""
    import torch
    import ttnn
    from ttnn._ttnn import get_device_operation_id
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    a = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    for _ in range(64):  # fill the program cache, absorb the first compile
        ttnn.deallocate(ttnn.relu(a))
    ttnn.synchronize_device(dev)

    # Chunked: a loud host preempts the loop, which can only ADD time. The
    # minimum chunk is therefore the least-contaminated estimate of the
    # uncontended per-enqueue cost; the median chunk is what this box delivers
    # under its current load. Reporting both makes the contention visible
    # instead of folding it into one number.
    chunk = 100
    n_chunks = max(1, reps // chunk)
    chunk_us = []
    id0 = get_device_operation_id()
    w0, c0 = time.perf_counter(), time.process_time()
    for _ in range(n_chunks):
        t0 = time.perf_counter()
        for _ in range(chunk):
            ttnn.deallocate(ttnn.relu(a))
        chunk_us.append((time.perf_counter() - t0) / chunk * 1e6)
    w_enq, c_enq = time.perf_counter() - w0, time.process_time() - c0
    ttnn.synchronize_device(dev)
    w_tot = time.perf_counter() - w0
    ops = get_device_operation_id() - id0
    reps = n_chunks * chunk
    cs = sorted(chunk_us)
    return {
        "chunk_size": chunk, "n_chunks": n_chunks,
        "enqueue_us_per_op_min_chunk": round(cs[0], 3),
        "enqueue_us_per_op_p10_chunk": round(cs[len(cs) // 10], 3),
        "enqueue_us_per_op_median_chunk": round(cs[len(cs) // 2], 3),
        "probe_shape": list(shape), "probe_reps": reps, "probe_device_ops": ops,
        "enqueue_wall_us_per_op": round(w_enq / reps * 1e6, 3),
        "enqueue_cpu_us_per_op": round(c_enq / reps * 1e6, 3),
        "enqueue_plus_drain_us_per_op": round(w_tot / reps * 1e6, 3),
        "enqueue_host_share_of_total": round(w_enq / w_tot, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esmfold2-fast")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-reps", type=int, default=2000)
    ap.add_argument("--probe-shape", type=int, nargs=2, default=[32, 32])
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--probe-only", action="store_true",
                    help="open the device and run the enqueue probe only, no model load")
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    import perf_regression as PR
    import ttnn
    from ttnn._ttnn import get_device_operation_id
    from tt_bio.tenstorrent import get_device, arch_name, cleanup
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor

    if args.probe_only:
        from tt_bio.main import _detect_p300_devices as _dp, _find_ttnn_mesh_graph_descriptor as _fm
        if _dp() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
            _mgd = _fm("p150_mesh_graph_descriptor.textproto")
            if _mgd:
                os.environ["TT_MESH_GRAPH_DESC_PATH"] = _mgd
        load1 = os.getloadavg()
        clk = ClkSampler()
        clk.start()
        res = enqueue_probe(args.probe_reps, tuple(args.probe_shape))
        clk.stop_flag = True
        clk.join(timeout=2)
        res.update(
            probe_only=True, machine_id=PR.detect_machine_id(), card_type=PR.detect_card_type(),
            tt_bio_version=PR._version(),
            inspector_env=os.environ.get("TT_METAL_INSPECTOR", "(unset: tt-metal default ON)"),
            visible_devices=os.environ.get("TT_VISIBLE_DEVICES"),
            loadavg_before=load1, loadavg_after=os.getloadavg(), nproc=os.cpu_count(),
            aiclk_during=clk.summary(),
            utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        cleanup()
        return

    spec = PR.SPECS[args.model]
    assert spec["kind"] in ("fold", "embed"), "%s: unsupported kind %s" % (args.model, spec["kind"])
    def _noop(*a, **k):
        return None
    _E.set_progress(_noop)

    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    # v0.9.0 spells the trace reservation trace_region_size=<bytes>; main spells
    # it trace=<bool>. Accept both so one script measures either tree.
    _trace_sz = spec.get("trace_region_size", 0)
    try:
        get_device(trace_region_size=_trace_sz)
    except TypeError:
        get_device(trace=bool(_trace_sz))
    hw, card = arch_name(), PR.detect_card_type()

    work = Path(tempfile.mkdtemp(prefix="nihops-%s-" % args.model))
    struct_dir, msa_dir = work / "out", work / "msa"
    struct_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    if spec["kind"] == "fold":
        input_path = Path(spec.get("input") or PR.TRPCAGE)
    else:
        input_path = work / "embed.fasta"
        PR._write_embed_fasta(input_path, spec["n_seqs"])

    cfg = PR._build_cfg(args.model, spec, struct_dir, msa_dir)
    _ensure_local_artifacts(cfg)
    state = _WorkerState("tenstorrent")
    t_load = time.perf_counter()
    state.load_model(cfg)
    load_s = time.perf_counter() - t_load
    state.bind_run("perf", cfg)
    state.pfn = _noop
    if cfg["model"] == "boltz2":
        state.model.progress_fn = _noop
    job_cfg = dict(cfg)

    def one_call():
        job_cfg["struct_dir"] = str(struct_dir)
        for p in struct_dir.glob("*"):
            p.unlink()
        id0 = get_device_operation_id()
        w0, c0 = time.perf_counter(), time.process_time()
        state.predict_one(input_path, job_cfg)
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        return wall, cpu, get_device_operation_id() - id0

    for _ in range(args.warmup):
        one_call()

    load1 = os.getloadavg()
    clk = ClkSampler()
    clk.start()
    draws = [one_call() for _ in range(args.repeat)]
    clk.stop_flag = True
    clk.join(timeout=2)
    load2 = os.getloadavg()

    walls = sorted(d[0] for d in draws)
    median = walls[len(walls) // 2]
    ops = [d[2] for d in draws]
    cpus = sorted(d[1] for d in draws)
    cpu_med = cpus[len(cpus) // 2]
    ops_med = sorted(ops)[len(ops) // 2]
    n = 1 if spec["kind"] == "fold" else spec["n_seqs"]

    result = dict(
        model=args.model, kind=spec["kind"], unit=spec["unit"], hardware=hw, card_type=card,
        machine_id=PR.detect_machine_id(), tt_bio_version=PR._version(),
        inspector_env=os.environ.get("TT_METAL_INSPECTOR", "(unset: tt-metal default ON)"),
        visible_devices=os.environ.get("TT_VISIBLE_DEVICES"),
        throughput=round(n / median, 6), latency_ms=round(median * 1000, 2),
        median_s=round(median, 4), times_s=[round(w, 4) for w in walls],
        cpu_s=[round(c, 4) for c in cpus], cpu_median_s=round(cpu_med, 4),
        cpu_fraction_of_wall=round(cpu_med / median, 4),
        device_ops_per_call=ops, device_ops_median=ops_med,
        wall_us_per_op=round(median / ops_med * 1e6, 3),
        cpu_us_per_op=round(cpu_med / ops_med * 1e6, 3),
        load_s=round(load_s, 1), warmup=args.warmup, repeat=args.repeat,
        loadavg_before=load1, loadavg_after=load2, nproc=os.cpu_count(),
        aiclk_during=clk.summary(),
        sampling_steps=PR.SAMPLING_STEPS, diffusion_samples=PR.DIFFUSION_SAMPLES,
        recycling_steps=PR.RECYCLING_STEPS,
        input="trpcage (20 aa, single-seq)" if spec["kind"] == "fold"
              else "%dx ubiquitin (76 aa), batch %s" % (spec["n_seqs"], spec["batch_size"]),
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    if not args.skip_probe:
        result.update(enqueue_probe(args.probe_reps, tuple(args.probe_shape)))

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in
                      ("model", "throughput", "latency_ms", "device_ops_median",
                       "wall_us_per_op", "cpu_us_per_op", "cpu_fraction_of_wall",
                       "aiclk_during", "loadavg_before")}, indent=2))
    state.reset()
    cleanup()


if __name__ == "__main__":
    main()
