"""Critical path inside `reblock_permute_gated`, measured by removal.

The op is the largest single program hiding inside the census's `GenericOp` row: 29.6 % of that
row's device time on the Blackhole capture, 994.5 us a call at the production shape
xw [1, 512, 512, 512] -> out [1, 128, 512, 512].

A counter tells you a thread was blocked. It does not tell you whether unblocking it would move the
wall, because a blocked thread can be off the critical path. So this deletes one stage at a time
from the kernel source and reads the change in wall. d(wall) from deleting stage X IS the time X
contributes on the path; a stage that is fully hidden reads zero.

Three arms produce WRONG DATA on purpose and are timing-only: `nogather`, `nowrite`, `nosigmoid`.
They are never a correctness claim and the parity arm is run separately on `base`.

No model code is touched. Kernel variants are generated into a scratch dir from the shipped
sources with explicit, asserted string edits, and the descriptor is built here rather than by
`tt_bio.reblock_permute` so CB depths are under this file's control.
"""
from __future__ import annotations

import argparse, json, os, re, statistics as st, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
SRC = REPO / "tt_bio" / "kernels" / "reblock_permute_gated"
SCRATCH = Path(os.environ.get("K10_SCRATCH", "/tmp/k10_gated_ablate"))

TILE = 32
P_CB, G_CB, SIG_CB, MUL_CB = 0, 1, 2, 3
OUT_CB, STAGE_CB = 16, 24
GROUP_TILES = 32


def edit(text, old, new, name, count=1):
    n = text.count(old)
    assert n == count, f"{name}: expected {count} occurrences of {old[:60]!r}, found {n}"
    return text.replace(old, new)


# --- the arms -------------------------------------------------------------------------------------
# Each returns (reader_src, compute_src, writer_src, gran, cb_depth_pg, extra_note).

def _srcs():
    return (
        (SRC / "reader_reblock_permute_gated.cpp").read_text(),
        (SRC / "compute_reblock_permute_gated.cpp").read_text(),
        (SRC / "writer_reblock_permute_gated.cpp").read_text(),
    )


def arm_base():
    r, c, w = _srcs()
    return r, c, w, 2, 4, "shipped kernels, GATE_GRANULARITY=2"


def arm_nogather():
    """Writer: keep every address computation, delete only the 64 NOC transactions per tile."""
    r, c, w = _srcs()
    w = edit(w, "noc_async_read_one_packet_with_state(s0, d0);\n", "", "nogather.s0", count=2)
    w = re.sub(r" *noc_async_read_one_packet_with_state\(s1, d1\);\n", "", w)
    assert "with_state(s1" not in w
    return r, c, w, 2, 4, "writer L1->L1 gather transactions deleted (WRONG DATA)"


def arm_nowrite():
    """Writer: delete the 2 KB DRAM tile write, keep the gather and both barriers."""
    r, c, w = _srcs()
    w = edit(w, "noc_async_write(stage_base, s.get_noc_addr(out_page), tile_bytes);",
             "(void)stage_base;", "nowrite")
    return r, c, w, 2, 4, "writer DRAM tile write deleted (WRONG DATA)"


def arm_nosigmoid():
    """Compute: skip_sigmoid compile-time arg. The SFPU sigmoid disappears, the CB round trip stays."""
    r, c, w = _srcs()
    return r, c, w, 2, 4, "SFPU sigmoid deleted, sig_cb round trip kept (WRONG DATA)"


def arm_notranspose():
    """Compute: transpose_wh -> copy_tile. Isolates the unpacker-side transpose (WRONG DATA)."""
    r, c, w = _srcs()
    c = edit(c, "        transpose_wh_init(mul_cb, out_cb);", "        copy_tile_to_dst_init_short(mul_cb);", "notr.init")
    c = edit(c, "            transpose_wh_tile(mul_cb, j, j);", "            copy_tile(mul_cb, j, j);", "notr.op")
    return r, c, w, 2, 4, "transpose_wh -> copy_tile (WRONG DATA)"


def arm_fusedsig():
    """Compute: collapse stages 1 and 2. sigmoid stays in DST, the sig_cb round trip is deleted.

    7 tile moves a tile -> 5. NOT bit-exact against the shipped kernel by construction: the
    multiply then reads the sigmoid at full DST precision instead of at bf16, which is closer to
    the reference and further from ttnn. Correctness is scored in Angstrom by the campaign, not
    here; this arm exists to price the round trip.
    """
    r, c, w = _srcs()
    old_s1 = c[c.index("        cb_wait_front(g_cb, n);"):c.index("        cb_wait_front(p_cb, n);")]
    c = edit(c, old_s1, "", "fusedsig.stage1")
    old_s2 = """        cb_wait_front(p_cb, n);
        cb_wait_front(sig_cb, n);
        cb_reserve_back(mul_cb, n);
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            // Two DST slots a tile, so the multiply reads the same two operands it always did.
            copy_tile_to_dst_init_short(p_cb);
            copy_tile(p_cb, j, 2 * j);
            copy_tile_to_dst_init_short(sig_cb);
            copy_tile(sig_cb, j, 2 * j + 1);
            mul_binary_tile_init();
            mul_binary_tile(2 * j, 2 * j + 1, 2 * j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile(2 * j, mul_cb);
        }
        tile_regs_release();
        cb_pop_front(p_cb, n);
        cb_pop_front(sig_cb, n);
        cb_push_back(mul_cb, n);
"""
    new_s2 = """        cb_wait_front(p_cb, n);
        cb_wait_front(g_cb, n);
        cb_reserve_back(mul_cb, n);
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            copy_tile_to_dst_init_short(p_cb);
            copy_tile(p_cb, j, 2 * j);
            copy_tile_to_dst_init_short(g_cb);
            copy_tile(g_cb, j, 2 * j + 1);
            if constexpr (!skip_sigmoid) {
                sigmoid_tile_init();
                sigmoid_tile(2 * j + 1);
            }
            mul_binary_tile_init();
            mul_binary_tile(2 * j, 2 * j + 1, 2 * j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile(2 * j, mul_cb);
        }
        tile_regs_release();
        cb_pop_front(p_cb, n);
        cb_pop_front(g_cb, n);
        cb_push_back(mul_cb, n);
"""
    c = edit(c, old_s2, new_s2, "fusedsig.stage2")
    return r, c, w, 2, 4, "sigmoid folded into the multiply's DST, sig_cb round trip deleted"


def _reader_batched(depth):
    """Reader: issue `depth` tile PAIRS behind one barrier instead of one pair behind one barrier."""
    r, c, w = _srcs()
    old = """            for (uint32_t il = 0; il < rows_valid; ++il) {
                cb_reserve_back(cb_p, onetile);
                cb_reserve_back(cb_g, onetile);
                noc_async_read_page(p_page, s, get_write_ptr(cb_p));
                noc_async_read_page(g_page, s, get_write_ptr(cb_g));
                noc_async_read_barrier();
                cb_push_back(cb_p, onetile);
                cb_push_back(cb_g, onetile);
                p_page += row_stride;
                g_page += row_stride;
            }"""
    new = """            for (uint32_t il = 0; il < rows_valid; il += BATCH) {
                const uint32_t nb = (rows_valid - il < BATCH) ? (rows_valid - il) : BATCH;
                cb_reserve_back(cb_p, nb);
                cb_reserve_back(cb_g, nb);
                uint32_t wp = get_write_ptr(cb_p);
                uint32_t wg = get_write_ptr(cb_g);
                for (uint32_t b = 0; b < nb; ++b) {
                    noc_async_read_page(p_page, s, wp);
                    noc_async_read_page(g_page, s, wg);
                    p_page += row_stride;
                    g_page += row_stride;
                    wp += TILE_BYTES;
                    wg += TILE_BYTES;
                }
                noc_async_read_barrier();
                cb_push_back(cb_p, nb);
                cb_push_back(cb_g, nb);
            }"""
    r = edit(r, old, new, "readbatch.valid")
    r = edit(r, "    constexpr uint32_t TILE_HEIGHT = 32;",
             f"    constexpr uint32_t TILE_HEIGHT = 32;\n"
             f"    constexpr uint32_t BATCH = {depth};\n"
             f"    constexpr uint32_t TILE_BYTES = 2048;", "readbatch.const")
    return r, c, w


def _mk_batch(batch, depth):
    def f():
        r, c, w = _reader_batched(batch)
        return (r, c, w, 2, depth,
                f"reader: {batch} tile pairs behind one barrier, p/g CBs {depth} tiles deep "
                f"({depth // batch} batches in flight)")
    return f


def _mk_deep(depth):
    def f():
        r, c, w = _srcs()
        return r, c, w, 2, depth, f"p/g CBs {depth} tiles deep, one barrier per pair as shipped"
    return f


arm_readbatch8 = _mk_batch(8, 16)
arm_deepcb = _mk_deep(16)



def arm_noread():
    """Reader: delete the two DRAM page reads and the barrier, keep the CB reserve/push. Prices the
    DRAM read side of the op (WRONG DATA)."""
    r, c, w = _srcs()
    r = edit(r, """                noc_async_read_page(p_page, s, get_write_ptr(cb_p));
                noc_async_read_page(g_page, s, get_write_ptr(cb_g));
                noc_async_read_barrier();
""", "", "noread.valid")
    r = edit(r, """                noc_async_read_page(p_pad, s, get_write_ptr(cb_p));
                noc_async_read_page(g_pad, s, get_write_ptr(cb_g));
                noc_async_read_barrier();
""", "", "noread.pad")
    return r, c, w, 2, 4, "reader DRAM page reads deleted (WRONG DATA)"


def arm_nomul():
    """Compute: drop the SFPU multiply, keep both operand copies and every CB round trip."""
    r, c, w = _srcs()
    c = edit(c, """            mul_binary_tile_init();
            mul_binary_tile(2 * j, 2 * j + 1, 2 * j);
""", "", "nomul")
    return r, c, w, 2, 4, "SFPU multiply deleted, both copies and all CB traffic kept (WRONG DATA)"


def arm_passthru():
    """Compute: one stage, p_cb -> out_cb. Deletes the whole three-stage structure: 7 tile moves a
    tile become 2, and the two intermediate CBs disappear from the dependency chain (WRONG DATA)."""
    r, c, w = _srcs()
    body = c[c.index("    for (uint32_t i = 0; i < num_tiles; i += GRAN) {"):c.rindex("}\n")]
    new = """    for (uint32_t i = 0; i < num_tiles; i += GRAN) {
        const uint32_t n = (num_tiles - i < GRAN) ? (num_tiles - i) : GRAN;
        cb_wait_front(p_cb, n);
        cb_wait_front(g_cb, n);
        cb_reserve_back(out_cb, n);
        transpose_wh_init(p_cb, out_cb);
        tile_regs_acquire();
        for (uint32_t j = 0; j < n; ++j) {
            transpose_wh_tile(p_cb, j, j);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < n; ++j) {
            pack_tile(j, out_cb);
        }
        tile_regs_release();
        cb_pop_front(p_cb, n);
        cb_pop_front(g_cb, n);
        cb_push_back(out_cb, n);
    }
"""
    c = edit(c, body, new, "passthru")
    return r, c, w, 2, 4, "compute collapsed to one transpose stage, 7 tile moves -> 2 (WRONG DATA)"



def _mk_bankspread(mask, depth=4):
    """Reader: XOR the low bits of the page index so a core's 32 reads walk DRAM banks instead of
    hammering one.

    In the shipped kernel `page = 256*row + 16*jt + off + ct` for the 512 aa trimul, so
    `page % 8 == ct % 8` with `ct` fixed inside a group: EVERY read a core issues for a group lands
    on the same DRAM bank. XOR-ing `il & mask` into the low bits keeps the page inside the same
    16-page (row, jt) block, so it is always a valid address, and spreads it over `mask+1` banks.
    The data is wrong on purpose; this measures what bank parallelism is worth and nothing else.
    """
    def f():
        r, c, w = _srcs()
        r = edit(r, "noc_async_read_page(p_page, s, get_write_ptr(cb_p));",
                 f"noc_async_read_page(p_page ^ (il & {mask}u), s, get_write_ptr(cb_p));",
                 "bank.p")
        r = edit(r, "noc_async_read_page(g_page, s, get_write_ptr(cb_g));",
                 f"noc_async_read_page(g_page ^ (il & {mask}u), s, get_write_ptr(cb_g));",
                 "bank.g")
        return r, c, w, 2, depth, f"reader pages XOR il&{mask}: {mask+1} DRAM banks a core (WRONG DATA)"
    return f


ARMS = {
    "base": arm_base, "nogather": arm_nogather, "nowrite": arm_nowrite,
    "nosigmoid": arm_nosigmoid, "notranspose": arm_notranspose,
    "fusedsig": arm_fusedsig, "readbatch8": arm_readbatch8, "deepcb": arm_deepcb,
    "noread": arm_noread, "nomul": arm_nomul, "passthru": arm_passthru,
    "bank2": _mk_bankspread(1), "bank4": _mk_bankspread(3), "bank8": _mk_bankspread(7),
    "bank8d32": _mk_bankspread(7, 32),
    "deepcb32": _mk_deep(32), "deepcb64": _mk_deep(64),
    "batch4d32": _mk_batch(4, 32), "batch8d64": _mk_batch(8, 64), "batch2d16": _mk_batch(2, 16),
}
WRONG = {"nogather", "nowrite", "nosigmoid", "notranspose", "noread", "nomul",
         "passthru", "bank2", "bank4", "bank8", "bank8d32"}


def materialise(name):
    r, c, w, gran, pg_depth, note = ARMS[name]()
    d = SCRATCH / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "reader_reblock_permute_gated.cpp").write_text(r)
    (d / "compute_reblock_permute_gated.cpp").write_text(c)
    (d / "writer_reblock_permute_gated.cpp").write_text(w)
    return d, gran, pg_depth, note


# --- descriptor -------------------------------------------------------------------------------------

def build(ttnn, device, x, out, kdir, gran, pg_depth, skip_sigmoid):
    from tt_bio import reblock_permute as RP
    N = int(out.shape[2])
    Ctw = int(x.shape[3]) // TILE
    Ct = int(out.shape[1]) // TILE
    Nt = (N + TILE - 1) // TILE
    Nrt = (int(x.shape[1]) + TILE - 1) // TILE
    num_groups = Nrt * Nt * Ct
    plan = RP._split_plan(device, num_groups)
    assert plan is not None
    _, _, (_, core_grid, cg1, cg2, work1, work2) = plan
    tb = TILE * TILE * 2

    def cb(idx, depth):
        fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16, page_size=tb)
        return ttnn.CBDescriptor(total_size=depth * tb, core_ranges=core_grid,
                                 format_descriptors=[fmt])

    cbs = [cb(P_CB, pg_depth), cb(G_CB, pg_depth), cb(SIG_CB, 2 * gran), cb(MUL_CB, 2 * gran),
           cb(OUT_CB, GROUP_TILES * 2), cb(STAGE_CB, 2)]
    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    start = 0
    for group, per_core in ((cg1, work1), (cg2, work2)):
        for cr in group.ranges():
            for cx in range(cr.start.x, cr.end.x + 1):
                for cy in range(cr.start.y, cr.end.y + 1):
                    reader_rt[cx][cy] = [start, per_core, Nt, N, Ct, Ctw]
                    compute_rt[cx][cy] = [per_core * GROUP_TILES]
                    writer_rt[cx][cy] = [start, per_core, Nt, N, Ct]
                    start += per_core
    assert start == num_groups
    reader_ct = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    writer_ct = [2, OUT_CB, TILE, TILE, 16, 16, STAGE_CB]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    reader = ttnn.KernelDescriptor(
        kernel_source=str(kdir / "reader_reblock_permute_gated.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=core_grid,
        compile_time_args=reader_ct, runtime_args=reader_rt,
        common_runtime_args=[0, 0, 0, 0], config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=str(kdir / "writer_reblock_permute_gated.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=core_grid,
        compile_time_args=writer_ct, runtime_args=writer_rt,
        common_runtime_args=[0, 0], config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=str(kdir / "compute_reblock_permute_gated.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=core_grid,
        compile_time_args=[P_CB, G_CB, SIG_CB, MUL_CB, OUT_CB, int(skip_sigmoid), gran],
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                            fp32_dest_acc_en=False))
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    return {"pd": pd, "kernels": [reader, writer, compute], "cbs": cbs, "cores": core_grid,
            "num_groups": num_groups}


def run_once(ttnn, e, x, out, p_slice, g_slice):
    pd = e["pd"]
    pd.kernels[0].common_runtime_args = [x.buffer_address(), p_slice // TILE, g_slice // TILE, 0]
    pd.kernels[1].common_runtime_args = [out.buffer_address(), 0]
    return ttnn.generic_op([x, out], pd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--cw", type=int, default=512)
    ap.add_argument("--slice-c", type=int, default=128)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ablate.json"))
    a = ap.parse_args()

    from tt_bio import runtime
    for k, v in runtime.host_thread_cap_env(1, host_threads=8).items():
        os.environ.setdefault(k, v)
    import torch, ttnn
    from tt_bio import tenstorrent as TT

    device = TT.get_device()
    grid = device.compute_with_storage_grid_size()
    arch = str(device.arch()) if hasattr(device, "arch") else "?"
    print(f"device grid {grid.x}x{grid.y} = {grid.x*grid.y} cores, arch {arch}", flush=True)

    N, Cw, sc = a.n, a.cw, a.slice_c
    torch.manual_seed(0)
    xt = torch.randn(1, N, N, Cw, dtype=torch.float32).bfloat16()
    x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, sc, N, N]), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    p_slice, g_slice = 2 * sc, 0

    entries, notes = {}, {}
    for name in a.arms.split(","):
        kdir, gran, pg, note = materialise(name)
        e = build(ttnn, device, x, out, kdir, gran, pg, skip_sigmoid=(name == "nosigmoid"))
        entries[name], notes[name] = e, note
        run_once(ttnn, e, x, out, p_slice, g_slice)          # JIT + program cache
        ttnn.synchronize_device(device)
        print(f"  compiled {name:12s} groups={e['num_groups']}  {note}", flush=True)

    # A/A floor: `base` is timed twice per round under two independent labels.
    labels = list(entries) + ["base_aa"]
    times = {k: [] for k in labels}
    for rnd in range(a.rounds):
        for lab in labels:
            e = entries["base" if lab == "base_aa" else lab]
            for _ in range(3):
                run_once(ttnn, e, x, out, p_slice, g_slice)
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            for _ in range(a.reps):
                run_once(ttnn, e, x, out, p_slice, g_slice)
            ttnn.synchronize_device(device)
            times[lab].append((time.perf_counter() - t0) / a.reps * 1e6)
        print(f"  round {rnd}: " + "  ".join(f"{k}={st.median(v):.1f}" for k, v in times.items()),
              flush=True)

    base = st.median(times["base"])
    res = {}
    for lab in labels:
        v = times[lab]
        res[lab] = {"us": st.median(v), "all": v, "spread": max(v) / min(v),
                    "delta_us": st.median(v) - base, "ratio": base / st.median(v),
                    "note": notes.get(lab.replace("_aa", ""), ""), "wrong_data": lab in WRONG}
    aa = res["base_aa"]["us"] / base
    print(f"\nA/A floor  base vs base_aa = {aa:.5f}x  ({res['base_aa']['delta_us']:+.1f} us)")
    print(f"{'arm':13s} {'us':>9s} {'d(wall)':>9s} {'x':>8s} {'spread':>7s}  note")
    for lab in sorted(res, key=lambda k: res[k]["us"]):
        r = res[lab]
        print(f"{lab:13s} {r['us']:9.1f} {r['delta_us']:+9.1f} {r['ratio']:8.4f} "
              f"{r['spread']:7.4f}  {'[WRONG DATA] ' if r['wrong_data'] else ''}{r['note']}")

    # Parity: every arm that is not timing-only must reproduce `base` on device, and `base` itself
    # must reproduce the two-op reference. Bit-exactness is measured because it is free, not
    # because it is the bar.
    run_once(ttnn, entries["base"], x, out, p_slice, g_slice)
    ttnn.synchronize_device(device)
    base_out = ttnn.to_torch(out).float().clone()
    hostref = torch.chunk(xt, 4, dim=-1)
    hostref = (hostref[p_slice // sc].float() * torch.sigmoid(hostref[g_slice // sc].float())
               ).permute(0, 3, 1, 2).contiguous()
    par = {"base_vs_fp32_host": {"max_abs_diff": (base_out - hostref).abs().max().item()}}
    print(f"parity base vs fp32 host reference: max|d| = {par['base_vs_fp32_host']['max_abs_diff']:.6g}")
    for lab in entries:
        if lab in WRONG or lab == "base":
            continue
        run_once(ttnn, entries[lab], x, out, p_slice, g_slice)
        ttnn.synchronize_device(device)
        got = ttnn.to_torch(out).float()
        d = (got - base_out).abs().max().item()
        rel = (got - hostref).abs().max().item()
        par[lab] = {"max_abs_diff_vs_base": d, "bit_exact_vs_base": bool(d == 0.0),
                    "max_abs_diff_vs_fp32_host": rel}
        print(f"parity {lab:12s} vs base max|d| = {d:.6g}"
              f"{'  BIT-EXACT' if d == 0.0 else ''}   vs fp32 host = {rel:.6g}")

    meta = {"grid": [grid.x, grid.y], "cores": grid.x * grid.y, "arch": arch,
            "shape": {"N": N, "Cw": Cw, "slice_c": sc}, "reps": a.reps, "rounds": a.rounds,
            "num_groups": entries["base"]["num_groups"], "aa_floor": aa, "results": res, "parity": par}
    Path(a.out).write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
