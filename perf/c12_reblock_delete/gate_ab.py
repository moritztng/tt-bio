#!/usr/bin/env python3
"""Arm 1a op-level A/B: does the fused gate epilogue beat the DRAM round trip it deletes?

WHAT IS COMPARED, and why it is the decisive number for arm 1a. Both arms are the SAME
`generic_minimal_matmul` at the SAME shape with the SAME two destination tensors and the same
number of device programs. The only difference is `gate`: the gated arm's output stage consumes a
tile-interleaved (value, gate) pair and emits one tile, so it writes HALF the output bytes. The
time difference is therefore the deleted write, and the achieved bandwidth it implies is what
settles the inherited `genop_audit`'s 6 Z against the kernel's own 5 Z addressing -- the
disagreement this row disclosed, and the reason the campaign prices the lever at the conservative
5 Z figure (1.0062 s) rather than 1.0990 s.

It does NOT measure the move side. Arm 1a keeps a plain `reblock_permute` where production runs
`reblock_permute_gated`, and that delta is a separate pair of programs. The fold-level arm is what
prices the whole lever; this prices the half that the prediction's band actually rests on.

PROTOCOL, which is the campaign's and not this script's invention:
  * arms INTERLEAVED rep by rep inside ONE process, because the session is the independent unit
    and a constant sibling load cancels in a paired interleaved A/B but not across sessions
  * an A/A arm every rep: the ungated arm runs TWICE per rep, so the session carries its own floor,
    and a |B - A| smaller than the spread of (A, A') is NOT a result
  * the clock SAMPLED DURING each rep from sysfs `tt_aiclk` (tt-smi is unusable on this box while
    any chip answers 0xffffffff). A rep under ~1200 MHz is reported as an artifact, not a
    regression, and every number is quoted with the clock it was taken at
  * `pair_idle.py` run before AND after every rep. A rep whose board-pair sibling held an fd at
    either end is recorded VOID and excluded, and the void count is reported. benchlock is
    structurally blind to a sibling that never asks for the lock, and a p300c's two chips share one
    board power budget
  * every rep reports cycles as well as seconds, because on this fixture seconds move with the
    clock and cycles do not
"""
import argparse, json, os, statistics, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import ttnn
from tt_bio import mm_generic as G

TILE = 32
REPO = Path(__file__).resolve().parents[2]
KERNEL_DIR = REPO / "tt_bio" / "kernels" / "mm_split"
GUARD = REPO / "perf" / "c12_orchestrator" / "pair_guard" / "pair_idle.py"
ROLES = ("p_a", "g_a", "p_b", "g_b")


def aiclk(card):
    """Sysfs, because tt-smi cannot enumerate this box while a chip reads 0xffffffff."""
    try:
        return int(Path(f"/sys/class/tenstorrent/tenstorrent!{card}/tt_aiclk").read_text().strip())
    except Exception:
        return None


def sibling_busy(card):
    """pair_idle.py exit 1 == the board-pair sibling holds an fd. 2 == it could not tell."""
    if not GUARD.exists():
        return None, "guard missing"
    r = subprocess.run([sys.executable, str(GUARD), "--card", str(card)],
                       capture_output=True, text=True)
    return (r.returncode != 0), r.stdout.strip()


def role_major_cols(group):
    return [(r, j) for r in ROLES for j in range(group)]


def tile_interleaved_cols(group):
    out = []
    for p, g in (("p_a", "g_a"), ("p_b", "g_b")):
        for j in range(group):
            out.append((p, j)); out.append((g, j))
    return out


def permute_w(w, src_order, dst_order, C):
    idx = {b: i for i, b in enumerate(src_order)}
    return torch.cat([w[:, idx[b] * C:(idx[b] + 1) * C] for b in dst_order], dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=512, help="pair-rep side; M = h*h")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--slice-c", type=int, default=128)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--inner", type=int, default=20, help="op invocations per timed block")
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", 0)))
    ap.add_argument("--out", default=None)
    # DIAGNOSTIC AXIS, not a shipping knob. fp32_dest_acc_en is a per-KERNEL compile config, so a
    # gate fused into the matmul inherits whatever the matmul needs. calculate_sigmoid branches on
    # it at compile time and takes the accurate (expensive) path when it is True, where the
    # standalone reblock_permute_gated op sets its own config and gets the cheap one. Turning it
    # off here changes the matmul's accumulation precision and is therefore NOT a candidate
    # configuration -- it exists only to attribute the 6.07x.
    ap.add_argument("--fp32acc", type=int, default=1)
    a = ap.parse_args()

    C = a.slice_c // a.group
    n_blocks = 4 * a.group
    N = n_blocks * C
    M = a.h * a.h
    assert C == TILE, ("this harness assumes one tile per channel block", C)

    torch.manual_seed(0)
    x_t = (torch.randn(1, a.h, a.h, a.k) * 0.5).bfloat16()
    w_major = (torch.randn(a.k, N) * 0.1).bfloat16()
    w_inter = permute_w(w_major, role_major_cols(a.group), tile_interleaved_cols(a.group), C)

    from tt_bio.tenstorrent import get_device, _MM_DEFAULT, COMPUTE_GRID_MAIN
    dev = get_device()
    res = {"shape": {"h": a.h, "k": a.k, "N": N, "M": M, "C": C},
           "card": a.card, "reps": a.reps, "inner": a.inner, "fp32acc": bool(a.fp32acc),
           "protocol": "interleaved A/B/A' in one session, own A/A floor, clock sampled during, "
                       "pair guard before and after each rep, void reps excluded"}
    try:
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=bool(a.fp32acc), packer_l1_acc=True)
        cfg = (_MM_DEFAULT, tuple(COMPUTE_GRID_MAIN))
        res["cfg"] = {"block": list(_MM_DEFAULT), "grid": list(COMPUTE_GRID_MAIN)}

        def dev_t(t):
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                                   memory_config=ttnn.DRAM_MEMORY_CONFIG)

        def dests(n_out_tiles, k):
            return [ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, a.h, a.h, n_out_tiles * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                dev, ttnn.DRAM_MEMORY_CONFIG) for _ in range(k)]

        x = dev_t(x_t)
        wm, wi = dev_t(w_major), dev_t(w_inter)
        # Two destinations on BOTH arms so the only variable is the gate. Ungated writes the full
        # 16 output tiles as 2 x 8; gated writes 8 as 2 x 4.
        out_ref = dests(n_blocks // 2, 2)
        out_gate = dests(n_blocks // 4, 2)
        nw_ref = [n_blocks // 2] * 2
        nw_gate = [n_blocks // 4] * 2

        def run(gate):
            G.generic_minimal_matmul(
                dev, x, wi if gate else wm, out_gate if gate else out_ref, cfg, G.ckc_args(ckc),
                {"MM_DUAL_NOC": 1}, KERNEL_DIR, None, ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                nw_gate if gate else nw_ref, KERNEL_DIR, gate)

        def timed(gate):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(a.inner):
                run(gate)
            ttnn.synchronize_device(dev)
            return (time.perf_counter() - t0) / a.inner

        # Compile and warm BOTH arms before the first timed rep, so neither pays a JIT cost inside
        # a measured block and the descriptor cache is populated for both keys.
        for g in (False, True):
            run(g)
        ttnn.synchronize_device(dev)

        reps = []

        def dump():
            """Write the session as it stands. Called after EVERY rep, because this box wedged
            four jobs in one day and a corpse with per-rep JSON is still a result: compose-fold's
            s3 host-spin wedged at rep 31 of 48 and lost nothing, only because it dumped per fold.
            Accumulating in memory and writing after the loop loses the clocks, the void flags and
            the guard output, and recovering those from printed text is lossy."""
            if not a.out:
                return
            res["reps"] = reps
            res["complete"] = len(reps) == a.reps
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(res, indent=2))

        dump()
        for i in range(a.reps):
            busy0, g0 = sibling_busy(a.card)
            c0 = aiclk(a.card)
            t_a = timed(False)
            t_b = timed(True)
            t_a2 = timed(False)
            c1 = aiclk(a.card)
            busy1, g1 = sibling_busy(a.card)
            void = bool(busy0) or bool(busy1)
            clk = [c for c in (c0, c1) if c]
            reps.append({
                "rep": i, "ref_s": t_a, "gate_s": t_b, "ref2_s": t_a2,
                "aa_spread_s": abs(t_a - t_a2),
                "clock_mhz": clk, "clock_min": min(clk) if clk else None,
                "sibling_busy_before": busy0, "sibling_busy_after": busy1, "void": void,
                "guard_before": g0, "guard_after": g1,
            })
            print(f"rep {i}: ref {t_a*1e3:.4f} ms  gate {t_b*1e3:.4f} ms  "
                  f"A/A |{abs(t_a-t_a2)*1e3:.4f}| ms  clk {clk}  "
                  f"{'VOID (sibling busy)' if void else 'ok'}", flush=True)
            dump()

        res["reps"] = reps
        res["complete"] = True
        good = [r for r in reps if not r["void"]]
        res["void_count"] = len(reps) - len(good)
        if not good:
            res["verdict"] = "NO RESULT -- every rep void, the board-pair sibling was busy"
        else:
            ref = statistics.median(r["ref_s"] for r in good)
            gate = statistics.median(r["gate_s"] for r in good)
            floor = statistics.median(r["aa_spread_s"] for r in good)
            clocks = [r["clock_min"] for r in good if r["clock_min"]]
            res["summary"] = {
                "n_good": len(good), "ref_ms": ref * 1e3, "gate_ms": gate * 1e3,
                "delta_ms": (ref - gate) * 1e3, "aa_floor_ms": floor * 1e3,
                "speedup": ref / gate if gate else None,
                "delta_over_floor": (ref - gate) / floor if floor else None,
                "clock_min_mhz": min(clocks) if clocks else None,
                "gate_mcycles": gate * min(clocks) * 1e6 / 1e6 if clocks else None,
            }
            # The three bars, applied rather than described.
            notes = []
            if clocks and min(clocks) < 1200:
                notes.append(f"ARTIFACT: clock fell to {min(clocks)} MHz (<1200), not a regression")
            if floor and (ref - gate) < 3 * floor:
                notes.append(f"NOT A RESULT: delta {(ref-gate)*1e3:.4f} ms is under 3x this "
                             f"session's own A/A floor {floor*1e3:.4f} ms")
            if gate * 1e3 > 1.1206:
                notes.append(f"KILL: gated op {gate*1e3:.4f} ms/call exceeds the pre-registered "
                             f"1.1206 ms/call pessimistic bound, no tuning rescue")
            res["notes"] = notes
    finally:
        ttnn.close_device(dev)

    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
