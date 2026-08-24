"""X-Cell architecture-only performance on one card. No trained weights exist, so this measures
the shape, not the biology.

    python3 scripts/xcell_perf.py              the throughput ladder, printed
    python3 scripts/xcell_perf.py --sdpa-ab    stock vs tuned SDPA program config

The A/B is the measurement that sets the program config in `tt_bio/xcell.py`, so it writes its
result to `perf/xcell/sdpa_ab.json` for the comment there to cite. Both modes run on the card the
process can see:

    PYTHONPATH=. TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 python3 scripts/xcell_perf.py --sdpa-ab
"""
import argparse, json, platform, time
from pathlib import Path

import torch, ttnn
from tt_bio.tenstorrent import get_device
import tt_bio.xcell_reference as R
import tt_bio.xcell as T

torch.manual_seed(0)

REPO = Path(__file__).resolve().parents[1]
LADDER = [(512, 8), (512, 32), (2048, 8), (2048, 32), (4000, 8), (4000, 32)]
# The three gene lengths the program config is claimed over, at the row count the comment cites.
AB_SHAPES = [(512, 8), (2048, 8), (4000, 8)]


def flops(cfg, S, N, C=32):
    d, dff, L = cfg.d_model, cfg.d_ff, cfg.n_layers
    ncross = len(cfg.cross_attn_layers)
    per_self = 4*S*d*d*2 + 2*S*S*d*2 + 2*S*d*dff*2
    per_cross = 2*S*d*d*2 + 2*C*d*d*2 + 2*S*C*d*2 + 2*S*d*dff*2
    h1, h2 = cfg.decoder_hidden
    dec = S*(2*d*h1 + h1*h2 + h2*d + d)*2
    return N * (L*per_self + ncross*per_cross + dec)


def make_inputs(cfg, G, N, priors_t):
    """One fixed input set per shape, so both A/B arms see identical numbers."""
    return dict(
        values=torch.rand(N, G) * 6,
        tokens=torch.randint(0, cfg.vocab_size, (N, G)),
        pert_mask=torch.zeros(N, G, dtype=torch.long),
        priors={n_: priors_t[n_].expand(N, -1).contiguous() for n_, _d, _ in R.PRIOR_SOURCES},
        pert_token=torch.randint(0, cfg.vocab_size, (N,)),
        prior_missing=torch.zeros(N, 6, dtype=torch.bool),
    )


def timed(tt, kw, dev, n_it=5):
    """Warm (compile + first dispatch), then the mean of `n_it` warm forwards."""
    out = tt.forward(**kw)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(n_it):
        out = tt.forward(**kw)
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / n_it * 1e3, out


def run_ladder(cfg, tt, priors_t, dev):
    print(f"{'G genes':>8} {'N cells':>8} {'S=G+1':>7} {'warm ms':>9} {'cells/s':>9} "
          f"{'TFLOP/s':>8} {'GFLOP/fwd':>10}")
    rows = []
    for G, N in LADDER:
        S = G + 1
        try:
            ms, _ = timed(tt, make_inputs(cfg, G, N, priors_t), dev, n_it=3)
            f = flops(cfg, S, N)
            print(f"{G:>8} {N:>8} {S:>7} {ms:>9.1f} {N/(ms/1e3):>9.1f} "
                  f"{f/(ms/1e3)/1e12:>8.2f} {f/1e9:>10.1f}")
            rows.append((G, N, ms, f))
        except Exception as e:
            print(f"{G:>8} {N:>8} {S:>7}   FAILED {type(e).__name__}: {str(e)[:90]}")

    if rows:
        print("\nOne predict() call = 4 diffusion steps. At the published inference defaults")
        print("(n_cells=64, batch_size=8 -> 512 cell-sequences, 4 steps):")
        for G, N, ms, f in rows:
            if N == 32:
                per_cell_ms = ms / N
                total = per_cell_ms * 512 * 4 / 1e3
                print(f"  G={G:>5}: {per_cell_ms:.2f} ms/cell/step -> {total:8.1f} s "
                      f"for 512 cells x 4 steps ({flops(cfg,G+1,512)*4/1e12:.1f} TFLOP)")


def run_sdpa_ab(cfg, tt, priors_t, dev, out_dir):
    """Time the same forward with and without the SDPA program config.

    `tt_bio.xcell` imports the helper by name, so replacing it in that namespace is what a caller
    passing no program_config would get: the stock heuristic. Nothing else about the model moves,
    which is what makes the ratio attributable to this one lever.
    """
    tuned_fn = T._sdpa_program_config_for_lengths
    stock_fn = lambda *_a, **_k: None
    print(f"{'G genes':>8} {'N cells':>8} {'stock ms':>10} {'tuned ms':>10} {'speedup':>8} "
          f"{'max|d|':>9} {'rel':>8}")
    rows = []
    for G, N in AB_SHAPES:
        kw = make_inputs(cfg, G, N, priors_t)
        try:
            T._sdpa_program_config_for_lengths = stock_fn
            stock_ms, stock_out = timed(tt, kw, dev)
            T._sdpa_program_config_for_lengths = tuned_fn
            tuned_ms, tuned_out = timed(tt, kw, dev)
        except Exception as e:
            print(f"{G:>8} {N:>8}   FAILED {type(e).__name__}: {str(e)[:80]}")
            continue
        finally:
            T._sdpa_program_config_for_lengths = tuned_fn
        d = float((stock_out - tuned_out).abs().max())
        rel = d / max(float(stock_out.abs().max()), 1e-9)
        print(f"{G:>8} {N:>8} {stock_ms:>10.2f} {tuned_ms:>10.2f} "
              f"{stock_ms/tuned_ms:>7.2f}x {d:>9.4f} {rel:>8.4f}")
        rows.append(dict(genes=G, cells=N, seq_len=G + 1, stock_ms=round(stock_ms, 3),
                         tuned_ms=round(tuned_ms, 3), speedup=round(stock_ms / tuned_ms, 3),
                         max_abs_diff=round(d, 6), rel_diff=round(rel, 6),
                         stock_max_abs=round(float(stock_out.abs().max()), 6)))

    op_rows = run_sdpa_op_ab(cfg, dev)
    parity = run_parity_unchanged(dev)

    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "sdpa_ab.json"
    dest.write_text(json.dumps(dict(
        what="X-Cell self-attention: the ttnn SDPA program config against the stock heuristic",
        model="X-Cell Mini (d=512 L=12 H=8), architecture-only, random weights",
        host=platform.node(), ttnn=getattr(ttnn, "__version__", "0.68.0"),
        note="warm mean of 5 reps after one warmup; identical inputs on both arms",
        end_to_end=rows,
        isolated_op=op_rows,
        isolated_op_caveat=(
            "one op between two synchronize_device calls, so the absolutes carry the sync and "
            "read high. The RATIO is the number to use: both arms pay the same sync."),
        parity_unchanged=parity), indent=2) + "\n")
    print(f"\nwrote {dest.relative_to(REPO)}")


def run_sdpa_op_ab(cfg, dev, n_it=20):
    """The same A/B on the bare SDPA call, at the shapes the model hands it.

    The end-to-end ratio is diluted by everything that is not attention; this one is what the
    comment in tt_bio/xcell.py claims. One op between two syncs reads high in absolute terms
    (memory `tt-bio-isolated-op-timing-oversync-inflates-cost`), which is why only the ratio is
    quoted: both arms pay the same sync.
    """
    from tt_bio.tenstorrent import _sdpa_program_config_for_lengths as pcfg
    d_head = cfg.d_model // cfg.n_heads
    print(f"\nisolated SDPA op ({cfg.n_heads} heads x {d_head}):")
    print(f"{'S':>8} {'N rows':>8} {'stock ms':>10} {'tuned ms':>10} {'speedup':>8}")
    rows = []
    for G, N in AB_SHAPES:
        S = G + 1
        shape = (N, cfg.n_heads, S, d_head)
        made = [ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev) for _ in range(3)]
        q, k, v = made

        def once(program_config):
            o = ttnn.transformer.scaled_dot_product_attention(
                q, k, v, is_causal=False, scale=d_head ** -0.5, program_config=program_config)
            ttnn.deallocate(o)

        def bench(program_config):
            once(program_config)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for _ in range(n_it):
                once(program_config)
            ttnn.synchronize_device(dev)
            return (time.perf_counter() - t0) / n_it * 1e3

        try:
            stock_ms = bench(None)
            tuned_ms = bench(pcfg(S, S))
        except Exception as e:
            print(f"{S:>8} {N:>8}   FAILED {type(e).__name__}: {str(e)[:70]}")
            for t in made:
                ttnn.deallocate(t)
            continue
        print(f"{S:>8} {N:>8} {stock_ms:>10.2f} {tuned_ms:>10.2f} {stock_ms/tuned_ms:>7.2f}x")
        rows.append(dict(seq_len=S, rows=N, heads=cfg.n_heads, d_head=d_head,
                         stock_ms=round(stock_ms, 3), tuned_ms=round(tuned_ms, 3),
                         speedup=round(stock_ms / tuned_ms, 3), reps=n_it))
        for t in made:
            ttnn.deallocate(t)
    return rows


def run_parity_unchanged(dev):
    """Both arms against the torch reference, at the parity harness's own small config.

    A faster arm that moved the answer would not be a lever, it would be a defect. This is the
    check that the different bf16 accumulation order costs nothing that reaches the output.
    """
    torch.manual_seed(0)
    D, G, N = 128, 96, 4
    cfg = R.XCellConfig(d_model=D, n_layers=6, n_heads=4, vocab_size=300, max_genes=G)
    ref = R.XCell(cfg).eval()
    sd = ref.state_dict()
    kw = dict(values=torch.rand(N, G) * 6,
              tokens=torch.randint(0, cfg.vocab_size, (N, G)),
              pert_mask=torch.randint(0, 2, (N, G)),
              priors={n_: torch.randn(N, d) for n_, d, _ in R.PRIOR_SOURCES},
              pert_token=torch.randint(0, cfg.vocab_size, (N,)),
              prior_missing=torch.zeros(N, 6, dtype=torch.bool))
    with torch.no_grad():
        want = ref(kw["values"], kw["tokens"], kw["pert_mask"], kw["priors"],
                   kw["pert_token"], kw["prior_missing"])

    def pcc(a, b):
        a, b = a.float().flatten(), b.float().flatten()
        a, b = a - a.mean(), b - b.mean()
        d = a.norm() * b.norm()
        return float((a @ b) / d) if float(d) > 0 else 1.0

    tuned_fn = T._sdpa_program_config_for_lengths
    tt = T.XCell(cfg, sd)
    out = {}
    for arm, fn in (("stock", lambda *_a, **_k: None), ("tuned", tuned_fn)):
        T._sdpa_program_config_for_lengths = fn
        try:
            out[arm] = round(pcc(want, tt.forward(**kw)), 6)
        finally:
            T._sdpa_program_config_for_lengths = tuned_fn
    print(f"\nparity vs torch reference (d={D} L={cfg.n_layers} G={G} N={N}): "
          f"stock PCC {out['stock']:.6f}, tuned PCC {out['tuned']:.6f}")
    out["config"] = f"d={D} L={cfg.n_layers} H={cfg.n_heads} G={G} N={N}"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sdpa-ab", action="store_true",
                    help="measure the SDPA program config against the stock heuristic")
    ap.add_argument("--out", type=Path, default=REPO / "perf" / "xcell",
                    help="directory for the JSON artifact (default: perf/xcell)")
    args = ap.parse_args()

    cfg = R.XCellConfig(vocab_size=19400)   # Mini: d=512 L=12 H=8 cross@(2,5,8,11)
    print(f"X-Cell Mini shape: d={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads} "
          f"d_ff={cfg.d_ff} cross@{cfg.cross_attn_layers} head={cfg.output_head}")
    print("ARCHITECTURE-ONLY: random weights, no trained checkpoint exists.\n")

    dev = get_device()
    try:
        ref = R.XCell(cfg).eval()
        tt = T.XCell(cfg, ref.state_dict())
        priors_t = {n_: torch.randn(1, d) for n_, d, _ in R.PRIOR_SOURCES}
        if args.sdpa_ab:
            run_sdpa_ab(cfg, tt, priors_t, dev, args.out)
        else:
            run_ladder(cfg, tt, priors_t, dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
