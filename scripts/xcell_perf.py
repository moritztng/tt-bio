"""X-Cell architecture-only performance on one card. No trained weights exist, so this measures
the shape, not the biology."""
import sys, time, torch, ttnn
from tt_bio.tenstorrent import get_device
import tt_bio.xcell_reference as R
import tt_bio.xcell as T

torch.manual_seed(0)

def flops(cfg, S, N, C=32):
    d, dff, L = cfg.d_model, cfg.d_ff, cfg.n_layers
    ncross = len(cfg.cross_attn_layers)
    per_self = 4*S*d*d*2 + 2*S*S*d*2 + 2*S*d*dff*2
    per_cross = 2*S*d*d*2 + 2*C*d*d*2 + 2*S*C*d*2 + 2*S*d*dff*2
    h1, h2 = cfg.decoder_hidden
    dec = S*(2*d*h1 + h1*h2 + h2*d + d)*2
    return N * (L*per_self + ncross*per_cross + dec)

cfg = R.XCellConfig(vocab_size=19400)   # Mini: d=512 L=12 H=8 cross@(2,5,8,11)
print(f"X-Cell Mini shape: d={cfg.d_model} L={cfg.n_layers} H={cfg.n_heads} "
      f"d_ff={cfg.d_ff} cross@{cfg.cross_attn_layers} head={cfg.output_head}")
print("ARCHITECTURE-ONLY: random weights, no trained checkpoint exists.\n")

dev = get_device()
ref = R.XCell(cfg).eval()
sd = ref.state_dict()
tt = T.XCell(cfg, sd)
priors_t = {n_: torch.randn(1, d) for n_, d, _ in R.PRIOR_SOURCES}

print(f"{'G genes':>8} {'N cells':>8} {'S=G+1':>7} {'warm ms':>9} {'cells/s':>9} "
      f"{'TFLOP/s':>8} {'GFLOP/fwd':>10}")
rows = []
for G, N in [(512, 8), (512, 32), (2048, 8), (2048, 32), (4000, 8), (4000, 32)]:
    S = G + 1
    values = torch.rand(N, G) * 6
    tokens = torch.randint(0, cfg.vocab_size, (N, G))
    pmask = torch.zeros(N, G, dtype=torch.long)
    priors = {n_: priors_t[n_].expand(N, -1).contiguous() for n_, _d, _ in R.PRIOR_SOURCES}
    ptok = torch.randint(0, cfg.vocab_size, (N,))
    miss = torch.zeros(N, 6, dtype=torch.bool)
    try:
        tt.forward(values, tokens, pmask, priors, ptok, miss)          # compile + warm
        ttnn.synchronize_device(dev)
        n_it = 3
        t0 = time.perf_counter()
        for _ in range(n_it):
            tt.forward(values, tokens, pmask, priors, ptok, miss)
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) / n_it * 1e3
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
ttnn.close_device(dev)
