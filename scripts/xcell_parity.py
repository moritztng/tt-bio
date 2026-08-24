"""X-Cell device parity: `tt_bio.xcell` (ttnn) against `tt_bio.xcell_reference` (torch).

Both take the SAME `state_dict`, so this scores the port's arithmetic and nothing else. There are
no trained X-Cell weights in existence (see `xcell_reference`), so these numbers say the device
reproduces our reference and say NOTHING about biology. No accuracy claim follows from them.

Run on the assigned card:

    PYTHONPATH=. TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 \
      python3 scripts/xcell_parity.py

What it checks, in the order a divergence is easiest to localise: the input encoding (A2-A5), the
six prior projections and the assembled context (A6-A11), one self-attention block and one
cross-attention block, the full forward, the full forward with a prior source absent, then the
4-step refinement loop **per step** rather than at step 0 only, and finally the same forward at
ragged gene lengths. That last group is the point: the gene axis runs at its true length under a
bias-free SDPA, so raggedness has to be shown harmless at model level, not just per op.
"""
import sys, torch, ttnn
from tt_bio.tenstorrent import get_device
import tt_bio.xcell_reference as R
import tt_bio.xcell as T

torch.manual_seed(0)

def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if a.numel() != b.numel():
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    d = (a.norm() * b.norm())
    return float((a @ b) / d) if float(d) > 0 else 1.0

_SCORES = []


def report(tag, ref, got, bar=0.99):
    p = pcc(ref, got)
    _SCORES.append((tag.strip(), p))
    err = float((ref.float() - got.float()).abs().max())
    rel = err / max(float(ref.float().abs().max()), 1e-9)
    ok = "PASS" if p > bar else "FAIL"
    print(f"  {ok}  {tag:34s} PCC {p:.6f}  maxerr {err:.5f}  rel {rel:.5f}")
    return p > bar

D, G, N = 128, 96, 4
cfg = R.XCellConfig(d_model=D, n_layers=6, n_heads=4, vocab_size=300, max_genes=G)
print(f"config: d={D} L={cfg.n_layers} H={cfg.n_heads} block={cfg.block} "
      f"head={cfg.output_head} cross@{cfg.cross_attn_layers} G={G} N={N}")

ref = R.XCell(cfg).eval()
sd = ref.state_dict()

values = torch.rand(N, G) * 6
tokens = torch.randint(0, cfg.vocab_size, (N, G))
pmask = torch.randint(0, 2, (N, G))
priors = {n_: torch.randn(N, d) for n_, d, _ in R.PRIOR_SOURCES}
ptok = torch.randint(0, cfg.vocab_size, (N,))
missing = torch.zeros(N, 6, dtype=torch.bool)

dev = get_device()
tt = T.XCell(cfg, sd)
ok = True

# ---- component 1: input embedding (A2-A5)
with torch.no_grad():
    r_h = ref.embed(values, tokens, pmask)
    r_raw = ref.embed.raw_gene_embedding(tokens)
g_h, g_raw = tt.model.embed(tt._up(values.reshape(N, G, 1)), tt._ids(tokens), tt._ids(pmask))
ok &= report("A2-A5 input embedding", r_h, ttnn.to_torch(g_h))
ok &= report("A2 raw gene embedding", r_raw, ttnn.to_torch(g_raw))

# ---- component 2: the six prior projections + context (A6-A11)
with torch.no_grad():
    r_ctx = ref.perturbation_context(priors, ptok)
g_ctx = tt.model.context(
    {n_: tt._up(priors[n_].reshape(N, 1, d)) for n_, d, _ in R.PRIOR_SOURCES},
    tt._ids(ptok.reshape(N, 1)))
ok &= report("A6-A11 context (6 real tokens)", r_ctx, ttnn.to_torch(g_ctx)[:, :6])
pad_max = float(ttnn.to_torch(g_ctx)[:, 6:].abs().max())
print(f"        context tile padding max abs = {pad_max:.6f} (must be 0, it is masked anyway)")

# ---- component 3: one block, then the full stack
bias_t = tt._up(T.cross_bias(N, G + 1, missing))
for i in (0, 2):
    with torch.no_grad():
        r_b = ref.blocks[i](r_h, r_ctx, missing)
    g_b = tt.model.blocks[i](g_h, g_ctx, bias_t)
    kind = "cross" if ref.blocks[i].cross is not None else "self "
    ok &= report(f"block[{i}] ({kind})", r_b, ttnn.to_torch(g_b))

# ---- component 4: full forward
with torch.no_grad():
    r_out = ref(values, tokens, pmask, priors, ptok, missing)
g_out = tt.forward(values, tokens, pmask, priors, ptok, missing)
ok &= report("FULL forward (all 6 layers)", r_out, g_out)

# ---- component 5: a missing prior source must change the answer the same way
miss2 = missing.clone(); miss2[:, 1] = True
with torch.no_grad():
    r_m = ref(values, tokens, pmask, priors, ptok, miss2)
g_m = tt.forward(values, tokens, pmask, priors, ptok, miss2)
ok &= report("FULL forward, STRING absent", r_m, g_m)
print(f"        device delta from masking a source: {float((g_out-g_m).abs().max()):.5f} "
      f"(reference {float((r_out-r_m).abs().max()):.5f})")

# ---- component 6: the 4-step loop, PCC PER STEP not just step 0
print("  4-step cumulative refinement, shared ranks:")
for steps in (1, 2, 3, 4):
    with torch.no_grad():
        r_p = R.predict(ref, values, tokens, priors, ptok, missing, n_steps=steps,
                        generator=torch.Generator().manual_seed(5))
    g_p = T.predict(tt, values, tokens, priors, ptok, missing, n_steps=steps,
                    generator=torch.Generator().manual_seed(5))
    ok &= report(f"  after {steps} step(s)", r_p, g_p)

# ---- component 7: a ragged gene axis (the axis is run at its true length)
print("  ragged gene axis (no pad, no mask -- the measured-safe route):")
for g2 in (32, 65, 97):
    v2 = torch.rand(N, g2) * 6
    t2 = torch.randint(0, cfg.vocab_size, (N, g2))
    m2 = torch.zeros(N, g2, dtype=torch.long)
    with torch.no_grad():
        r2 = ref(v2, t2, m2, priors, ptok, missing)
    g2o = tt.forward(v2, t2, m2, priors, ptok, missing)
    ok &= report(f"  G={g2} (ragged={g2 % 32 != 0})", r2, g2o)

print("\nRESULT:", "ALL COMPONENTS PASS (PCC > 0.99)" if ok else "SOME COMPONENTS FAILED")
# The gate parses this line rather than the table, so a formatting change cannot silently
# un-gate the leg. worst_component names WHERE the minimum was, which is the first thing
# anyone asks when it drops.
worst = min(_SCORES, key=lambda kv: kv[1]) if _SCORES else ("none", float("nan"))
# Space-free, because the gate parses this line by splitting on whitespace.
print(f"XCELL_PARITY components={len(_SCORES)} min_pcc={worst[1]:.6f} "
      f"worst_component={worst[0].replace(' ', '_')} result={'PASS' if ok else 'FAIL'}")
ttnn.close_device(dev)
sys.exit(0 if ok else 1)
