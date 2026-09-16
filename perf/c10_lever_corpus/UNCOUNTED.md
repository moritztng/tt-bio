# Uncounted — levers found in the corpus and not priced here

Found by a regex scan for a `TT_BIO_*` flag and a ratio on the same line, over all 1303 markdown
files under `~/.coworker/state` including `b2z/` and `b2z2/`. Forty-two distinct flags came back with
at least one ratio beside them. The ledger prices or explicitly names seventeen. The rest are listed
here so nobody can claim they were dropped.

Reproduce the scan:

```sh
python3 - <<'EOF'
import re, glob
rat = re.compile(r'\b\d\.\d{2,5}x\b')
hits = {}
for p in glob.glob('*.md') + glob.glob('b2z2/*.md') + glob.glob('b2z/*.md'):
    for l in open(p, errors='replace'):
        fs = set(re.findall(r'TT_BIO_[A-Z_0-9]+', l))
        if fs and rat.search(l):
            for f in fs: hits.setdefault(f, set()).add(p)
for f in sorted(hits): print(f, sorted(hits[f])[:2])
EOF
```
(run it from `~/.coworker/state`)

| flag | first source | why it is not priced here |
|---|---|---|
| `TT_BIO_ATOM_AXIS_BUCKET` | `b2x-flag-levers.md` | wave-1 lever, already default-on; its ratios (1.043-1.067x) are pre-b2z2 and on a cell three re-cells ago |
| `TT_BIO_ATOM_HEADS_UNPADDED` | `b2z2-step-fusion-next-sites.md` | readings span 0.73666x to 1.36x; needs its own arbitration pass |
| `TT_BIO_FP32_SOFTMAX_L1_PADDED` | `pxdesign-perf.md` | PXDesign, not Boltz-2 |
| `TT_BIO_FUSE_BIAS_STACKS` | `b2z-host-residual-kill.md` | wave-1 host lever, default-on; ratios 1.00100x-1.05738x across different rulers |
| `TT_BIO_GATE_DST_RESIDENT` | `b2z2-bh-compose-landed.PREDICTION.md` | first source is a PREDICTION file, not a measurement |
| `TT_BIO_GATE_GRANULARITY` | `b2z-levers-default-on.md` | ratios straddle 1.0 (0.99843x-1.00551x), inside any plausible floor |
| `TT_BIO_MSA_DEPTH_LADDER` | `b2z2-bh-compose-landed.PREDICTION.md` | PREDICTION file; the shipped form is `TT_BIO_MSA_LADDER`, which is in the ledger |
| `TT_BIO_PROTENIX_TOKEN_BUCKET` | `pxdesign-af2ig-land.md` | Protenix, not Boltz-2 |
| `TT_BIO_PWA_RESIDENCY` | `b2z2-everything-union-wh.PREDICTION.md` | 1.02603x on `MSALayer`, from a PREDICTION file; distinct from the `PairWeightedAveraging.proj_z` row in the ledger |
| `TT_BIO_SDPA_ADD_GRANULARITY` | `k10-transfer-function.md` | 1.00077x-1.00173x, inside any floor |
| `TT_BIO_SDPA_RAGGED_PAD` | `fused-sdpa-adopt.md` | 2.55x/3.03x are op ratios on the ragged tail, not a fold number |
| `TT_BIO_SDPA_WIDE_K` | `triatt-sdpa-wide-k-envelope-gate.md` | readings straddle 1.0 (0.9104x-1.018x); an envelope gate, not a lever |
| `TT_BIO_SOFTMAX_CKC` | `pxdesign-perf.md` | PXDesign |
| `TT_BIO_TOKEN_BUCKET` | `protenix-opendde-qb2-cell-reanchor.md` | the 3.996x-4.131x figures are recompilation avoidance, not fold work |
| `TT_BIO_TRANSPOSE_L1_RESERVE` | `pxdesign-perf.md` | PXDesign |
| `TT_BIO_TRIATT_ABLATE` | `roof-triatt-build-levers.md` | an ablation switch for measurement, not a shippable lever |
| `TT_BIO_TRIATT_FUSED_HIFI` | `openbind-perf.md`, `rf3-4x-with-accuracy.md` | 2.364x-3.03x on RF3 and OpenBind, default OFF; 1.0019x at 512 aa on Boltz-2, i.e. nothing for this cell |
| `TT_BIO_TRIATT_FUSE_QKV` | `roof-triatt-rate-fix.md` | 1.4982x is an op ratio on the pair; `roof-qkv-sdpa-build` records the prize shrinking on contact with the real trunk |
| `TT_BIO_TRIATT_QKV_GATE_FUSED` | `b2z2/FINDINGS.md` | 1.0037x-1.0047x, inside any floor |
| `TT_BIO_TRIMUL_MASK_AFTER_MOVE` | `b2x-integrate.md` | wave-1 lever; 1.045x-1.1368x on cells superseded by three re-cells and at least one counter defect (`b2x-baseline-attrib`'s 1.178x byte undercount) |
| `TT_BIO_TRIMUL_MM_TRANSPOSE` | `util-op-deletes.md` | 1.00613x, inside any floor |
| `TT_BIO_TRIMUL_OUT_FUSED` | `perfwar-qb1-rebaseline-and-land.md` | qb1 rebaseline, a different part and a superseded cell |
| `TT_BIO_TRUNK_MATH_FIDELITY` | `wh-perf-opendde.md` | OpenDDE, and a fidelity knob is an accuracy trade rather than a work deletion |

Three further classes of uncounted material, named rather than dropped:

* **`state/b2z2/FINDINGS.md` (461 kB), `state/b2z2/CONTEXT.md` (72 kB), `state/b2z/FINDINGS.md`
  (190 kB) were regex-scanned but not read.** Every ledger row is sourced from a document that was
  read end to end. Anything that exists only in those three files is uncounted.
* **The 266 directories under `perf/` were not opened.** The ledger cites the state documents that
  report those runs, not the JSON they wrote. A row whose only record is a `perf/*/out/*.json` and
  never reached a state doc is uncounted.
* **Levers on other models** — RF3, OpenBind, OpenFold3, Protenix-v2, PXDesign, OpenDDE, ESMC — are
  out of scope for a Boltz-2 512 aa cycle budget and are not counted, except where a Boltz-2 lever's
  accuracy spend was paid by one of them (`TT_BIO_UNFUSED_SILU`).
