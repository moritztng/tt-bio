# b2z2-everything-union-wh — PREDICTION, written before the first fold

Written 2026-09-13, branch `wk/b2z2-everything-union-wh` @ `c884a4d19`, before any device
process of this row started. Not edited after. The composition (five commits off
`origin/main` `84da2a49b`) and `perf/b2z2_everything/lever_overlap.py`'s output are already
committed; nothing below has been measured by this row.

## The population I am composing

Nine levers, not eleven. Two rows of the brief's table are not levers:

* `wk/b2z2-step-program-fusion` is an ancestor of `-step-fusion-next-sites`, `-atom-window-next`,
  `-step-binaryng-fusion` and `-step-matmul-group`, and the lever it contributes is
  `TT_BIO_ATOM_KEY_WINDOW`, which gathers the wrong atoms. Dropped.
* `wk/b2z2-step-binaryng-fusion`'s `TT_BIO_MAC_FUSE` is its own row's NO-GO, 1.6 % slower.
  Dropped.

| arm member | flag | stage | single, as published |
|---|---|---|---|
| SHG | `TT_BIO_ATOM_SHIFT_GATHER` | sampler | 1.06391x WH step |
| KVP | `TT_BIO_ATOM_KV_PREPROJ` | sampler | 1.00947x WH step marginal on SHG |
| L1 | `TT_BIO_ATOM_L1` | sampler | 1.08425x WH step |
| SDPAQ | `TT_BIO_SDPA_GRID_Q_CHUNK` | sampler | 1.01831x WH step |
| QKVG | `TT_BIO_TRIATT_FUSED_QKVG` | trunk | 1.01459x on the block |
| PWA | `TT_BIO_PWA_RESIDENCY` | MSA | 1.02603x on `MSALayer` |
| COND | `TT_BIO_DEVICE_CONDITIONING` | host | in the 1.06495x |
| ZINIT | `TT_BIO_DEVICE_ZINIT` | host | in the 1.06495x |
| CONF | `TT_BIO_DEVICE_CONFIDENCE` | host | in the 1.06495x |

## P1 — the fold ratio of the union

Built by Amdahl on the WH fold shares this campaign has already measured: sampler 24.9 % of a
41.07 s fold, MSA track 8.82 %, and the host trio's 1.06495x measured directly on a WH fold.

    sampler   1.08702x in-fold (SHG+KVP+L1, measured in a fold) x ~1.018 (SDPAQ)  = 1.1066x
              on 24.9 %                                                          = 1.0246x fold
    trunk     1.01459x on the block, block ~50 % of the fold                      = 1.0072x fold
    MSA       1.02603x on the layer, track 8.82 %                                 = 1.0022x fold
    host      measured on the fold                                                = 1.0650x fold
    product                                                                       = 1.1015x

**Predicted union fold ratio: 1.085x, band 1.06x - 1.11x.** The point sits below the product
because I predict a discount (P2).

## P2 — the additivity discount

**Predicted 1.5 %, band 0 - 4 %.** The four stages are disjoint code — `lever_overlap.py` says
so at both granularities for every cross-stage pair — so the only route to a discount is a
shared resource, not a shared line. Two candidates: the three host levers remove host work that
overlaps device work the sampler levers also shorten, and `TT_BIO_ATOM_L1` competes for the same
L1 the PWA residency lever wants, in different stages of the same fold.

**Pre-registered falsifier (the brief's): if the measured union is more than 5 % below the
product of the surviving singles, the campaign's per-stage levers are competing and every
stacked projection in `CLOSING.md` is too high.** I predict it does NOT fire.

## P3 — which levers do not survive composition

The A/A floor on a paired WH fold in this checkout was 1.01145x (n=6) for the sampler row. Any
lever whose fold contribution is under that is not individually readable, whatever it is worth.

* **QKVG will not be readable on the fold.** 1.01459x on the block is about 0.7 % of the fold,
  under the floor. It survives as a member of the union, not as a fold lever.
* **PWA will not be readable on the fold.** ~0.2 % of the fold, well under the floor.
* **SDPAQ marginal.** ~0.45 % of the fold on Amdahl. Not readable alone; its step number is.
* **The host trio and the sampler union both clear the floor** and are the only two arms whose
  fold ratio I expect to be readable in isolation.

I predict no lever LOSES: no arm reads below 1.0 outside the floor.

## P4 — the two hash bars

* **BX (all six bit-exact levers) writes the base CIF byte for byte**, one sha256 across every
  rep, because each member is `torch.equal` on its own leg. If it does not, one member is not
  bit-exact in composition and the per-arm legs say which. I predict it holds.
* **ALL does NOT write the base CIF.** Three of its members move atoms in bf16. An identical
  hash on both arms here would mean the toggle never reached the code, not that the union is
  free.
* **Sharper, and the one I most expect to be told I am wrong about: ALL writes the same CIF as
  HOSTONLY, byte for byte.** Bit-exact levers change no structure, so composing six of them onto
  the host trio should move nothing at all. If ALL and HOSTONLY differ, bit-exactness does not
  compose through the host path and that is a finding larger than the ratio.

## P5 — parity

The host union alone measured **0.295206 Å at 298 aa**, below `z_init`'s 0.298180 Å alone
because two bf16 roundings of the same pair tensor partially cancel. If P4's third clause holds,
**the union's parity IS the host union's parity, exactly** — not near it, the same number, at
both 298 and 512 aa. Predicted 512 aa worst per-pseudo-domain all-atom under the 0.60 Å bar.

## P6 — the per-stage walls

Predicted stage ratios of ALL against base: sampler **1.10x - 1.12x**, trunk **1.01x - 1.03x**
(QKVG plus whatever `TT_BIO_DEVICE_ZINIT` removes from the trunk's input), confidence
**1.15x or better** (the pair assembly moves onto the device), diffusion conditioning
**1.3x or better** (`TT_BIO_DEVICE_CONDITIONING` deletes a 201 MB download and an upload).
The conditioning and confidence walls are small in seconds, which is why the fold ratio is
dominated by the sampler and the trunk however large those two ratios come back.
