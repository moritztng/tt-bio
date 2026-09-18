#!/usr/bin/env python3
"""c13-transfer-audit: every lever of record with a pre-registered prediction AND a measurement.

R = measured / predicted in the row's OWN pre-registered currency.
Ratios are converted to (ratio - 1) first, because a 1.02x lever predicted at 1.01x is 2x wrong,
not 1.01x wrong. Where a band and a point are both registered, the point drives R and the band
drives IN_BAND. Where only a band exists, its midpoint drives R.

KIND (the screen that produced the prediction):
  replay      replay/trace gap read off a capture between two replay configurations
  roof        roofline headroom: an efficiency fraction against a roof
  count       source-derived count model: programs/calls/bytes x a constant rate
  op_ab       an on-device isolated op or block A/B, extrapolated by call count or Amdahl
  fold_tx     a MEASURED fold ratio transferred to another arch/tree/stack (incl. products of folds)
  novel       a mechanism that does not exist yet, priced on how it ought to behave
LEVEL: where the MEASUREMENT that scores it was taken.
"""
import statistics as st

# slug, lever, kind, pred_point, pred_lo, pred_hi, meas, unit, level, cite, note
R = [
 # ---------------- fold-level measurements ----------------
 ("c12-reblock-delete","fused gated in-projection","novel",1.0062,0.9342,1.2007,-2.7578,"s","fold(op x 560 calls)",
  "c12-reblock-delete.md:786,808,880","sign flip; +2.7578 s ADDED; 7.4369 vs 1.2226 ms/call"),
 ("c10-trace-lever","DiT trace replay","count",3.04,1.5,3.6,-0.0214,"s","fold",
  "c10-trace-lever.md:21-29,37","inside the 0.055 s A/A floor; 219,200 calls x 13.878 us"),
 ("c10-core-grid","core_grid on bare sites (inherited screen)","replay",4.908,None,None,0.1446,"s","fold",
  "c10-core-grid.md:13-17,57-60,134","c10-fold-census measured it between two of its own replay configs"),
 ("c10-core-grid","core_grid on bare sites (own prereg)","count",0.2,0.0,0.4,0.1446,"s","fold",
  "c10-core-grid.md:11-16","read the source: only 5 live bare sites, not 18"),
 ("c10-qchunk-sign","SDPA q-chunk rule, 512 aa","op_ab",0.045,None,None,0.0643,"s","fold",
  "c10-qchunk-sign.md:29,47","CPU replay of the rule + b2z2_qchunk BH op census"),
 ("c10-qchunk-sign","SDPA q-chunk rule, 298 aa","op_ab",0.030,0.020,0.050,0.8288,"s","fold",
  "c10-qchunk-sign.md:31-32,48","row's own words: 'wrong by 28x and it is the finding'"),
 ("c12-compose-fold","TT_BIO_DIT_COND_HOIST","op_ab",0.2134,None,None,0.2880,"s","fold",
  "c12-compose-fold.md:918-922; c12-cond-hoist-block-timing.md:208","block A/B -> pooled fold"),
 ("c12-compose-fold","TT_BIO_UNFUSED_SILU","op_ab",0.2843,None,None,0.3342,"s","fold",
  "c12-compose-fold.md:918-922; c12-unfused-silu-bh.md:74-82","op A/B x call count -> pooled fold"),
 ("c12-compose-fold","hoist+silu stack","op_ab",0.4977,None,None,0.5756,"s","fold",
  "c12-compose-fold.md:918-920,928-934","EXCEEDED its own additive ceiling by 0.0779 s"),
 ("b2z2-elision-bh-measure","TT_BIO_ATOM_SHIFT_GATHER on BH","fold_tx",0.008,0.005,0.013,0.01985,"ratio-1","fold",
  "b2z2-elision-bh-measure.md:20-31","WH sampler 1.04694x Amdahl'd then discounted by 1/3: 'the discount was backwards'"),
 ("b2z2-zinit-ship","TT_BIO_DEVICE_ZINIT on BH","fold_tx",0.01090,None,None,0.01955,"ratio-1","fold",
  "b2z2-zinit-ship.md:1208-1212","WH fold 1.01090x, 6/6 positive, carried to BH"),
 ("b2z2-trunk-byte-round2-ship","3 trunk byte flags, post-merge re-read","fold_tx",0.01522,None,None,0.01573,"ratio-1","fold",
  "b2z2-trunk-byte-round2-ship.md:19-24","pre-merge BH fold reading re-read after merge"),
 ("b2z2-qchunk-isolated-bh","SDPA q-chunk BH fold (P4)","op_ab",0.0033,0.000,0.020,0.00355,"ratio-1","fold",
  "b2z2-qchunk-isolated-bh.md:34-37","4800 calls x 13.58 us = 0.065 s of 19.78 s, computed before the folds ran"),
 ("b2z2-trunk-fold-ab-bh","all3 trunk byte levers, WH fold (P1)","op_ab",0.0245,0.015,0.032,0.02648,"ratio-1","fold",
  "b2z2-trunk-fold-ab-bh.md:28-34","block 1.05106x x ~53 % Pairformer share; block-to-fold coefficient 1.0"),
 ("b2z2-trunk-fold-ab-bh","qkvg alone, WH fold (P2)","op_ab",0.0073,0.003,0.011,0.00213,"ratio-1","fold",
  "b2z2-trunk-fold-ab-bh.md:29,34-36","inside the 1.00329x floor: does not reach the fold at all"),
 ("b2z2-trunk-bh-leg","all3 trunk byte levers on BH (P1)","fold_tx",0.0200,0.0120,0.0265,0.01522,"ratio-1","fold",
  "b2z2-trunk-bh-leg.md:30-38","WH 1.02648x carried to BH; paired estimator"),
 ("b2z2-trunk-bh-leg","qkvg alone on BH (P2)","fold_tx",0.0010,-0.0025,0.0055,0.00748,"ratio-1","fold",
  "b2z2-trunk-bh-leg.md:32-40","predicted NOT to clear its floor, and did not"),
 ("b2z2-bh-compose-landed","K2 on BH","fold_tx",0.020,0.015,0.025,0.00448,"ratio-1","fold",
  "b2z2-bh-compose-landed.md:25-38","WH->BH; 0.2 % outside a 1.00229x floor"),
 ("b2z2-bh-compose-landed","DST-resident on BH","fold_tx",0.002,None,None,-0.00252,"ratio-1","fold",
  "b2z2-bh-compose-landed.md:25-38","predicted inside the floor; measured negative inside the floor"),
 ("b2z2-bh-compose-landed","MSA ladder on BH","fold_tx",0.040,0.035,0.045,0.04650,"ratio-1","fold",
  "b2z2-bh-compose-landed.md:25-38","perf held; killed later on accuracy at 0.9765 A"),
 ("b2z2-bh-compose-landed","K2+DST+MSA union on BH","fold_tx",0.060,0.055,0.065,0.05985,"ratio-1","fold",
  "b2z2-bh-compose-landed.md:25-38","dead on the point estimate"),
 ("b2z2-bh-union-clean","UNION of three on BH","fold_tx",0.070,0.055,0.085,0.09858,"ratio-1","fold",
  "b2z2-bh-union-clean.md:215-233","row: 'WRONG, low. my discount was imaginary'"),
 ("b2z2-bh-union-clean","HOST on BH","fold_tx",0.045,0.030,0.060,0.04642,"ratio-1","fold",
  "b2z2-bh-union-clean.md:215-233","0.1 % off the point"),
 ("b2z2-bh-union-clean","SILU on BH","fold_tx",0.023,0.015,0.030,0.02423,"ratio-1","fold",
  "b2z2-bh-union-clean.md:215-233",""),
 ("b2z2-bh-union-clean","AKW on BH","fold_tx",0.010,0.000,0.020,0.02744,"ratio-1","fold",
  "b2z2-bh-union-clean.md:215-233","'priced as a sampler-only lever scaled by the sampler share and under-read by 1.7 %'"),
 ("b2z2-bh-union-step","five-lever STACK on BH","fold_tx",0.1148,0.100,0.128,0.12862,"ratio-1","fold",
  "b2z2-bh-union-step.md:63-66","top of the range; later corrected to 1.11982x for a defective gather"),
 ("b2z2-bh-stack-atom","corrected five-lever stack on BH","fold_tx",0.1204,None,None,0.11982,"ratio-1","fold",
  "b2z2-bh-stack-atom.md:88-96","0.05 % apart: the best transfer in the corpus"),
 ("b2z2-everything-union-wh","whole union on WH (P1)","op_ab",0.085,0.060,0.110,0.10789,"ratio-1","fold",
  "b2z2-everything-union-wh.md:'## 7' P1","landed for partly unrelated reasons: it multiplied BLOCK and STEP ratios"),
 ("b2z2-everything-union-wh","sampler stage (P6)","op_ab",0.110,0.100,0.120,0.18816,"ratio-1","fold-stage",
  "b2z2-everything-union-wh.md:'## 7' P6","all four stages beat their band, in the same direction"),
 ("b2z2-everything-union-wh","trunk stage (P6)","op_ab",0.020,0.010,0.030,0.04429,"ratio-1","fold-stage",
  "b2z2-everything-union-wh.md:'## 7' P6",""),
 ("b2z2-everything-union-wh","confidence stage (P6)","op_ab",0.150,0.150,None,0.58106,"ratio-1","fold-stage",
  "b2z2-everything-union-wh.md:'## 7' P6","predicted >=1.15x as a floor"),
 ("b2z2-everything-union-wh","conditioning stage (P6)","op_ab",0.300,0.300,None,1.97452,"ratio-1","fold-stage",
  "b2z2-everything-union-wh.md:'## 7' P6","predicted >=1.3x as a floor"),
 ("b2z2-conf-device-ship","confidence head device port on BH","fold_tx",0.040,0.030,0.050,0.04743,"ratio-1","fold",
  "b2z2-conf-device-ship.md:379-383","WH 1.0128x on a 40.2 s fold -> BH; 8/8 pairs, 45x its floor"),
 ("b2z2-host-device-compose","union of three host->device ports, WH","fold_tx",0.052,0.045,0.058,0.06495,"ratio-1","fold",
  "b2z2-host-device-compose.md:33-49","UNRESOLVABLE: the session's own A/A floor was 5.07 %"),
 ("b2z2-sharded-sampler","token-axis sampler shard (P3)","count",0.150,0.100,0.200,-0.007,"ratio-1","fold",
  "b2z2-sharded-sampler.md:22-35","sign flip at the fold: 0.993x once the mesh tax is paid"),
 ("b2z2-atom-shard-wire","atom-axis window shard (P1)","op_ab",0.130,0.100,0.160,-0.0062,"ratio-1","fold",
  "b2z2-atom-shard-wire.md:22-24","sign flip: 0.99380x untraced"),
 ("b2z2-dual-chip-fold","sharded dual-chip fold","count",2.19,None,None,-0.0317,"s","fold",
  "b2z2-dual-chip-fold.md:242-245,393-394","projected 2.19 s off the trunk; fold measured 0.9984x on 19.7677 s"),
 ("b2z2-orchestrator","wave-2 campaign headline","roof",0.60,0.45,0.75,0.0,"ratio-1","fold",
  "b2z2-orchestrator.md:18-31","'the measured answer is 1.000x'; the shippable stack later read 1.11982x"),
 ("b2z-sampler-steps","200 -> 50 sampling steps on BH*","count",4.878,None,None,3.96,"s","fold",
  "b2z-sampler-steps.md:13-23","*WORK-REMOVAL CHEAT, excluded from every statistic; priced off a per-step cost that predated two shipped levers"),
 # ---------------- step / block / op-level measurements ----------------
 ("b2z2-step-layernorm-fusion","AdaLN shared s-norm","count",0.0390,0.030,0.040,0.03932,"ratio-1","step",
  "b2z2-step-layernorm-fusion.md:21-23","47 x 30.21 us LayerNorm off a committed capture"),
 ("b2z2-step-matmul-group","W1 s-norm fold","count",0.0354,None,None,0.03975,"ratio-1","step",
  "b2z2-step-matmul-group.md:49-51","-1.420 ms/step predicted, -1.590 measured"),
 ("b2z2-step-matmul-group","W2 batched -> 2D matmul","roof",0.700,0.400,1.000,0.0127,"ratio-1","op",
  "b2z2-step-matmul-group.md:52-53","predicted 1.4x-2.0x off an arithmetic-intensity gap; inside the A/A floor"),
 ("b2z2-step-matmul-group","N-stacking the s-projections","count",0.0025,0.000,0.005,-0.1684,"ratio-1","op",
  "b2z2-step-matmul-group.md:54","predicted dead under 1.005x; measured 0.8316x, 1.20x SLOWER"),
 ("b2z2-step-fusion-next-sites","atom L1 residency (P1)","op_ab",0.034,0.020,0.060,0.08461,"ratio-1","step",
  "b2z2-step-fusion-next-sites.md:42-45","the off-fold screen UNDER-priced it; 2.4x the discounted point"),
 ("b2z2-step-fusion-next-sites","unpadded head split (P2)","count",-0.020,-0.050,0.010,-0.2633,"ratio-1","step",
  "b2z2-step-fusion-next-sites.md:50-54","predicted 0.95x-1.01x, measured 0.73666x"),
 ("b2z2-step-program-fusion","indexing-matrix window build","op_ab",3.97,None,None,2.88,"ms/step","step",
  "b2z2-step-program-fusion.md:38-50","off-fold micro-probe over-predicted the step by 38 %"),
 ("b2z2-step-binaryng-fusion","2-chain BinaryNg fusion (P2)","count",0.0435,0.028,0.059,-0.01586,"ratio-1","step",
  "b2z2-step-binaryng-fusion.md:31-37,49-53","sign flip; programs-removed x 9.76 us predicted 0.586/0.293/0.293 ms, returned -0.286/+0.018/-0.477"),
 ("b2z2-step-adaln-sdpa","SDPA bias dtype (P4)","roof",0.025,0.015,0.035,0.0,"ratio-1","op",
  "b2z2-step-adaln-sdpa.md:26-38","P3 byte-bound premise REFUTED and P4 'dead with it'"),
 ("b2z2-step-adaln-sdpa","norm-of-a parallelism (P2)","count",0.0125,0.005,0.020,0.0,"ratio-1","op",
  "b2z2-step-adaln-sdpa.md:28-36","mechanism confirmed, fix REFUTED as unavailable"),
 ("b2z2-step-layout-elision","arm A padded tail","count",0.0173,None,None,0.02549,"ratio-1","step",
  "b2z2-step-layout-elision.md:25-34","'the tail beat its prediction by 47 %'"),
 ("b2z2-step-layout-elision","arm B fused qkv head split","count",0.0623,None,None,0.04124,"ratio-1","step",
  "b2z2-step-layout-elision.md:28-35","fell short: the fused matmul keeps the bias, write scatter ~3 us/call"),
 ("b2z2-atom-window-next","K/V projection onto the atom axis","op_ab",0.582,None,None,0.5967,"ms/step","step",
  "b2z2-atom-window-next.md:22-27","'the prediction was 2.6 % low'; off-fold probe -> step"),
 ("b2z2-dst-resident-fusion","delete the mul_cb L1 round trip","count",0.008,None,None,0.0541,"ratio-1","op",
  "b2z2-dst-resident-fusion.md:10-16","pass model under-predicted a fused deletion by 1.65x (16.2 us predicted, 26.7 saved)"),
 ("b2z2-mcast-operand-build","multicast the shared operand","novel",0.035,0.020,0.100,-0.06163,"ratio-1","block",
  "b2z2-mcast-operand-build.md:15-23","sign flip; 0.93837x on the block, reps spanning 0.14 %"),
 ("b2z2-arrival-skew-attack","two-injector skew absorber","novel",0.12,0.05,0.25,-0.10843,"ratio-1","op",
  "b2z2-arrival-skew-attack.md:18-28","sign flip; 0.89157x / 0.79451x / 0.4523x at k=2/4/per-core"),
 ("b2z2-pairformer-megakernel-build","fused Pairformer trimul megakernel","novel",0.900,None,None,-1.0255,"ms/call","op",
  "b2z2-pairformer-megakernel-build.md:44-46","sign flip; the fusion returned +0.35 to +0.56 ms a call"),
 ("b2z2-fusion-rebuild","fused trimul kernel on BH","fold_tx",0.020,-0.020,0.100,-0.026,"ratio-1","op",
  "b2z2-fusion-rebuild.md:21-24","0.974x at the best config, 0.7627x at the shipped one"),
 ("roof-fuse-trimul-out","g_out -> multiply_ epilogue, L1 product","op_ab",0.05,-0.05,0.10,-0.1488,"ratio-1","op",
  "roof-fuse-trimul-out.md:38-60","sign flip; 0.8512x/0.8568x; prereg was reasoned but not committed to a file first"),
 ("b2z2-trunk-byte-floor","triangle-attention normed-pair read (P2)","count",0.020,0.000,0.050,0.0146,"ratio-1","block",
  "b2z2-trunk-byte-floor.md:24-30","byte model UNDER-prices a deleted read by 1.76x at the realization level"),
 ("b2z2-trunk-byte-round2","rank 2 site (P1)","count",0.0146,0.010,0.017,0.01080,"ratio-1","block",
  "b2z2-trunk-byte-round2.md:24-33","realization 1.31, not the predicted 1.76"),
 ("b2z2-trunk-byte-round2","rank 1b site (P2)","count",0.008,0.004,0.012,0.02491,"ratio-1","block",
  "b2z2-trunk-byte-round2.md:26-33","realization 2.93, 3x its point and 2x its band ceiling"),
 ("b2z2-msa-movement-attack","L1-resident MSA row block (P3)","count",0.055,0.040,0.070,0.02603,"ratio-1","layer",
  "b2z2-msa-movement-attack.md:34-36","'the block is right and the size is right, the size of the win is not'"),
 ("b2z2-twopass-loop-bh","hoist the two-pass loop init (P2)","count",0.030,None,None,-0.0025,"ratio-1","op",
  "b2z2-twopass-loop-bh.md:21-26","falsifier fired: loop_p1 0.9975x"),
 ("b2z-custom-sdpa","exp as the missing SDPA term","roof",0.425,0.300,0.550,0.041,"ratio-1","op",
  "b2z-custom-sdpa.md:103-106","'the obvious story is simply wrong'; predicted 30-55 % of the op, measured 4.1 %"),
 ("b2z-work-removal","MSA depth ladder 1024 -> 64","count",0.675,0.600,0.750,0.422,"ratio-1","layer",
  "b2z-work-removal.md:51-53","depth-linear share predicted 60-75 %, measured 42.2 %"),
 ("c12-fused-eltwise-at-pin","gate sigmoid into the matmul epilogue","count",8.18,None,None,7.84,"us/call","op",
  "c12-fused-eltwise-at-pin.md:57-80","byte model at the key's own measured 192.2 GB/s; fold session DISCARDED by its own floor"),
 ("b2z2-l1-sharded-residency","L1LOCAL / DRAM per-tile cost","count",0.875,0.75,1.00,0.343,"ratio","op",
  "b2z2-l1-sharded-residency.md:11-18","falsifier L1LOCAL <= 0.5 x DRAM fired; 0.343 BH, 0.438 WH"),
 ("b2z2-datum-rate-floor","bare move-only kernel ns/tile","count",52.5,45.0,60.0,30.44,"ns/tile","op",
  "b2z2-datum-rate-floor.md:11-18","the campaign's 71.3 ns/tile assumption was 3.42x off"),
 ("b2z2-grid-qchunk-unify","BH step ratio for the q-chunk rule (P1)","fold_tx",0.0325,0.020,0.045,0.1398,"ratio-1","op",
  "b2z2-grid-qchunk-unify.md:22-34","P1 REFUTED: op reads 1.1398x on BH against WH's 1.3058x, so occupancy gave the right sign and the wrong size"),
]

def rowR(p,m):
    if p in (None,0): return None
    return m/p

print(f"{'slug':34s} {'lever':40s} {'kind':7s} {'pred':>9s} {'meas':>9s} {'unit':8s} {'R':>8s} sign band")
rows=[]
for slug,lev,kind,p,lo,hi,m,unit,level,cite,note in R:
    r=rowR(p,m)
    sign = (p>0) == (m>0) if (p!=0 and m!=0) else (abs(m)<1e-9 and abs(p)<1e-9)
    inband = None
    if lo is not None or hi is not None:
        inband = (lo is None or m>=lo) and (hi is None or m<=hi)
    rows.append(dict(slug=slug,lever=lev,kind=kind,pred=p,lo=lo,hi=hi,meas=m,unit=unit,
                     level=level,cite=cite,note=note,Rv=r,sign=sign,inband=inband))
    print(f"{slug:34s} {lev[:40]:40s} {kind:7s} {p:9.4f} {m:9.4f} {unit:8s} {r:8.3f} "
          f"{'HELD' if sign else 'FLIP'} {'' if inband is None else ('in' if inband else 'OUT')}")

import json
json.dump(rows, open("transfer_table.json","w"), indent=1)
print(f"\nROWS: {len(rows)}")

# ---- appended: roof/occupancy and analytic screens found in the C10/C12 record ----
EXTRA = [
 ("c12-matmul-key-attribution","matmul|1x128x512x512|K=512 prize","roof",0.7794,None,None,0.0584,"s","in-situ fold s",
  "c12-matmul-key-attribution.md:233","envelope priced a configuration the fold does not run; 13.35x down"),
 ("c12-matmul-key-attribution","whole `matmul` class prize","roof",0.9200,None,None,0.1352,"s","in-situ fold s",
  "c12-matmul-key-attribution.md:234","6.80x down; the two methods agree on the FLOOR to 2.5 %, disagree on the BASE"),
 ("c12-matmul-key-attribution","matmul|out=1024x32x512|K=512 prize","roof",0.1067,None,None,0.0,"s","in-situ fold s",
  "c12-matmul-key-attribution.md:246-250","in situ 0.0867 s is already BELOW the envelope's own 0.1738 s achievable"),
 ("b2z-diffusion-utilization","matmul grid height on a 16-tile row axis","roof",0.5042,None,None,0.0054,"s","arith re-derive",
  "b2z2-sampler-ceiling-map.md:14-15,115","94x less: every grid 64..110 cores does 6 tile-blocks per core"),
 ("c10-orchestrator","arithmetic_free_traffic bracket","roof",2.2985,2.074,2.523,4.1310,"s","measured",
  "c10-orchestrator.md:692","underpriced by 1.64-1.99x; a 22 % roof disagreement was treated as the uncertainty"),
 ("c12-host-decomp","F = 3.9830 s read as deletable host","count",3.9830,None,None,0.0960,"s","fold",
  "c12-host-decomp.md:492-498,1129-1134","exposed host is 1.6489-1.6720 s; reducible 0.0000-0.0960 s, ceiling 0.1896 s"),
 ("c10-grid-sweep","host issue per matmul call","count",30.0,15.0,45.0,6.98,"us/call","op",
  "c10-grid-sweep.md:10-27","predicted 15-45 us, measured 6.49-7.47 us and flat in grid"),
]
rows2=[]
for slug,lev,kind,p,lo,hi,m,unit,level,cite,note in EXTRA:
    r=rowR(p,m); sign=(p>0)==(m>0) if (p!=0 and m!=0) else (abs(m)<1e-9 and abs(p)<1e-9)
    inband=None
    if lo is not None or hi is not None: inband=(lo is None or m>=lo) and (hi is None or m<=hi)
    rows2.append(dict(slug=slug,lever=lev,kind=kind,pred=p,lo=lo,hi=hi,meas=m,unit=unit,
                      level=level,cite=cite,note=note,Rv=r,sign=sign,inband=inband))
    print(f"{slug:34s} {lev[:40]:40s} {kind:7s} {p:9.4f} {m:9.4f} {unit:8s} {r:8.3f} {'HELD' if sign else 'FLIP'}")
allrows = rows + rows2
json.dump(allrows, open("transfer_table.json","w"), indent=1)
print(f"TOTAL ROWS: {len(allrows)}")

# ---- appended 2: the trunk/sampler shard rows, whose count models HELD (corrects a first draft) ----
# and a sub-classification of `count`: was the RATE it multiplies measured on the same kernel, or
# borrowed from elsewhere (a launch floor, a clock-immune term, an assumed datum rate)?
EXTRA2 = [
 ("b2z2-trunk-shard-scale-wh","row shard block ratio @2 chips (P1)","count",0.350,None,None,0.3422,"ratio-1","block",
  "b2z2-trunk-shard-scale-wh.md:35-47","rate = a measured compute(N) fit, R2 0.9964"),
 ("b2z2-trunk-shard-scale-wh","row shard block ratio @4 chips (P1)","count",0.700,None,None,0.6404,"ratio-1","block",
  "b2z2-trunk-shard-scale-wh.md:35-47",""),
 ("b2z2-trunk-shard-scale-wh","row shard block ratio @8 chips (P1)","count",0.900,None,None,0.9424,"ratio-1","block",
  "b2z2-trunk-shard-scale-wh.md:35-47",""),
 ("b2z2-trunk-shard-scale-wh","per-block link cost @2 chips (P5)","count",6.2,None,None,6.0,"ms","op",
  "b2z2-trunk-shard-scale-wh.md:39-42","rate = the row's own measured link curve"),
 ("b2z2-trunk-shard-scale-wh","per-block link cost @4 chips (P5)","count",9.3,None,None,8.1,"ms","op",
  "b2z2-trunk-shard-scale-wh.md:39-42",""),
 ("b2z2-trunk-shard-scale-wh","per-block link cost @8 chips (P5)","count",10.8,None,None,8.6,"ms","op",
  "b2z2-trunk-shard-scale-wh.md:39-42",""),
 ("b2z2-shard-replication-attack","b-shard collective cost @2 (P3)","count",1.50,None,None,1.70,"ms","op",
  "b2z2-shard-replication-attack.md:33-48","'the one number the prediction got nearly exactly right'"),
 ("b2z2-shard-replication-attack","b-shard collective cost @4 (P3)","count",2.02,None,None,1.97,"ms","op",
  "b2z2-shard-replication-attack.md:33-48",""),
 ("b2z2-shard-replication-attack","fold route with the b-shard (P4)","count",0.510,None,None,0.4665,"ratio-1","analytic",
  "b2z2-shard-replication-attack.md:33-48","P4 'right in direction and 2.1x too optimistic in size' on the CONSTANT"),
 ("b2z2-shard-replication-attack","the b role's share of the constant (P2)","count",16.0,None,None,10.18,"ms","block",
  "b2z2-shard-replication-attack.md:44-48","the whole trimul constant is 15.6 ms, so b is a third of what the block replicates"),
 ("b2z2-shard-replication-attack","triatt_end's share of the constant (P2)","count",3.0,None,None,6.346,"ms","block",
  "b2z2-shard-replication-attack.md:44-46","the one term P2 got wrong, 2.1x high"),
]
rows3=[]
for slug,lev,kind,p,lo,hi,m,unit,level,cite,note in EXTRA2:
    r=rowR(p,m); sign=(p>0)==(m>0) if (p!=0 and m!=0) else (abs(m)<1e-9 and abs(p)<1e-9)
    inband=None
    if lo is not None or hi is not None: inband=(lo is None or m>=lo) and (hi is None or m<=hi)
    rows3.append(dict(slug=slug,lever=lev,kind=kind,pred=p,lo=lo,hi=hi,meas=m,unit=unit,
                      level=level,cite=cite,note=note,Rv=r,sign=sign,inband=inband))
    print(f"{slug:32s} {lev[:40]:40s} {kind:7s} {p:9.4f} {m:9.4f} {unit:8s} {r:8.3f} {'HELD' if sign else 'FLIP'}")
allrows = rows + rows2 + rows3
json.dump(allrows, open("transfer_table.json","w"), indent=1)
print(f"TOTAL ROWS: {len(allrows)}")
