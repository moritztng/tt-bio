# Predicted landings, written BEFORE the runs that test them

qb1 card 0, grid 13x10, ttnn 0.67.4, benchlock held. Baseline measured this session:
**36.343 s** (n=2, spread 0.177 s), plDDT 0.9284, CIF sha256 `8900eafb9cb2984a`.
A/A noise floor is taken as the 0.177 s spread until the cross-process baseB arm lands.

## L-F, --fast (bfp8 heavy matmuls) -- ALREADY RESOLVED, recorded for the ledger
Predicted before the run: bfp8 halves the stored bytes of the trimul intermediates and the
pair-FFN weights, so on a DRAM-traffic-bound trunk it should be worth 2-5 s. Kill gate 1.0 s.
MEASURED: **39.025 s, a 2.682 s LOSS**, plDDT 0.9284 -> 0.9274, CIF `1962b65ee08e09b2`.
The prediction was wrong and the mechanism is in `_transform_chunk`
(`tenstorrent.py:2250`): under `_FAST_MODE` the channel move is wrapped in
`typecast(bf16)` before and `typecast(bfloat8_b)` after, two extra full-tensor DRAM passes,
and the E6 fused gated move is bypassed completely -- `e6_served` is **0/0** in the fast arm
against **2152/0** in the base arm. So fast mode pays two extra passes AND gives up E6's
measured 1.228x on the same op. VERDICT: NO-GO as shipped.

## L-G, use the card's real core grid
`COMPUTE_GRID_MAIN = (11, 10)` is a module constant (`tenstorrent.py:335`), and
`_pair_proj_program_config` reads it rather than the device, so on this 13x10 card **20 of
130 cores are unused by construction** on every pair-track projection and every
`core_grid=CORE_GRID_MAIN` matmul.

Mechanism-based prediction: the DRAM-bound ops in the trimul (the E6 move at 43 % of the
bandwidth roof, both layer_norms, the gate multiply at 91 %, the transpose, the back move)
gain **nothing** from more cores, because DRAM bandwidth is a chip resource, not a per-core
one. Only the two matmul terms can gain: the in-projection (measured 44 % of the compute roof
and write-serialised, so 18 % more cores shrinks only the compute half) and the channel matmul
(measured 37 % of the compute roof at 64-of-110-core occupancy). Those two are 4.463 of the
15.219 ms trimul. 18 % more cores on at most that share, minus the write term, is 3-6 % of the
trimul and a similar share of the pair FFN's three matmuls.

**PREDICTED LANDING: -0.5 to -1.5 s, i.e. 34.8-35.8 s.** Bit-exact expected: num_cores changes
`per_core_M`, which is an M-row split; the k contraction order is untouched, so every output
row block is the same sum in the same order. If the CIF digest moves, the prediction is wrong
and it is a parity finding, not a perf one.
**KILL GATE: a delta below 0.35 s (2x the A/A spread) is NO-GO.**

## L-H, the device->host crossing rate
The predecessor's census (`esmfold2-3p4x-close.md` section 5.4) measured 242 crossings costing
5.881 s, of which 2.080 s is a blocking read that waits on device compute and is not transfer.
**Pool: ~3.8 s.** Every 268.4 MB (fp32) crossing lands at 0.151-0.153 s = 0.877 GB/s of actual
bf16 link bytes, against a x16 PCIe link, so >90 % of a crossing is not the link.
`TorchWrapper._to_torch` is `torch.Tensor(ttnn.to_torch(x)).to(torch.float32)`: a host-side
untilize of a TILE_LAYOUT tensor on one thread, then a second full host pass to widen.

Every prior pass asked "can this crossing be DELETED" and got 0.15-0.6 s answers. None asked
why the crossing costs what it costs.

Mechanism-based prediction: moving the untilize to the device puts the tile shuffle on 130
Tensix at DRAM bandwidth (268.4 MB of device traffic = 0.63 ms at the 422.9 GB/s roof), leaving
the host a contiguous copy plus the widen.
**PREDICTED: the isolated 268 MB crossing goes 0.153 s -> 0.02-0.05 s, 3-7x.**
Named risk, from memory `ttnn-untilize-single-core-fallback`: device untilize can silently
fall back to one core and be 36x WORSE. The screen measures it instead of assuming it.
**KILL GATE: below 2.0x on the isolated crossing is NO-GO.** Bit-exactness is checked with
`torch.equal` in the screen, not argued: untilize is a pure layout change and bf16 -> fp32 is
lossless (memory `bf16-roundtrip-bit-exact`).
