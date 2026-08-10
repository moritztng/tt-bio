# z-size-robustness pass 2 — predictions, written before any device was opened

Two competing models of the L1 term. Both agree at tile-aligned sizes and disagree by one token at
417, which is why 416/417 is the experiment rather than 416/432.

- **M-logical (the doc's pass-1 model).** The held bytes per bank are `N*N*256*2/130`, the logical
  volume, the same quantity `_l1_memory_config_if_it_fits` tests. 417 holds ~685 kB/bank, one token
  more than 416, so 417 folds and the edge is wherever the L1 term crosses.
- **M-padded (new this pass).** The allocator serves the TILE-PADDED volume, `ceil(N/32)*32` squared.
  416 -> 13 tiles, 417 -> 14 tiles, so 417 holds exactly what 448 holds, 790 465 B/bank, and the CB
  side jumps with it because `Nt` goes 13 (prime, `in0_block_w=1`) to 14 (=2x7, `in0_block_w=7`).
  Under M-padded 417 crashes and 416 does not: **the lower edge is one token wide, at 416/417.**

Pass-1 evidence already favours M-padded but does not settle it: at 506 the measured held is
1 039 872 B/bank, against 1 032 444 predicted from the padded volume (0.7 %) and 1 020 345 from the
logical volume (1.9 %).

| run | arm | prediction | which model it discriminates |
|---|---|---|---|
| 352 | on | ok | neither, minimum-list coverage |
| 384 | on | ok | neither, minimum-list coverage |
| 416 | on | **ok** | both models agree |
| **417** | on | **FAIL, CB clash, core range [(0,0)-(2,9)]** | **M-padded. M-logical says ok** |
| 432 | on | FAIL | both |
| 464 | on | FAIL | both |
| 480 | on | FAIL | both |
| 496 | on | FAIL | both |
| 416 | on, --fast | **FAIL** | H3: `_trimul_l1_max_seq` is 352 default and 704 fast on 13x10, so the fast arm additionally holds the trimul chunks in L1 at a size the default arm folds |
| 384 | tmc_l1 | FAIL | H1's second falsifier: forcing MORE L1 residency at a passing size |
| 448 | on, via the real CLI | clean non-zero exit, message intact | deliverable 6 |
