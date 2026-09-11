# Boltz-2, 512 aa, on two rooflines

Work per fold, counted on the real modules (`flops_bytes_boltz2.py`), 3 recycles / 4 trunk
passes / 200 sampling steps / 1 diffusion sample, 35-row MSA:

| phase | TFLOP/fold | share |
|---|---|---|
| pairformer_block | 128.72 | 63.0 % |
| msa_block | 9.52 | 4.7 % |
| confidence_pairformer_block | 4.02 | 2.0 % |
| diffusion_token_layer | 58.95 | 28.8 % |
| atom_transformer_layer | 3.14 | 1.5 % |
| **total** | **204.35** | |

Bytes per fold, bf16, two bounds:

- resident (parameters only, every activation stays on chip): **110.0 GB**, arithmetic intensity **1857 FLOP/byte**
- unfused (every aten op's inputs and outputs): **19.87 TB**, arithmetic intensity **10.3 FLOP/byte**

Machine balance (compute roof / bandwidth roof), the FLOP/byte above which a machine is
compute-bound:

- H200: 206 FLOP/byte
- p150a: 196 FLOP/byte

## Where each arm actually lands

| arm | s/fold | device s | achieved TFLOP/s | % of compute roof | implied GB/s if unfused | % of bandwidth roof |
|---|---|---|---|---|---|---|
| H200 boltz 2.2.1 (arm A) | 7.268 | 6.646 | 30.8 | 3.1 % | 2990 | 62 % |
| H200 BioIR 0.1.0 (arm C) | 2.445 | 2.070 | 98.7 | 10.0 % | 9602 | 200 % |
| p150a tt-bio (published cell) | 23.504 | 23.122 | 8.8 | 10.3 % | 859 | 196 % |

A row whose last column exceeds 100 % is proof by contradiction that the arm does not
move the unfused byte count: no implementation exceeds its own bandwidth roof.

## Kernel-class split of device time, H200 (census, per fold)

| arm | arithmetic-shaped kernels | traffic-shaped kernels | achieved TFLOP/s over the arithmetic part | % of compute roof |
|---|---|---|---|---|
| H200 boltz 2.2.1 (arm A) | 3147 ms | 2992 ms | 65 | 6.6 % |
| H200 BioIR 0.1.0 (arm C) | 1257 ms | 732 ms | 163 | 16.4 % |

## Roofs

p150a, measured on qb1 card 2 by `roofs_p150a.py`, Blackhole p150a, 11x10 grid:

- DRAM copy roof (read+write): **398 GB/s**
- DRAM read+write roof (2 reads, 1 write): **438 GB/s**
- dense bf16 square matmul, LoFi: **150.8 TFLOP/s**
- dense bf16 square matmul, HiFi2: **122.0 TFLOP/s**
- dense bf16 square matmul, HiFi4: **86.1 TFLOP/s**

H200 SXM roofs are **ASSERTED, not measured**: 4.8 TB/s HBM3e and 990 TFLOP/s bf16
dense (one eighth of the 7916 TFLOP/s DGX figure). This task was not funded to rent a
GPU, so every H200 percentage above is against a vendor number and is a *lower* bound on
utilisation: a measured roof is always below the datasheet, so the real percentages are
higher than printed.
