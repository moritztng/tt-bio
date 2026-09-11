# Boltz-2, 512 aa, on two rooflines

Work per fold, counted on the real modules (`flops_bytes_boltz2.py`), 3 recycles / 4 trunk
passes / 200 sampling steps / 1 diffusion sample, 7168 padded atoms, 35-row MSA:

| phase | TFLOP/fold | share |
|---|---|---|
| pairformer_block | 128.72 | 62.3 % |
| msa_block | 9.52 | 4.6 % |
| confidence_pairformer_block | 4.02 | 1.9 % |
| diffusion_token_layer | 58.95 | 28.5 % |
| atom_transformer_layer | 5.50 | 2.7 % |
| **total** | **206.71** | |

Bytes per fold, bf16, two bounds:

- resident (parameters only, every activation stays on chip): **110.0 GB**, arithmetic intensity **1879 FLOP/byte**
- unfused (every aten op's inputs and outputs): **20.38 TB**, arithmetic intensity **10.1 FLOP/byte**

Machine balance (compute roof / bandwidth roof), the FLOP/byte above which a machine is
compute-bound:

- H200: 206 FLOP/byte
- p300c: 200 FLOP/byte

## Where each arm actually lands

| arm | s/fold | device s | achieved TFLOP/s | % of compute roof | implied GB/s if unfused | % of bandwidth roof |
|---|---|---|---|---|---|---|
| H200 boltz 2.2.1 (arm A) | 7.268 | 6.646 | 31.1 | 3.1 % | 3066 | 64 % |
| H200 BioIR 0.1.0 (arm C) | 2.445 | 2.070 | 99.9 | 10.1 % | 9846 | 205 % |
| p300c tt-bio (published cell) | 23.504 | 23.122 | 8.9 | 10.4 % | 881 | 205 % |

A row whose last column exceeds 100 % is proof by contradiction that the arm does not
move the unfused byte count: no implementation exceeds its own bandwidth roof.

## Kernel-class split of device time, H200 (census, per fold)

| arm | arithmetic-shaped kernels | traffic-shaped kernels | achieved TFLOP/s over the arithmetic part | % of compute roof |
|---|---|---|---|---|
| H200 boltz 2.2.1 (arm A) | 3147 ms | 2992 ms | 66 | 6.6 % |
| H200 BioIR 0.1.0 (arm C) | 1257 ms | 732 ms | 164 | 16.6 % |

## Roofs

p300c, measured on tt-quietbox2 card 2 by `roofs_bh.py`, Arch.BLACKHOLE, 11x10 compute grid:

- DRAM copy roof (read+write): **396 GB/s**
- DRAM read+write roof (2 reads, 1 write): **430 GB/s**
- dense bf16 square matmul, LoFi: **150.3 TFLOP/s**
- dense bf16 square matmul, HiFi2: **121.8 TFLOP/s**
- dense bf16 square matmul, HiFi4: **86.0 TFLOP/s**

H200 SXM roofs are **ASSERTED, not measured**: 4.8 TB/s HBM3e and 990 TFLOP/s bf16
dense (one eighth of the 7916 TFLOP/s DGX figure). This task was not funded to rent a
GPU, so every H200 percentage above is against a vendor number and is a *lower* bound on
utilisation: a measured roof is always below the datasheet, so the real percentages are
higher than printed.
