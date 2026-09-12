# BEST-CONFIG-TABLE — Boltz-2 512 aa, every knob on every instance above 0.5 % of replayed device time

whglx card 1, **Wormhole**, compute grid 8x9, ttnn 0.68.0. Every number is a paired interleaved
ratio: incumbent, arm, incumbent, arm, ... in one process, each arm scored against the median of
the two incumbent runs that bracket it. `A/A` is the spread of the incumbent runs themselves and
is the noise floor a ratio has to clear. `max_abs` is against the incumbent's own output on
identical seeded operands. Winners must be re-measured on Blackhole before anything lands.

## The whole table

| instance | op | operands | ms/fold | knob | ratio | max_abs |
|---|---|---|---|---|---|---|
| `PairformerLayer#002` | matmul | `1x16x512x128 , 128x512` | 4652 | `fidelity=LoFi` | 1.059x | 1.56e-02 |
|  | | |  | `fidelity=HiFi2` | 1.074x | 3.91e-03 |
|  | | |  | `fidelity=HiFi3` | 1.064x | 3.91e-03 |
|  | | |  | `fp32acc=0` | 1.692x | 7.81e-03 |
|  | | |  | `packerl1=0` | 1.000x |  |
|  | | |  | `dstfull=1` | 1.000x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.683x | 7.81e-03 |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.680x | 7.81e-03 |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.695x | 7.81e-03 |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.682x | 7.81e-03 |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.690x | 7.81e-03 |
|  | | |  | `grid=8x8` | 6.173x | 3.91e-03 |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 7.121x | 7.81e-03 |
|  | | |  | `grid=8x7` | 5.428x | 3.91e-03 |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 6.569x | 7.81e-03 |
|  | | |  | `grid=7x7` | 4.966x | 3.91e-03 |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 6.179x | 7.81e-03 |
|  | | |  | `grid=8x4` | 4.407x | 3.91e-03 |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 5.304x | 7.81e-03 |
|  | | |  | `grid=4x8` | 4.966x | 3.91e-03 |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 6.724x | 7.81e-03 |
|  | | |  | `grid=8x2` | 2.337x | 3.91e-03 |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 2.782x | 7.81e-03 |
|  | | |  | `outbuf=L1` | 0.999x |  |
| | | | | **A/A floor** | **1.2137** | |
| `PairformerLayer#001` | matmul | `1x16x512x128 , 128x512` | 3999 | `fidelity=LoFi` | 1.063x | 1.56e-02 |
|  | | |  | `fidelity=HiFi2` | 1.072x | 3.91e-03 |
|  | | |  | `fidelity=HiFi3` | 1.064x | 3.91e-03 |
|  | | |  | `fp32acc=0` | 1.688x | 7.81e-03 |
|  | | |  | `packerl1=0` | 1.000x |  |
|  | | |  | `dstfull=1` | 0.999x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.683x | 7.81e-03 |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.687x | 7.81e-03 |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.688x | 7.81e-03 |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.679x | 7.81e-03 |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.687x | 7.81e-03 |
|  | | |  | `grid=8x8` | 6.223x | 3.91e-03 |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 7.197x | 7.81e-03 |
|  | | |  | `grid=8x7` | 5.509x | 3.91e-03 |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 6.601x | 7.81e-03 |
|  | | |  | `grid=7x7` | 5.002x | 3.91e-03 |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 6.144x | 7.81e-03 |
|  | | |  | `grid=8x4` | 3.965x | 3.91e-03 |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 4.863x | 7.81e-03 |
|  | | |  | `grid=4x8` | 4.898x | 3.91e-03 |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 6.384x | 7.81e-03 |
|  | | |  | `grid=8x2` | 2.197x | 3.91e-03 |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 2.797x | 7.81e-03 |
|  | | |  | `outbuf=L1` | 1.001x |  |
| | | | | **A/A floor** | **1.0029** | |
| `PairformerLayer#007` | matmul | `1x512x512x128 , 128x128` | 3363 | `fidelity=LoFi` | 1.040x |  |
|  | | |  | `fidelity=HiFi2` | 1.038x |  |
|  | | |  | `fidelity=HiFi3` | 1.020x |  |
|  | | |  | `fp32acc=0` | 1.057x | 5.00e-01 |
|  | | |  | `packerl1=0` | 0.999x |  |
|  | | |  | `dstfull=1` | 1.000x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.056x | 5.00e-01 |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.056x | 5.00e-01 |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.058x | 5.00e-01 |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.058x | 5.00e-01 |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.055x | 5.00e-01 |
|  | | |  | `grid=8x8` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |
|  | | |  | `grid=8x7` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |
|  | | |  | `grid=7x7` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |
|  | | |  | `grid=8x4` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |
|  | | |  | `grid=4x8` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |
|  | | |  | `grid=8x2` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |
|  | | |  | `outbuf=L1` | 1.000x |  |
| | | | | **A/A floor** | **1.0017** | |
| `DiffusionStep#000` | matmul | `1x512x768 , 768x1536` | 2879 | `fidelity=LoFi` | 1.017x |  |
|  | | |  | `fidelity=HiFi2` | 1.021x |  |
|  | | |  | `fidelity=HiFi3` | 1.011x |  |
|  | | |  | `fp32acc=0` | 0.526x |  |
|  | | |  | `packerl1=0` | 0.971x |  |
|  | | |  | `dstfull=1` | 1.000x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 0.526x |  |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 0.545x |  |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 0.532x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 0.527x |  |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 0.531x |  |
|  | | |  | `grid=8x8` | 1.060x | 1.00e+00 |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 1.083x | 2.34e-02 |
|  | | |  | `grid=8x7` | 0.910x |  |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 1.006x |  |
|  | | |  | `grid=7x7` | 0.847x |  |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 0.992x |  |
|  | | |  | `grid=8x4` | 0.791x |  |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 0.838x |  |
|  | | |  | `grid=4x8` | 0.675x |  |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 0.699x |  |
|  | | |  | `grid=8x2` | 0.490x |  |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 0.553x |  |
|  | | |  | `outbuf=L1` | 0.394x |  |
| | | | | **A/A floor** | **1.0426** | |
| `DiffusionStep#017` | matmul | `1x16x512x512 , 1x16x512x64` | 2388 | `fidelity=LoFi` | 1.011x |  |
|  | | |  | `fidelity=HiFi2` | 1.020x |  |
|  | | |  | `fidelity=HiFi3` | 1.004x |  |
|  | | |  | `fp32acc=0` | 1.247x | 1.56e-02 |
|  | | |  | `packerl1=0` | 1.036x |  |
|  | | |  | `dstfull=1` | 0.995x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.259x | 3.12e-02 |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.271x | 1.56e-02 |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.253x | 1.56e-02 |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.312x | 3.12e-02 |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.264x | 1.56e-02 |
|  | | |  | `grid=8x8` | 1.422x | 3.91e-03 |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 1.523x | 1.56e-02 |
|  | | |  | `grid=8x7` | 1.517x | 3.91e-03 |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 1.559x | 1.56e-02 |
|  | | |  | `grid=7x7` | 1.520x | 3.91e-03 |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 1.569x | 1.56e-02 |
|  | | |  | `grid=8x4` | 1.728x | 3.91e-03 |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 1.789x | 1.56e-02 |
|  | | |  | `grid=4x8` | 1.425x | 3.91e-03 |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 1.525x | 1.56e-02 |
|  | | |  | `grid=8x2` | 2.143x | 3.91e-03 |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 2.575x | 1.56e-02 |
|  | | |  | `outbuf=L1` | 1.026x |  |
| | | | | **A/A floor** | **1.0071** | |
| `PairformerLayer#004` | matmul | `1x16x512x512 , 512x128` | 2321 | `fidelity=LoFi` | 1.039x |  |
|  | | |  | `fidelity=HiFi2` | 1.042x |  |
|  | | |  | `fidelity=HiFi3` | 1.052x | 7.81e-03 |
|  | | |  | `fp32acc=0` | 1.070x | 1.56e-02 |
|  | | |  | `packerl1=0` | 1.045x |  |
|  | | |  | `dstfull=1` | 0.994x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.088x | 3.52e-02 |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.080x | 1.56e-02 |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.066x | 1.56e-02 |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.373x | 3.12e-02 |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.399x | 1.56e-02 |
|  | | |  | `grid=8x8` | 2.852x | 3.91e-03 |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 3.083x | 1.56e-02 |
|  | | |  | `grid=8x7` | 2.786x | 3.91e-03 |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 2.898x | 1.56e-02 |
|  | | |  | `grid=7x7` | 2.654x | 3.91e-03 |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 2.864x | 1.56e-02 |
|  | | |  | `grid=8x4` | 2.323x | 3.91e-03 |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 2.463x | 1.56e-02 |
|  | | |  | `grid=4x8` | 2.535x | 3.91e-03 |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 2.977x | 1.56e-02 |
|  | | |  | `grid=8x2` | 1.516x | 3.91e-03 |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 1.700x | 1.56e-02 |
|  | | |  | `outbuf=L1` | 1.039x |  |
| | | | | **A/A floor** | **1.5842** | |
| `DiffusionStep#029` | matmul | `1x224x32x128 , 128x256` | 1988 | `fidelity=LoFi` | 1.013x |  |
|  | | |  | `fidelity=HiFi2` | 1.009x |  |
|  | | |  | `fidelity=HiFi3` | 1.004x |  |
|  | | |  | `fp32acc=0` | 1.008x |  |
|  | | |  | `packerl1=0` | 1.004x |  |
|  | | |  | `dstfull=1` | 0.999x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.027x |  |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.020x |  |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.014x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.052x | 1.76e-02 |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.016x |  |
|  | | |  | `grid=8x8` | 5.043x | 3.91e-03 |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 5.239x | 7.81e-03 |
|  | | |  | `grid=8x7` | 5.133x | 3.91e-03 |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 5.214x | 7.81e-03 |
|  | | |  | `grid=7x7` | 5.054x | 3.91e-03 |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 5.118x | 7.81e-03 |
|  | | |  | `grid=8x4` | 4.996x | 3.91e-03 |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 4.974x | 7.81e-03 |
|  | | |  | `grid=4x8` | 5.349x | 3.91e-03 |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 5.421x | 7.81e-03 |
|  | | |  | `grid=8x2` | 4.189x | 3.91e-03 |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 4.442x | 7.81e-03 |
|  | | |  | `outbuf=L1` | 3.340x | 7.81e-03 |
| | | | | **A/A floor** | **1.0098** | |
| `DiffusionStep#035` | matmul | `1x224x32x128 , 128x128 , 128` | 1332 | `fidelity=LoFi` | 1.011x |  |
|  | | |  | `fidelity=HiFi2` | 1.002x |  |
|  | | |  | `fidelity=HiFi3` | 1.009x |  |
|  | | |  | `fp32acc=0` | 1.006x |  |
|  | | |  | `packerl1=0` | 0.901x |  |
|  | | |  | `dstfull=1` | 0.999x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.038x |  |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.011x |  |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.014x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.007x |  |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.012x |  |
|  | | |  | `grid=8x8` | 7.804x | 9.77e-04 |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 8.023x | 5.86e-03 |
|  | | |  | `grid=8x7` | 7.928x | 9.77e-04 |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 8.108x | 5.86e-03 |
|  | | |  | `grid=7x7` | 7.504x | 9.77e-04 |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 7.733x | 5.86e-03 |
|  | | |  | `grid=8x4` | 7.325x | 9.77e-04 |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 7.649x | 5.86e-03 |
|  | | |  | `grid=4x8` | 7.626x | 9.77e-04 |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 8.008x | 5.86e-03 |
|  | | |  | `grid=8x2` | 6.514x | 9.77e-04 |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 6.775x | 5.86e-03 |
|  | | |  | `outbuf=L1` | 3.871x | 7.81e-03 |
| | | | | **A/A floor** | **1.041** | |
| `DiffusionStep#036` | matmul | `1x224x32x128 , 128x128` | 1317 | `fidelity=LoFi` | 1.061x | 1.76e-02 |
|  | | |  | `fidelity=HiFi2` | 1.046x |  |
|  | | |  | `fidelity=HiFi3` | 1.030x |  |
|  | | |  | `fp32acc=0` | 1.043x |  |
|  | | |  | `packerl1=0` | 1.030x |  |
|  | | |  | `dstfull=1` | 1.018x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.067x | 1.76e-02 |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.061x | 5.86e-03 |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.052x | 7.81e-03 |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.092x | 1.76e-02 |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.062x | 5.86e-03 |
|  | | |  | `grid=8x8` | 7.847x | 1.95e-03 |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 7.966x | 5.86e-03 |
|  | | |  | `grid=8x7` | 7.806x | 1.95e-03 |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 8.085x | 5.86e-03 |
|  | | |  | `grid=7x7` | 7.677x | 1.95e-03 |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 7.780x | 5.86e-03 |
|  | | |  | `grid=8x4` | 7.442x | 1.95e-03 |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 7.805x | 5.86e-03 |
|  | | |  | `grid=4x8` | 8.076x | 1.95e-03 |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 8.158x | 5.86e-03 |
|  | | |  | `grid=8x2` | 6.956x | 1.95e-03 |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 7.012x | 5.86e-03 |
|  | | |  | `outbuf=L1` | 4.651x | 7.81e-03 |
| | | | | **A/A floor** | **1.0383** | |
| `DiffusionStep#043` | matmul | `1x224x128x128 , 128x256` | 1258 | `fidelity=LoFi` | 1.022x |  |
|  | | |  | `fidelity=HiFi2` | 1.017x |  |
|  | | |  | `fidelity=HiFi3` | 1.008x |  |
|  | | |  | `fp32acc=0` | 1.016x |  |
|  | | |  | `packerl1=0` | 1.033x |  |
|  | | |  | `dstfull=1` | 0.999x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.043x |  |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.036x |  |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.024x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.060x | 1.76e-02 |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.035x |  |
|  | | |  | `grid=8x8` | 2.187x | 0.00e+00 |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 2.211x | 7.81e-03 |
|  | | |  | `grid=8x7` | 2.170x | 0.00e+00 |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 2.192x | 7.81e-03 |
|  | | |  | `grid=7x7` | 2.163x | 0.00e+00 |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 2.158x | 7.81e-03 |
|  | | |  | `grid=8x4` | 2.126x | 0.00e+00 |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 2.101x | 7.81e-03 |
|  | | |  | `grid=4x8` | 2.338x | 0.00e+00 |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 2.267x | 7.81e-03 |
|  | | |  | `grid=8x2` | 1.757x | 0.00e+00 |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 2.005x | 7.81e-03 |
|  | | |  | `outbuf=L1` | 1.369x | 7.81e-03 |
| | | | | **A/A floor** | **1.0054** | |
| `PairformerLayer#017` | matmul | `1x128x512x512 , 1x128x512x512` | 1233 | `fidelity=LoFi` | 1.075x | 9.99e-01 |
|  | | |  | `fidelity=HiFi2` | 1.059x | 1.00e+00 |
|  | | |  | `fidelity=HiFi3` | 1.034x |  |
|  | | |  | `fp32acc=0` | 1.062x | 1.00e+00 |
|  | | |  | `packerl1=0` | 0.941x |  |
|  | | |  | `dstfull=1` | 1.000x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.111x | 1.00e+00 |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.097x | 1.00e+00 |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.084x | 1.00e+00 |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.146x | 9.99e-01 |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.097x | 1.00e+00 |
|  | | |  | `grid=8x8` | 1.001x |  |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 1.740x | 1.00e+00 |
|  | | |  | `grid=8x7` | 0.903x |  |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 1.635x | 1.00e+00 |
|  | | |  | `grid=7x7` | 0.553x |  |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 1.636x | 1.00e+00 |
|  | | |  | `grid=8x4` | 0.787x |  |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 1.473x | 1.00e+00 |
|  | | |  | `grid=4x8` | 0.860x |  |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 1.491x | 1.00e+00 |
|  | | |  | `grid=8x2` | 0.510x |  |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 1.033x |  |
|  | | |  | `outbuf=L1` | RuntimeError: TT_THROW @ /project/tt_metal/impl/program/prog |  |
| | | | | **A/A floor** | **1.0023** | |
| `DiffusionStep#016` | softmax | `1x16x512x512` | 1089 | `fidelity=LoFi` | 1.005x |  |
|  | | |  | `fidelity=HiFi2` | 1.008x |  |
|  | | |  | `fidelity=HiFi3` | 1.004x |  |
|  | | |  | `fp32acc=0` | 1.702x | 9.20e-05 |
|  | | |  | `packerl1=0` | 0.998x |  |
|  | | |  | `dstfull=1` | 1.000x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.798x | 1.22e-04 |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.833x | 9.20e-05 |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.793x | 9.20e-05 |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.851x | 1.22e-04 |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.836x | 9.20e-05 |
|  | | |  | `outbuf=L1` | 1.023x |  |
| | | | | **A/A floor** | **1.0071** | |
| `MSALayer#015` | matmul | `1024x32x512 , 1x512x512` | 1053 | `fidelity=LoFi` | 1.007x |  |
|  | | |  | `fidelity=HiFi2` | 1.003x |  |
|  | | |  | `fidelity=HiFi3` | 1.001x |  |
|  | | |  | `fp32acc=0` | 1.010x |  |
|  | | |  | `packerl1=0` | 1.038x |  |
|  | | |  | `dstfull=1` | 1.000x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.020x |  |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.018x |  |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.015x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.051x | 1.00e+00 |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.018x |  |
|  | | |  | `grid=8x8` | 7.140x | 9.96e-01 |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 7.294x | 9.99e-01 |
|  | | |  | `grid=8x7` | 5.040x | 9.96e-01 |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 5.131x | 9.99e-01 |
|  | | |  | `grid=7x7` | 5.047x | 9.96e-01 |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 5.146x | 9.99e-01 |
|  | | |  | `grid=8x4` | 5.108x | 9.96e-01 |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 6.438x | 9.99e-01 |
|  | | |  | `grid=4x8` | 5.366x | 9.96e-01 |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 7.368x | 9.99e-01 |
|  | | |  | `grid=8x2` | 3.247x | 9.96e-01 |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 5.143x | 9.99e-01 |
|  | | |  | `outbuf=L1` | 1.043x |  |
| | | | | **A/A floor** | **1.0006** | |
| `PairformerLayer#005` | layernorm | `1x512x512x128 , 128 , 128` | 988 | `fidelity=LoFi` | 1.069x | 7.03e-02 |
|  | | |  | `fidelity=HiFi2` | 1.049x |  |
|  | | |  | `fidelity=HiFi3` | 1.024x |  |
|  | | |  | `fp32acc=0` | 1.269x | 1.56e-02 |
|  | | |  | `packerl1=0` | 1.001x |  |
|  | | |  | `dstfull=1` | 1.001x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.346x | 5.86e-02 |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.292x | 1.56e-02 |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.311x | 1.56e-02 |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.333x | 5.86e-02 |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.319x | 1.56e-02 |
|  | | |  | `outbuf=L1` | 1.007x |  |
| | | | | **A/A floor** | **1.0033** | |
| `DiffusionStep#053` | matmul | `1x224x32x256 , 256x128` | 985 | `fidelity=LoFi` | 1.040x |  |
|  | | |  | `fidelity=HiFi2` | 1.028x |  |
|  | | |  | `fidelity=HiFi3` | 1.014x |  |
|  | | |  | `fp32acc=0` | 1.032x |  |
|  | | |  | `packerl1=0` | 1.002x |  |
|  | | |  | `dstfull=1` | 0.997x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.059x | 2.34e-02 |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.050x |  |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.047x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.079x | 2.34e-02 |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.052x | 1.17e-02 |
|  | | |  | `grid=8x8` | 8.707x | 3.91e-03 |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 8.559x | 1.17e-02 |
|  | | |  | `grid=8x7` | 8.553x | 3.91e-03 |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 8.630x | 1.17e-02 |
|  | | |  | `grid=7x7` | 8.499x | 3.91e-03 |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 8.479x | 1.17e-02 |
|  | | |  | `grid=8x4` | 7.821x | 3.91e-03 |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 8.107x | 1.17e-02 |
|  | | |  | `grid=4x8` | 8.952x | 3.91e-03 |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 9.138x | 1.17e-02 |
|  | | |  | `grid=8x2` | 7.108x | 3.91e-03 |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 7.039x | 1.17e-02 |
|  | | |  | `outbuf=L1` | 3.958x | 1.37e-02 |
| | | | | **A/A floor** | **1.0067** | |
| `PairformerLayer#006` | binary | `1x512x512x128 , 1x512x512x128 , 1x512x512x128` | 895 | `outbuf=L1` | 1.001x |  |
| | | | | **A/A floor** | **1.0024** | |
| `PairformerLayer#019` | matmul | `512x512x128 , 128x4` | 855 | `fidelity=LoFi` | 1.017x |  |
|  | | |  | `fidelity=HiFi2` | 1.012x |  |
|  | | |  | `fidelity=HiFi3` | 1.007x |  |
|  | | |  | `fp32acc=0` | 1.028x |  |
|  | | |  | `packerl1=0` | 0.999x |  |
|  | | |  | `dstfull=1` | 1.001x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.034x |  |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.034x |  |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.033x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.034x |  |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.031x |  |
|  | | |  | `grid=8x8` | 1.334x | 1.95e-03 |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 1.346x | 5.86e-03 |
|  | | |  | `grid=8x7` | 1.512x | 1.95e-03 |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 1.287x | 5.86e-03 |
|  | | |  | `grid=7x7` | 1.445x | 1.95e-03 |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 1.293x | 5.86e-03 |
|  | | |  | `grid=8x4` | 1.281x | 1.95e-03 |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 1.289x | 5.86e-03 |
|  | | |  | `grid=4x8` | 1.501x | 1.95e-03 |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 1.494x | 5.86e-03 |
|  | | |  | `grid=8x2` | 1.110x | 1.95e-03 |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 1.119x | 5.86e-03 |
|  | | |  | `outbuf=L1` | 1.123x | 0.00e+00 |
| | | | | **A/A floor** | **1.0023** | |
| `PairformerLayer#014` | binary | `1x512x512x128 , 1x512x512x1 , 1x512x512x128` | 835 | `outbuf=L1` | 1.001x |  |
| | | | | **A/A floor** | **1.0001** | |
| `PairformerLayer#025` | transpose | `1x512x512x128` | 800 | `outbuf=L1` | 1.001x |  |
| | | | | **A/A floor** | **1.0004** | |
| `DiffusionStep#015` | binary | `1x16x512x512 , 1x16x512x512` | 745 | `outbuf=L1` | 1.007x |  |
| | | | | **A/A floor** | **1.0011** | |
| `DiffusionStep#013` | matmul | `1x16x512x64 , 1x16x64x512` | 719 | `fidelity=LoFi` | 1.021x |  |
|  | | |  | `fidelity=HiFi2` | 1.013x |  |
|  | | |  | `fidelity=HiFi3` | 1.010x |  |
|  | | |  | `fp32acc=0` | 1.024x |  |
|  | | |  | `packerl1=0` | 1.002x |  |
|  | | |  | `dstfull=1` | 0.998x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.026x |  |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.028x |  |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.019x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.019x |  |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.022x |  |
|  | | |  | `grid=8x8` | 0.991x |  |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 0.964x |  |
|  | | |  | `grid=8x7` | 0.985x |  |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 1.060x | 3.91e-03 |
|  | | |  | `grid=7x7` | 0.900x |  |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 1.055x | 3.91e-03 |
|  | | |  | `grid=8x4` | 0.939x |  |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 1.375x | 3.91e-03 |
|  | | |  | `grid=4x8` | 0.912x |  |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 0.968x |  |
|  | | |  | `grid=8x2` | 0.799x |  |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 1.724x | 3.91e-03 |
|  | | |  | `outbuf=L1` | 0.476x |  |
| | | | | **A/A floor** | **1.0224** | |
| `DiffusionStep#040` | matmul | `1x16x128x448 , 448x1792` | 716 | `fidelity=LoFi` | 1.013x |  |
|  | | |  | `fidelity=HiFi2` | 1.021x |  |
|  | | |  | `fidelity=HiFi3` | 1.016x |  |
|  | | |  | `fp32acc=0` | 1.022x |  |
|  | | |  | `packerl1=0` | 1.015x |  |
|  | | |  | `dstfull=1` | 1.001x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.033x |  |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.044x |  |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.026x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.062x | 2.34e-02 |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.045x |  |
|  | | |  | `grid=8x8` | 2.157x | 0.00e+00 |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 2.364x | 1.56e-02 |
|  | | |  | `grid=8x7` | 1.963x | 0.00e+00 |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 2.525x | 1.56e-02 |
|  | | |  | `grid=7x7` | 2.088x | 0.00e+00 |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 2.602x | 1.56e-02 |
|  | | |  | `grid=8x4` | 1.497x | 0.00e+00 |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 1.652x | 1.56e-02 |
|  | | |  | `grid=4x8` | 1.699x | 0.00e+00 |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 2.057x | 1.56e-02 |
|  | | |  | `grid=8x2` | 0.979x |  |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 1.125x | 1.56e-02 |
|  | | |  | `outbuf=L1` | 1.399x | 0.00e+00 |
| | | | | **A/A floor** | **1.0113** | |
| `PairformerLayer#008` | binary | `1x512x512x128 , 1x512x512x128 , 1x512x512x128` | 674 | `outbuf=L1` | 1.001x |  |
| | | | | **A/A floor** | **1.006** | |
| `MSALayer#010` | matmul | `1024x512x64 , 64x32` | 654 | `fidelity=LoFi` | 1.024x |  |
|  | | |  | `fidelity=HiFi2` | 1.015x |  |
|  | | |  | `fidelity=HiFi3` | 1.010x |  |
|  | | |  | `fp32acc=0` | 1.020x |  |
|  | | |  | `packerl1=0` | 1.002x |  |
|  | | |  | `dstfull=1` | 1.000x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0` | 1.045x |  |
|  | | |  | `fidelity=HiFi2,fp32acc=0` | 1.038x |  |
|  | | |  | `fidelity=HiFi3,fp32acc=0` | 1.028x |  |
|  | | |  | `fidelity=LoFi,fp32acc=0,packerl1=0` | 1.047x |  |
|  | | |  | `fidelity=HiFi2,fp32acc=0,dstfull=1` | 1.037x |  |
|  | | |  | `grid=8x8` | 1.673x | 1.95e-03 |
|  | | |  | `grid=8x8,fidelity=HiFi2,fp32acc=0` | 1.703x | 4.88e-03 |
|  | | |  | `grid=8x7` | 1.516x | 1.95e-03 |
|  | | |  | `grid=8x7,fidelity=HiFi2,fp32acc=0` | 1.549x | 4.88e-03 |
|  | | |  | `grid=7x7` | 1.812x | 1.95e-03 |
|  | | |  | `grid=7x7,fidelity=HiFi2,fp32acc=0` | 1.848x | 4.88e-03 |
|  | | |  | `grid=8x4` | 1.575x | 1.95e-03 |
|  | | |  | `grid=8x4,fidelity=HiFi2,fp32acc=0` | 1.578x | 4.88e-03 |
|  | | |  | `grid=4x8` | 1.951x | 1.95e-03 |
|  | | |  | `grid=4x8,fidelity=HiFi2,fp32acc=0` | 1.979x | 4.88e-03 |
|  | | |  | `grid=8x2` | 1.364x | 1.95e-03 |
|  | | |  | `grid=8x2,fidelity=HiFi2,fp32acc=0` | 1.419x | 4.88e-03 |
|  | | |  | `outbuf=L1` | 1.185x | 0.00e+00 |
| | | | | **A/A floor** | **1.0024** | |
