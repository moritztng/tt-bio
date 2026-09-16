# Bounded default-fold census

Measured program-marker spans at during-window sampled 1350 MHz, with graph/observer/sum profiling and node-3 co-tenancy. Counts cover only the named windows; no whole-fold extrapolation. Matrix counts cover only the stated calls. Other arithmetic remains uncounted or symbolic.

| Class | Calls | Program spans (cycles) | Counted matrix calls | Matrix shape FLOPs | Conditional tile32 FLOPs | Modeled DRAM bytes (known calls) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MatmulDeviceOperation | 512 | 53004546 | 512 | 843154653696 | 845918240768 | 4656789250 (512) |
| BinaryNgDeviceOperation | 414 | 25851252 | 0 | unknown/symbolic | unknown | 5439624832 (414) |
| TransposeDeviceOperation | 76 | 11150150 | 0 | unknown/symbolic | unknown | 1625751552 (76) |
| LayerNormDeviceOperation | 155 | 10806599 | 0 | unknown/symbolic | unknown | 1925099776 (155) |
| generic:generic_minimal_matmul:kernels,triatt | 8 | 8027258 | 0 | unknown/symbolic | unknown | 1753907200 (8) |
| generic:generic_minimal_matmul:kernels,mm_split | 4 | 7908436 | 4 | 171798691840 | 171798691840 | 1611268096 (4) |
| generic:reblock_permute_gated:reblock_permute_gated | 8 | 6971571 | 0 | unknown/symbolic | unknown | 2684354560 (8) |
| generic:sdpa:compute,dataflow | 4 | 6955273 | 0 | unknown/symbolic | unknown | 1082130432 (4) |
| ReshapeViewDeviceOperation | 26 | 4887559 | 0 | unknown/symbolic | unknown | 1112276992 (26) |
| EmbeddingsDeviceOperation | 1 | 4074502 | 0 | unknown/symbolic | unknown | 17825920 (1) |
| UntilizeDeviceOperation | 8 | 4040351 | 0 | unknown/symbolic | unknown | 1090617344 (8) |
| TilizeDeviceOperation | 8 | 3456374 | 0 | unknown/symbolic | unknown | 1092091904 (8) |
| SDPAOperation | 30 | 3061301 | 0 | unknown/symbolic | unknown | 398327808 (30) |
| generic:reblock_permute_back:reblock_permute,reblock_permute_back | 4 | 2785100 | 0 | unknown/symbolic | unknown | 536870912 (4) |
| NlpCreateHeadsDeviceOperation | 31 | 2125400 | 0 | unknown/symbolic | unknown | None (0) |
| SliceDeviceOperation | 115 | 1806457 | 0 | unknown/symbolic | unknown | 1764179968 (115) |
| SoftmaxDeviceOperation | 6 | 1591751 | 0 | unknown/symbolic | unknown | 285212672 (6) |
| PermuteDeviceOperation | 8 | 1200883 | 0 | unknown/symbolic | unknown | 25165824 (8) |
| ConcatDeviceOperation | 8 | 645941 | 0 | unknown/symbolic | unknown | 184942592 (8) |
| PadDeviceOperation | 18 | 616471 | 0 | unknown/symbolic | unknown | 100712448 (18) |
| TernaryDeviceOperation | 1 | 209401 | 0 | unknown/symbolic | unknown | 67108864 (1) |
| CopyDeviceOperation | 26 | 164516 | 0 | unknown/symbolic | unknown | 28573696 (26) |
| NLPConcatHeadsDeviceOperation | 6 | 106540 | 0 | unknown/symbolic | unknown | 13762560 (6) |
| UnaryNgDeviceOperation | 1 | 3054 | 0 | unknown/symbolic | unknown | 1024 (1) |

Physical reads and issued instructions are unknown for every class. Bytes model logical payload crossing DRAM boundaries once; they exclude physical padding, rereads and spills. Utilization and cycles above a binding roof remain unmeasured for every model class.

| Window | Calls | Span sum | Span union | Overlap | Fenced envelope | Unclassified gap | Clock valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| MSALayer_0 | 227 | 79518145 | 79518145 | 0 | 6267323118 | 6187804973 | True |
| PairformerLayer_0 | 137 | 40820155 | 40820155 | 0 | 6069565550 | 6028745395 | True |
| DiffusionModule_0 | 1096 | 28604465 | 28604465 | 0 | 4985670830 | 4957066365 | True |
| ConfidenceHeadsDevice_0 | 18 | 12507921 | 12507921 | 0 | 3525703018 | 3513195097 | True |
| stream_copy_0 | 4 | 3643374 | 3643374 | 0 | 43042038 | 39398664 | True |
| stream_add_0 | 4 | 4843153 | 4843153 | 0 | 44068031 | 39224878 | True |
| dense_hifi4_0 | 4 | 48689330 | 48689330 | 0 | 64998937 | 16309607 | True |
| stream_copy_1 | 4 | 3637803 | 3637803 | 0 | 42691209 | 39053406 | True |
| stream_add_1 | 4 | 4859147 | 4859147 | 0 | 43217569 | 38358422 | True |
| dense_hifi4_1 | 4 | 48694899 | 48694899 | 0 | 65232572 | 16537673 | True |
| stream_copy_2 | 4 | 3633530 | 3633530 | 0 | 43421037 | 39787507 | True |
| stream_add_2 | 4 | 4853811 | 4853811 | 0 | 42974877 | 38121066 | True |
| dense_hifi4_2 | 4 | 48684670 | 48684670 | 0 | 65214345 | 16529675 | True |

Gaps are not CPU work. Outer graph spans are not used; all retained native operations have unique matching end nodes. Program IDs are joined through the recorded 10-bit device-ID encoding. The model completed its full default protocol; raw data outside named windows is excluded with a hash/byte drain ledger.

Verdict: STOP. Automatic host report exceeded resource budget (61.8 GB CSV, at least 182 GB RSS); owned device-closed postprocessor stopped
