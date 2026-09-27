| op (site) | dtype | grad | rel L2 vs float64 | grade | backward alone | systematic mean | sigma | bias? | repeat bit-exact |
|---|---|---|---|---|---|---|---|---|---|
| `layer_norm/pair_K128` | bfloat16 | `dx` | 3.902e-03 | **PASS** | - | +6.58e-06 | 1 | **NOISE** | yes |
| `layer_norm/pair_K128` | bfloat16 | `dgamma` | 3.826e-03 | **PASS** | - | -3.47e-04 | 5 | **NOISE** | yes |
| `layer_norm/pair_K128` | bfloat16 | `dbeta` | 1.701e-03 | **PASS** | - | -2.27e-05 | 1 | **NOISE** | yes |
| `layer_norm/pair_K128` | float32 | `dx` | 1.481e-04 | **PASS** | - | +1.36e-04 | 1654 | **NOISE** | yes |
| `layer_norm/pair_K128` | float32 | `dgamma` | 1.447e-04 | **PASS** | - | +1.36e-04 | 209 | **NOISE** | yes |
| `layer_norm/pair_K128` | float32 | `dbeta` | 9.300e-08 | **PASS** | - | -5.76e-10 | 0 | **NOISE** | yes |
| `layer_norm/single_K384` | bfloat16 | `dx` | 4.317e-03 | **PASS** | - | +1.91e-03 | 91 | **BIAS** | yes |
| `layer_norm/single_K384` | bfloat16 | `dgamma` | 4.048e-03 | **PASS** | - | +1.58e-03 | 38 | **BIAS** | yes |
| `layer_norm/single_K384` | bfloat16 | `dbeta` | 1.656e-03 | **PASS** | - | +4.58e-06 | 0 | **NOISE** | yes |
| `layer_norm/single_K384` | float32 | `dx` | 2.089e-03 | **MARGINAL** | - | +2.08e-03 | 11327 | **BIAS** | yes |
| `layer_norm/single_K384` | float32 | `dgamma` | 2.090e-03 | **MARGINAL** | - | +2.08e-03 | 1106 | **BIAS** | yes |
| `layer_norm/single_K384` | float32 | `dbeta` | 8.023e-08 | **PASS** | - | -1.58e-10 | 0 | **NOISE** | yes |
| `layer_norm/single_K384_CONTROL` | bfloat16 | `dx` | 4.317e-03 | **PASS** | - | +1.91e-03 | 91 | **BIAS** | yes |
| `layer_norm/single_K384_CONTROL` | bfloat16 | `dgamma` | 4.048e-03 | **PASS** | - | +1.58e-03 | 38 | **BIAS** | yes |
| `layer_norm/single_K384_CONTROL` | bfloat16 | `dbeta` | 1.656e-03 | **PASS** | - | +4.58e-06 | 0 | **NOISE** | yes |
| `layer_norm/single_K384_CONTROL` | float32 | `dx` | 2.089e-03 | **MARGINAL** | - | +2.08e-03 | 11327 | **BIAS** | yes |
| `layer_norm/single_K384_CONTROL` | float32 | `dgamma` | 9.484e-04 | **PASS** | - | +6.73e-04 | 83 | **NOISE** | yes |
| `layer_norm/single_K384_CONTROL` | float32 | `dbeta` | 1.534e-03 | **MARGINAL** | - | -1.41e-03 | 152 | **BIAS** | yes |
| `layer_norm/single_K384_MEANCFG` | bfloat16 | `dx` | 4.317e-03 | **PASS** | - | +1.91e-03 | 91 | **BIAS** | yes |
| `layer_norm/single_K384_MEANCFG` | bfloat16 | `dgamma` | 4.048e-03 | **PASS** | - | +1.58e-03 | 38 | **BIAS** | yes |
| `layer_norm/single_K384_MEANCFG` | bfloat16 | `dbeta` | 1.656e-03 | **PASS** | - | +4.58e-06 | 0 | **NOISE** | yes |
| `layer_norm/single_K384_MEANCFG` | float32 | `dx` | 2.089e-03 | **MARGINAL** | - | +2.08e-03 | 11327 | **BIAS** | yes |
| `layer_norm/single_K384_MEANCFG` | float32 | `dgamma` | 9.484e-04 | **PASS** | - | +6.73e-04 | 83 | **NOISE** | yes |
| `layer_norm/single_K384_MEANCFG` | float32 | `dbeta` | 1.534e-03 | **MARGINAL** | - | -1.41e-03 | 152 | **BIAS** | yes |
| `layer_norm/single_K384_RSQRTSPLIT` | bfloat16 | `dx` | 5.392e-03 | **PASS** | - | +3.41e-03 | 159 | **BIAS** | yes |
| `layer_norm/single_K384_RSQRTSPLIT` | bfloat16 | `dgamma` | 5.153e-03 | **PASS** | - | +3.12e-03 | 66 | **BIAS** | yes |
| `layer_norm/single_K384_RSQRTSPLIT` | bfloat16 | `dbeta` | 1.656e-03 | **PASS** | - | +4.58e-06 | 0 | **NOISE** | yes |
| `layer_norm/single_K384_RSQRTSPLIT` | float32 | `dx` | 2.089e-03 | **MARGINAL** | - | +2.08e-03 | 11336 | **BIAS** | yes |
| `layer_norm/single_K384_RSQRTSPLIT` | float32 | `dgamma` | 9.484e-04 | **PASS** | - | +6.73e-04 | 83 | **NOISE** | yes |
| `layer_norm/single_K384_RSQRTSPLIT` | float32 | `dbeta` | 1.534e-03 | **MARGINAL** | - | -1.41e-03 | 152 | **BIAS** | yes |
| `layer_norm/single_K384_SUMSCALE` | bfloat16 | `dx` | 4.410e-03 | **PASS** | - | +2.06e-03 | 91 | **BIAS** | yes |
| `layer_norm/single_K384_SUMSCALE` | bfloat16 | `dgamma` | 4.169e-03 | **PASS** | - | +1.71e-03 | 41 | **BIAS** | yes |
| `layer_norm/single_K384_SUMSCALE` | bfloat16 | `dbeta` | 1.656e-03 | **PASS** | - | +4.58e-06 | 0 | **NOISE** | yes |
| `layer_norm/single_K384_SUMSCALE` | float32 | `dx` | 1.399e-04 | **PASS** | - | +1.36e-04 | 1000 | **NOISE** | yes |
| `layer_norm/single_K384_SUMSCALE` | float32 | `dgamma` | 1.414e-03 | **MARGINAL** | - | -1.26e-03 | 155 | **BIAS** | yes |
| `layer_norm/single_K384_SUMSCALE` | float32 | `dbeta` | 1.534e-03 | **MARGINAL** | - | -1.41e-03 | 152 | **BIAS** | yes |
| `layer_norm/pair_K128_SUMSCALE` | bfloat16 | `dx` | 3.902e-03 | **PASS** | - | +6.58e-06 | 1 | **NOISE** | yes |
| `layer_norm/pair_K128_SUMSCALE` | bfloat16 | `dgamma` | 4.151e-03 | **PASS** | - | -3.51e-04 | 6 | **NOISE** | yes |
| `layer_norm/pair_K128_SUMSCALE` | bfloat16 | `dbeta` | 2.336e-03 | **PASS** | - | +2.15e-05 | 1 | **NOISE** | yes |
| `layer_norm/pair_K128_SUMSCALE` | float32 | `dx` | 1.481e-04 | **PASS** | - | +1.36e-04 | 1654 | **NOISE** | yes |
| `layer_norm/pair_K128_SUMSCALE` | float32 | `dgamma` | 2.121e-03 | **MARGINAL** | - | -1.98e-03 | 141 | **BIAS** | yes |
| `layer_norm/pair_K128_SUMSCALE` | float32 | `dbeta` | 2.241e-03 | **MARGINAL** | - | -2.13e-03 | 171 | **BIAS** | yes |
| `softmax/triatt_K64` | bfloat16 | `dx` | 2.391e-02 | **PASS** | 2.49e-03 | -1.19e-02 | 77 | **BIAS** | yes |
| `softmax/triatt_K64` | float32 | `dx` | 2.201e-02 | **MARGINAL** | 9.75e-05 | -9.29e-03 | 64 | **BIAS** | yes |
| `softmax/apb_K64` | bfloat16 | `dx` | 2.392e-02 | **PASS** | 2.47e-03 | -1.19e-02 | 177 | **BIAS** | yes |
| `softmax/apb_K64` | float32 | `dx` | 2.206e-02 | **MARGINAL** | 1.01e-04 | -9.27e-03 | 131 | **BIAS** | yes |
| `softmax/triatt_K384` | bfloat16 | `dx` | 2.250e-02 | **PASS** | 2.38e-03 | -3.57e-03 | 48 | **BIAS** | yes |
| `softmax/triatt_K384` | float32 | `dx` | 2.230e-02 | **MARGINAL** | 3.98e-05 | -8.41e-03 | 133 | **BIAS** | yes |
| `softmax/triatt_K64_CONTROL` | bfloat16 | `dx` | 2.391e-02 | **PASS** | 2.49e-03 | -1.19e-02 | 77 | **BIAS** | yes |
| `softmax/triatt_K64_CONTROL` | float32 | `dx` | 2.201e-02 | **MARGINAL** | 9.75e-05 | -9.29e-03 | 64 | **BIAS** | yes |
| `softmax/triatt_K64_SUMCFG` | bfloat16 | `dx` | 2.391e-02 | **PASS** | 2.49e-03 | -1.19e-02 | 77 | **BIAS** | yes |
| `softmax/triatt_K64_SUMCFG` | float32 | `dx` | 2.201e-02 | **MARGINAL** | 9.75e-05 | -9.29e-03 | 64 | **BIAS** | yes |
| `softmax/triatt_K64_FWDCFG` | bfloat16 | `dx` | 2.948e-03 | **PASS** | 2.44e-03 | +2.11e-04 | 10 | **NOISE** | yes |
| `softmax/triatt_K64_FWDCFG` | float32 | `dx` | 9.086e-04 | **PASS** | 1.02e-04 | -5.28e-04 | 70 | **NOISE** | yes |
| `linear/pair_128_128` | bfloat16 | `dx` | 1.706e-03 | **PASS** | - | -1.35e-04 | 153 | **NOISE** | yes |
| `linear/pair_128_128` | bfloat16 | `dw` | 4.889e-04 | **PASS** | - | -3.06e-04 | 391 | **NOISE** | yes |
| `linear/pair_128_128` | bfloat16 | `dbias` | 1.701e-03 | **PASS** | - | -2.27e-05 | 1 | **NOISE** | yes |
| `linear/pair_128_128` | float32 | `dx` | 1.345e-03 | **MARGINAL** | - | -1.19e-03 | 1697 | **BIAS** | yes |
| `linear/pair_128_128` | float32 | `dw` | 1.500e-03 | **MARGINAL** | - | -1.36e-03 | 1532 | **BIAS** | yes |
| `linear/pair_128_128` | float32 | `dbias` | 9.300e-08 | **PASS** | - | -5.76e-10 | 0 | **NOISE** | yes |
| `matmul/pair_4096x128x128` | bfloat16 | `da` | 1.706e-03 | **PASS** | - | -1.35e-04 | 153 | **NOISE** | yes |
| `matmul/pair_4096x128x128` | bfloat16 | `db` | 4.889e-04 | **PASS** | - | -3.06e-04 | 391 | **NOISE** | yes |
| `matmul/pair_4096x128x128` | float32 | `da` | 1.345e-03 | **MARGINAL** | - | -1.19e-03 | 1697 | **BIAS** | yes |
| `matmul/pair_4096x128x128` | float32 | `db` | 1.500e-03 | **MARGINAL** | - | -1.36e-03 | 1532 | **BIAS** | yes |
| `multiply/pair` | bfloat16 | `da` | 1.653e-03 | **PASS** | - | -4.39e-05 | 37 | **NOISE** | yes |
| `multiply/pair` | bfloat16 | `db` | 1.653e-03 | **PASS** | - | -3.95e-05 | 31 | **NOISE** | yes |
| `multiply/pair` | float32 | `da` | 2.530e-08 | **PASS** | - | +2.15e-11 | 1 | **NOISE** | yes |
| `multiply/pair` | float32 | `db` | 2.532e-08 | **PASS** | - | -1.30e-11 | 1 | **NOISE** | yes |
| `sigmoid/pair` | bfloat16 | `dx` | 2.959e-03 | **PASS** | - | +4.38e-04 | 364 | **NOISE** | yes |
| `sigmoid/pair` | float32 | `dx` | 7.041e-08 | **PASS** | - | +4.09e-10 | 15 | **NOISE** | yes |

| case | fd rel err (bar 1e-6) | forward rel L2 | device row sum - 1 |
|---|---|---|---|
| `layer_norm/pair_K128` bfloat16 | 6.2e-10 OK | 1.931e-03 | - |
| `layer_norm/pair_K128` float32 | 5.0e-10 OK | 1.666e-03 | - |
| `layer_norm/single_K384` bfloat16 | 1.6e-11 OK | 1.927e-03 | - |
| `layer_norm/single_K384` float32 | 2.6e-11 OK | 1.649e-03 | - |
| `layer_norm/single_K384_CONTROL` bfloat16 | 1.6e-11 OK | 1.927e-03 | - |
| `layer_norm/single_K384_CONTROL` float32 | 2.6e-11 OK | 1.649e-03 | - |
| `layer_norm/single_K384_MEANCFG` bfloat16 | 1.6e-11 OK | 1.927e-03 | - |
| `layer_norm/single_K384_MEANCFG` float32 | 2.6e-11 OK | 1.649e-03 | - |
| `layer_norm/single_K384_RSQRTSPLIT` bfloat16 | 1.6e-11 OK | 1.927e-03 | - |
| `layer_norm/single_K384_RSQRTSPLIT` float32 | 2.6e-11 OK | 1.649e-03 | - |
| `layer_norm/single_K384_SUMSCALE` bfloat16 | 1.6e-11 OK | 1.927e-03 | - |
| `layer_norm/single_K384_SUMSCALE` float32 | 2.6e-11 OK | 1.649e-03 | - |
| `layer_norm/pair_K128_SUMSCALE` bfloat16 | 6.2e-10 OK | 1.931e-03 | - |
| `layer_norm/pair_K128_SUMSCALE` float32 | 5.0e-10 OK | 1.666e-03 | - |
| `softmax/triatt_K64` bfloat16 | 2.6e-11 OK | 2.624e-02 | -8.07e-03 |
| `softmax/triatt_K64` float32 | 1.5e-11 OK | 2.422e-02 | -4.57e-03 |
| `softmax/apb_K64` bfloat16 | 4.7e-11 OK | 2.631e-02 | -8.12e-03 |
| `softmax/apb_K64` float32 | 1.6e-10 OK | 2.435e-02 | -4.62e-03 |
| `softmax/triatt_K384` bfloat16 | 3.8e-10 OK | 2.286e-02 | -4.18e-04 |
| `softmax/triatt_K384` float32 | 2.4e-10 OK | 2.293e-02 | -4.26e-03 |
| `softmax/triatt_K64_CONTROL` bfloat16 | 2.6e-11 OK | 2.624e-02 | -8.07e-03 |
| `softmax/triatt_K64_CONTROL` float32 | 1.5e-11 OK | 2.422e-02 | -4.57e-03 |
| `softmax/triatt_K64_SUMCFG` bfloat16 | 2.6e-11 OK | 2.624e-02 | -8.07e-03 |
| `softmax/triatt_K64_SUMCFG` float32 | 1.5e-11 OK | 2.422e-02 | -4.57e-03 |
| `softmax/triatt_K64_FWDCFG` bfloat16 | 2.6e-11 OK | 1.803e-03 | -4.06e-04 |
| `softmax/triatt_K64_FWDCFG` float32 | 1.5e-11 OK | 9.323e-04 | -4.44e-04 |
| `linear/pair_128_128` bfloat16 | 4.5e-10 OK | 1.699e-03 | - |
| `linear/pair_128_128` float32 | 4.1e-09 OK | 1.514e-03 | - |
| `matmul/pair_4096x128x128` bfloat16 | 4.5e-10 OK | 1.706e-03 | - |
| `matmul/pair_4096x128x128` float32 | 2.6e-09 OK | 1.346e-03 | - |
| `multiply/pair` bfloat16 | 1.4e-10 OK | 1.653e-03 | - |
| `multiply/pair` float32 | 2.3e-11 OK | 2.531e-08 | - |
| `sigmoid/pair` bfloat16 | 6.7e-10 OK | 1.679e-03 | - |
| `sigmoid/pair` float32 | 3.3e-10 OK | 5.151e-08 | - |
