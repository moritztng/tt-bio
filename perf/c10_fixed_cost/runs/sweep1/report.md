Reduced verdict: **GO**

### 512 aa

| clock | folds | median | min | max | stdev | adjacent \|delta\| | mean board W | host CPU s | elapsed Mcycles |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1350 MHz | 7 | 14.9306 s | 14.8828 s | 15.0958 s | 0.0781 s | 0.0781 s | 77.4 | 16.816 | 20156.3 |
| 1200 MHz | 7 | 16.1081 s | 16.0364 s | 16.1636 s | 0.0429 s | 0.0415 s | 65.7 | 17.981 | 19329.7 |
| 1000 MHz | 7 | 18.5518 s | 18.4907 s | 18.7602 s | 0.0952 s | 0.0843 s | 52.1 | 20.447 | 18551.8 |
| 800 MHz | 7 | 22.3574 s | 22.1914 s | 22.5835 s | 0.1230 s | 0.1257 s | 43.3 | 24.231 | 17885.9 |

- fit on all accepted folds (n=28, clocks [800, 1000, 1200, 1350]): **F = 3.9830 s +-0.1181**, C = 14665.0 Mcycles +-121.1, max |residual| 0.2692 s, rms 0.1195 s
- fit on cell medians (n=4, clocks [800, 1000, 1200, 1350]): **F = 3.9543 s +-0.2895**, C = 14678.2 Mcycles +-296.8, max |residual| 0.1035 s, rms 0.0812 s
- closed form on the 1350/800 MHz endpoints alone: **F = 4.1280 s +-0.0991**, C = 14583.5 Mcycles +-108.1; the endpoint arms enter F with gains 2.455 and -1.455
- inverse-clock model: A/A timing floor 0.1257 s, cell-median max |residual| 0.1035 s, per-fold rms 0.1195 s -> **the exact two-parameter form holds, F is point-identified**
- three estimators of F: all_folds 3.9830 s +-0.1181, cell_medians 3.9543 s +-0.2895, two_clock_endpoints 4.1280 s +-0.0991 -> F bounded to 3.6648 - 4.2439 s
- structure: 1 distinct CIF over 28 accepted folds, max pairwise domain RMSD 0.00e+00 A against the 0.6 A bar (byte-identical in every arm)
- demand for 10.0 s at 1350 MHz (now 14.8460 s):
  - F untouched: 8122.9 Mcycles allowed, a cut of 6542.1 Mcycles, **44.6 %** of the work term
  - F cut to 1.0 s: 12150.0 Mcycles allowed, a cut of 2515.0 Mcycles, **17.1 %**
  - cutting F to 1.0 s and touching no cycle at all: 11.8630 s

### 298 aa

| clock | folds | median | min | max | stdev | adjacent \|delta\| | mean board W | host CPU s | elapsed Mcycles |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1350 MHz | 7 | 9.6687 s | 9.6324 s | 9.8305 s | 0.0662 s | 0.0478 s | 70.2 | 11.018 | 13052.8 |
| 1200 MHz | 7 | 10.5973 s | 10.5833 s | 10.6229 s | 0.0145 s | 0.0198 s | 58.6 | 11.944 | 12716.7 |
| 1000 MHz | 7 | 12.3147 s | 12.2884 s | 12.3452 s | 0.0197 s | 0.0171 s | 47.4 | 13.658 | 12314.7 |
| 800 MHz | 7 | 14.9746 s | 14.9491 s | 15.0110 s | 0.0209 s | 0.0195 s | 38.0 | 16.327 | 11979.7 |

- fit on all accepted folds (n=28, clocks [800, 1000, 1200, 1350]): **F = 1.9500 s +-0.0438**, C = 10403.4 Mcycles +-44.9, max |residual| 0.1742 s, rms 0.0443 s
- fit on cell medians (n=4, clocks [800, 1000, 1200, 1350]): **F = 1.9154 s +-0.0844**, C = 10432.2 Mcycles +-86.5, max |residual| 0.0330 s, rms 0.0237 s
- closed form on the 1350/800 MHz endpoints alone: **F = 1.9511 s +-0.0625**, C = 10418.8 Mcycles +-51.5; the endpoint arms enter F with gains 2.455 and -1.455
- inverse-clock model: A/A timing floor 0.0478 s, cell-median max |residual| 0.0330 s, per-fold rms 0.0443 s -> **the exact two-parameter form holds, F is point-identified**
- three estimators of F: all_folds 1.9500 s +-0.0438, cell_medians 1.9154 s +-0.0844, two_clock_endpoints 1.9511 s +-0.0625 -> F bounded to 1.8310 - 2.0136 s
- structure: 1 distinct CIF over 28 accepted folds, max pairwise domain RMSD 0.00e+00 A against the 0.35 A bar (byte-identical in every arm)
- demand for 10.0 s at 1350 MHz (now 9.6563 s):
  - this size already folds in 9.6563 s, under the 10.0 s target, so the figures below are headroom rather than a demand
  - F untouched: 10867.5 Mcycles allowed, a cut of -464.0 Mcycles, **-4.5 %** of the work term
  - F cut to 1.0 s: 12150.0 Mcycles allowed, a cut of -1746.6 Mcycles, **-16.8 %**
  - cutting F to 1.0 s and touching no cycle at all: 8.7063 s

### across the two sizes

- F(512 aa) = 3.9830 s, F(298 aa) = 1.9500 s, difference 2.0330 s +-0.1260: **F is size DEPENDENT within 2 standard errors**
- work term 14665.0 Mcycles at 512 aa against 10403.4 at 298 aa: a work ratio of **1.410**
