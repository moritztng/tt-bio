## Gradient-path numbering minus predict-path numbering, per binder draw

| binder | target | target numbered from | draws | in-window pairs | in-window share | i_pTM gap (each draw) | median |
|---|---|---|---|---|---|---|---|
| 32 | 96 | 1 | 4 | 1552 | 50.5% | +0.2317, +0.3433, +0.3807, +0.4749 | +0.3620 |
| 32 | 96 | 4000 | 1 | 0 | 0.0% | +0.0000 | +0.0000 |
| 32 | 160 | 1 | 4 | 1552 | 30.3% | +0.2283, +0.2857, +0.3812, +0.4090 | +0.3335 |
| 32 | 160 | 4000 | 1 | 0 | 0.0% | +0.0000 | +0.0000 |
| 32 | 288 | 1 | 3 | 1552 | 16.8% | +0.3213, +0.3847, +0.4237 | +0.3847 |
| 96 | 128 | 1 | 2 | 5712 | 46.5% | +0.1281, +0.3202 | +0.2242 |

## Other metrics, same pairs (median over draws)

| binder | target | i_pTM | pTM | pLDDT | binder pLDDT |
|---|---|---|---|---|---|
| 32 | 96 | +0.3620 | +0.0280 | +0.0326 | +0.1061 |
| 32 | 160 | +0.3335 | -0.0256 | -0.0164 | +0.0310 |
| 32 | 288 | +0.3847 | -0.0056 | -0.0046 | -0.0189 |
| 96 | 128 | +0.2242 | +0.0625 | +0.0165 | +0.0346 |

## Identities (bit-for-bit checks)

- sequence_gradients vs predict with the chain break made the identity: 4 of 4 metrics bit-identical, largest difference 0
- sequence_gradients fed the chain-broken numbering vs predict: 4 of 4 metrics bit-identical, largest difference 0
- single chain, sequence_gradients vs predict: 2 of 4 metrics bit-identical, largest difference 7.57e-09
- zero-window control, the two numberings against each other: 4 of 4 metrics bit-identical, largest difference 0
