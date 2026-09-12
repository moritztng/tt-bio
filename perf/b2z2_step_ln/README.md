# One `layer_norm(s)` for a whole DiffusionTransformer stack

`BOLTZ2_ADALN_SHARED_SNORM`, default **off**. On Wormhole it takes the Boltz-2 diffusion step from
**41.5766 ms to 40.0037 ms, 1.03932x**, by deleting 47 of the step's 114 LayerNorm programs.

Every AdaLN in the token stack normalises the same `s` and differs only by a per-channel
`s_norm.weight` and the projection that consumes it. Since

    layer_norm(s, weight=w) @ W  ==  layer_norm(s) @ (w[:, None] * W)

the stack can normalise `s` once and let each site carry its weight inside the matrix it already
multiplies by. The atom stack is excluded because it already memoises the same value for the whole
rollout, so a shared norm there would add a program and save none.

**It is not bit-exact** — `bf16(w*W)` is not `bf16(w)*bf16(W)` — but it is the more accurate of the
two arms. Scored against a torch fp32 reference at all 48 AdaLNs, with the real `s` lifted off a
settled step, the folded form is closer on every statistic and its worst layer improves by 22-30 %.
At the fold it moves each pseudo-domain 0.28-0.35 A at 512 aa and 0.28-0.30 A on the 298 aa control,
against a 0.60 A bar and a 1.6-1.8 A seed floor, with structure quality against the experimental
CDK2 structure unchanged.

## Running it

    ln_sites.py        --out <json>              # which source line each LayerNorm comes from
    step_ab.py --mode time|parity --out <json>   # interleaved A/B on one settled step
    fold_probe.py      --out <json>              # both arms against fp32, per AdaLN
    fold_seeds_snorm.py --out <json> --cifdir <dir> --seeds 0,1 --sizes 512,298

The folds are scored by `perf/b2z2_fusebias/score.py`, unchanged:

    score.py <cifdir> --runs folds_wh_c11.json --split 298 --out score_wh_c11.json

All committed numbers are Wormhole, whglx card 11, 8x9 grid, 200 sampling steps, 3 recycles, full
35-row MSA. The write-up is `~/.coworker/state/b2z2-step-layernorm-fusion.md`.
