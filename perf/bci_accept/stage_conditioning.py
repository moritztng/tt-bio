"""How much of the sequence-gradient direction survives a bf16-level error in the trunk's output
cotangent, stage by stage.

Issue #17: the on-card design loop holds i_pTM through `anneal` and collapses at `harden`. The
stages differ in one thing that matters to a noisy trunk. BindCraft 2 differentiates the loss all
the way back to the sequence logits, and the last link of that chain is its own, on the host in
float32:

    features = sequence_features_from_logits(z, softmax_weight, one_hot_weight, temperature, 2.0)

The trunk only supplies the cotangent g = dL/dfeatures. Everything the optimizer then does with
dL/dz is direction-only: `normalize_sequence_gradient` divides by the norm, so a trunk error that
is purely a scale is free and a trunk error that rotates the vector is not.

What this measures: feed the exact g and a perturbed g*(1 + eps*n) through that last link, take
BindCraft 2's own step with each, and count the designed positions that end the round on a different
amino acid. That is what the next forward pass sees. The amplification is a property of the stage's own
(softmax_weight, one_hot_weight, temperature) and of the logits, and needs no card and no trunk.

Run with --logits <npz> to use logits captured from a real trajectory (`capture_logits.py`);
without it, synthetic logits at a stated spread, which is a calibration, not a measurement.
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np

from bindcraft.sequence_optimization import sequence_features_from_logits, normalize_sequence_gradient

LOGIT_SCALE = 2.0

#: (softmax_weight, one_hot_weight, temperature) at the point of each stage that matters. `screen`
#: and `refine` ramp softmax_weight (bindcraft/trajectory.py:215-216), `anneal` anneals temperature
#: on a quadratic schedule so it is still at 0.26 at its own midpoint, and `harden` sits at full
#: saturation for all five of its rounds. `mutate` takes no gradient at all
#: (bindcraft/settings.py:610 drops it), which is why both arms collapse there.
STAGES = {
    "screen_start":  (0.0,  0.0, 1.0),
    "screen_end":    (0.9,  0.0, 1.0),
    "refine_end":    (1.0,  0.0, 1.0),
    "anneal_mid":    (1.0,  0.0, 0.26),
    "anneal_end":    (1.0,  0.0, 0.01),
    "harden":        (1.0,  1.0, 0.01),
}


def logit_gradient(logits, cotangent, softmax_weight, one_hot_weight, temperature):
    """dL/dlogits given dL/dfeatures, through BindCraft 2's own feature map."""
    def features(z):
        return sequence_features_from_logits(
            z,
            jnp.asarray(softmax_weight), jnp.asarray(one_hot_weight),
            jnp.asarray(temperature), jnp.asarray(LOGIT_SCALE),
        )
    _, vjp = jax.vjp(features, logits)
    return vjp(cotangent)[0]


#: BindCraft 2's own step: SGD at lr 0.1 on the normalized gradient, damped by
#: (1 - softmax_weight + softmax_weight*temperature) (sequence_optimization.py:98-99).
LEARNING_RATE = 0.1


def sequence_update(logits, cotangent, stage):
    softmax_weight, _, temperature = stage
    gradient = normalize_sequence_gradient(logit_gradient(logits, cotangent, *stage))
    damping = 1.0 - softmax_weight + softmax_weight * temperature
    return logits - LEARNING_RATE * damping * gradient


def decided_by_noise(logits, cotangent, stage, eps, noise):
    """What fraction of the designed positions end a round on a different amino acid because the
    trunk's cotangent carried a relative error of `eps`.

    This is the quantity the next forward pass sees. A cosine on the gradient is useless here: a
    0.4% relative perturbation always leaves cos ~ 1 - O(eps^2). What changes the sequence is the
    argmax, and whether a position's top two amino acids are close enough that the low bits pick
    the winner."""
    exact = sequence_update(logits, cotangent, stage)
    perturbed = sequence_update(logits, cotangent * (1.0 + eps * noise), stage)
    return float(jnp.mean(jnp.argmax(exact, -1) != jnp.argmax(perturbed, -1)))


def synthetic_logits(key, residues, spread):
    """Logits with a realistic per-residue spread. BindCraft 2 starts them at about 0.01 and SGD at
    lr 0.1 damped by (1 - sw + sw*T) moves them by about 1e-3 a round, so a hardened trajectory sits
    at a few hundredths. `spread` is the standard deviation across the 20 amino acids."""
    return jax.random.normal(key, (residues, 20)) * spread


def margin_report(logits, cotangent, stage, eps, noise):
    """Why a stage is or is not sensitive, in one ratio.

    A position changes amino acid when the noise-induced difference in its update exceeds the gap
    between its own top two logits. Both are in logit units, so their ratio says how far a stage is
    from being decided by the trunk's low bits, and it says it without waiting for a flip to be
    rare enough that counting them stops being informative."""
    exact = sequence_update(logits, cotangent, stage)
    perturbed = sequence_update(logits, cotangent * (1.0 + eps * noise), stage)
    ordered = jnp.sort(logits, axis=-1)
    gap = ordered[..., -1] - ordered[..., -2]
    disturbance = jnp.max(jnp.abs(perturbed - exact), axis=-1)
    return float(jnp.median(gap)), float(jnp.median(disturbance))


def captured_stages(path):
    """(name, (softmax_weight, one_hot_weight, temperature), logits) from a capture_logits.py npz.

    The capture keys each stage as `<stage>:<chain>` with a sibling `<stage>:<chain>:params`, so the
    stage parameters come from the trajectory that produced the logits rather than from this file's
    own table. A stage whose params are missing is skipped rather than guessed at."""
    stages = []
    for key in sorted(captured_keys(path)):
        params = f"{key}:params"
        if params not in path:
            continue
        softmax_weight, one_hot_weight, temperature = (float(v) for v in path[params])
        stages.append((key, (softmax_weight, one_hot_weight, temperature), jnp.asarray(path[key])))
    return stages


def captured_keys(path):
    return [k for k in path.files if not k.endswith(":params")]


def report(label, stages, eps_values, repeats):
    print(f"== {label} ==")
    print("stage".ljust(24) + "".join(f"{'eps=2^%d' % int(np.log2(e)):>12}" for e in eps_values))
    margins = []
    for name, stage, logits in stages:
        cells = []
        for eps in eps_values:
            flips = []
            for repeat in range(repeats):
                cotangent = jax.random.normal(jax.random.key(repeat), logits.shape)
                noise = jax.random.normal(jax.random.key(1000 + repeat), logits.shape)
                flips.append(decided_by_noise(logits, cotangent, stage, eps, noise))
            cells.append(f"{100 * np.mean(flips):11.2f}%")
        print(name.ljust(24) + "".join(cells))
        # One cotangent draw moves this ratio by an order of magnitude, so it is averaged over the
        # same repeats as the flip counts and reported as a median, not a single reading.
        samples = [margin_report(
            logits, jax.random.normal(jax.random.key(repeat), logits.shape), stage, eps_values[0],
            jax.random.normal(jax.random.key(1000 + repeat), logits.shape))
            for repeat in range(repeats)]
        margins.append((name, (float(np.median([g for g, _ in samples])),
                               float(np.median([d for _, d in samples])))))
    print()
    print("why, at eps=2^%d: a position flips when the noise moves its update by more than the"
          % int(np.log2(eps_values[0])))
    print("gap to its own runner-up amino acid. Both in logit units.\n")
    print("stage".ljust(24) + f"{'top-2 gap':>12}{'noise moves':>14}{'gap / noise':>14}")
    for name, (gap, disturbance) in margins:
        ratio = gap / disturbance if disturbance > 0 else float("inf")
        print(name.ljust(24) + f"{gap:>12.2e}{disturbance:>14.2e}{ratio:>14.1f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residues", type=int, default=90)
    parser.add_argument("--spread", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    parser.add_argument("--eps", type=float, nargs="+", default=[2 ** -8, 2 ** -11, 2 ** -24])
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--logits", default=None, help="npz of captured per-stage logits")
    args = parser.parse_args()

    print("eps: 2^-8 = bfloat16 mantissa, 2^-11 = float16, 2^-24 = float32\n")
    if args.logits:
        report(f"captured logits {args.logits}", captured_stages(np.load(args.logits)),
               args.eps, args.repeats)
        return
    for spread in args.spread:
        stages = [(name, stage, synthetic_logits(jax.random.key(0), args.residues, spread))
                  for name, stage in STAGES.items()]
        report(f"synthetic logits, {args.residues} residues, spread {spread}",
               stages, args.eps, args.repeats)


if __name__ == "__main__":
    main()
