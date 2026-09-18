"""A small trunk that is real work through the shipped ops, for measuring the launcher.

What is being measured here is the LAUNCHER, so the model is the smallest thing that makes
every part of the training path do its real job: the sites route through ``tt_bio.ops.linear``
so the LoRA census finds them and the adapters attach for real, the tape records and replays a
genuine backward, the gradients are real device tensors of a real size, and the optimizer keeps
real fp32 masters. Nothing here is a stub or a stand-in; it is a stack of projections rather
than a protein model, which is a statement about what the numbers mean and not about how they
were produced.

The loss is the shipped ``af3`` row with ``mse`` as its only live term, so the gradient seed
comes back through the same code a Protenix fine-tune uses. It is not a scientific objective on
this input and it is not presented as one: a random target gives a real gradient of a real
magnitude, which is what a scaling measurement needs and all it needs.

``Dataset.device`` is resolved LAZILY and that is load-bearing, not a style choice. A
data-parallel run is one process per chip because ``TT_VISIBLE_DEVICES`` is read at
``import ttnn``, so the process that spawns the ranks must hold no card when it does; a dataset
that opened a device in its constructor would take a chip in the driver and the launcher refuses
to spawn. Every dataset that wants to work under the launcher has to defer the open this way.
"""

import numpy as np

from tt_bio import ops


def block(x, w1, w2):
    h = ops.linear(x, w1)      # adaptable site 1
    return ops.linear(h, w2)   # adaptable site 2


class Trunk:
    """``blocks`` pairs of projections, and a final one down to 3 coordinate channels.

    The weights are the frozen trunk: built once, never in the optimizer, and the adapters are
    what train. That is the shape LoRA fine-tuning actually has, so the gradient the launcher
    exchanges is an adapter gradient and not a whole model's.
    """

    def __init__(self, tokens, channels, blocks, seed=0):
        self.tokens, self.channels, self.blocks = tokens, channels, blocks
        self.seed = seed
        self._w = None

    def weights(self, device):
        if self._w is None:
            import ttnn
            from tt_bio.train.tensors import to_device
            rng = np.random.default_rng(self.seed)
            c = self.channels
            # 1/sqrt(c) keeps the activation scale flat through the depth, so a 48-block stack
            # neither saturates bf16 nor decays into it and the gradient stays a real number.
            s = 1.0 / np.sqrt(c)
            self._w = [[to_device(rng.normal(0, s, (c, c)).astype(np.float32), device,
                                  dtype=ttnn.bfloat16) for _ in (0, 1)]
                       for _ in range(self.blocks)]
            self._out = to_device(rng.normal(0, s, (c, 3)).astype(np.float32), device,
                                  dtype=ttnn.bfloat16)
        return self._w, self._out

    def __call__(self, batch):
        """``{"pred_xyz": Tensor}``. One example per call, which is the micro batch here."""
        device = batch["device"]
        w, wout = self.weights(device)
        x = batch["x"]
        for w1, w2 in w:
            x = block(x, w1, w2)
        return {"pred_xyz": ops.linear(x, wout)}


class Dataset:
    """``__len__``, ``tokens``, ``device``, ``batch(indices)`` -- the four-member contract.

    ``batch`` returns one example because the micro batch is one here: the axis under test is
    the data-parallel one, and folding a second batch axis into the same step would put two
    things in one number.
    """

    def __init__(self, n, tokens, channels, seed=0):
        self.n, self.tokens, self.channels, self.seed = n, tokens, channels, seed
        self._device = None
        self._cache = {}

    @property
    def device(self):
        """Opened on FIRST USE, never in the constructor. See the module docstring."""
        if self._device is None:
            from tt_bio.tenstorrent import get_device
            self._device = get_device()
        return self._device

    def __len__(self):
        return self.n

    def batch(self, indices):
        import ttnn
        from tt_bio.train.tensors import to_device
        idx = list(indices)
        if len(idx) != 1:
            raise ValueError(f"this dataset's micro batch is one example, got {len(idx)}. "
                             f"Pass global_batch equal to the number of chips")
        i = idx[0]
        if i not in self._cache:
            rng = np.random.default_rng([self.seed, i])
            x = rng.normal(0, 1, (self.tokens, self.channels)).astype(np.float32)
            true = rng.normal(0, 10, (self.tokens, 3)).astype(np.float32)
            self._cache[i] = (x, true)
        x, true = self._cache[i]
        return {"device": self.device,
                "x": to_device(x, self.device, dtype=ttnn.bfloat16),
                # The label stays on the host: the loss is host float64 numpy, which is what
                # lets the objective return a gradient seed per device output.
                "true_xyz": true.astype(np.float64),
                "coord_mask": np.ones(self.tokens, np.float64)}


def build(tokens=2048, channels=2048, blocks=24, examples=64, seed=0):
    """``(forward, dataset)``, the pair every training entry point takes."""
    return (Trunk(tokens, channels, blocks, seed=seed),
            Dataset(examples, tokens, channels, seed=seed))
