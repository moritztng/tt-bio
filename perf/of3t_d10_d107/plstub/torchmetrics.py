"""Import stub. `grad_manager.py` builds `MeanMetric`/`MaxMetric` only under
`log_grad_norm` (:93-94), and this harness runs with `log_grad_norm=False`, so neither is
ever constructed. Constructing one raises rather than returning a quiet fake, which keeps the
inertness a fact rather than an assumption.
"""


class _NeverBuilt:
    def __init__(self, *a, **k):
        raise AssertionError(
            "the reference constructed a torchmetrics metric -- the stub is no longer "
            "inert and this arm is not upstream's code any more")


class MeanMetric(_NeverBuilt):
    pass


class MaxMetric(_NeverBuilt):
    pass
