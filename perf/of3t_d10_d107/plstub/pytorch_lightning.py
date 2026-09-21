"""Import stub. `grad_manager.py` names `pl` in exactly two places: the import at :18 and a
QUOTED annotation at :106 (`trainer: "pl.Trainer"`), which Python never evaluates. Nothing in
the executed path touches this module, and the attribute below raises if that ever changes.
"""


class _Never:
    def __getattr__(self, name):
        raise AssertionError(
            "the reference touched pytorch_lightning.%s -- the stub is no longer inert "
            "and this arm is not upstream's code any more" % name)


Trainer = _Never()
loggers = _Never()
