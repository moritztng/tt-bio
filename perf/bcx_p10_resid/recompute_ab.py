#!/usr/bin/env python3
"""`recompute=True` against `recompute=False`, interleaved round by round on one card.

`bcx-p10-resid` closed `bcx-p10-devmap`'s 2.603 s residual on the finding that the shipped
round runs gradient checkpointing: `EvoformerOnDevice` and `ExtraMsaOnDevice` both default
`recompute=True` (`tt_bio/bindcraft2.py:342`, `:575`), so the backward re-runs every block's
forward before it differentiates it. devmap's block harness ran `ckpt=False`, and the missing
forward pass -- 1.578 s at devmap's own split -- is 61 % of the residual.

That makes `recompute=False` a lever: the same gradient, the same blocks, one fewer pass over
them. What it costs is memory, which is the whole reason the flag exists, so this measures both.

`self.recompute` is read per call, so the arm is flipped at the round boundary and the two arms
alternate inside ONE process, on ONE card, against one campaign. Device memory is sampled at
the phase boundaries only -- each one already ends in `trunk.sync()`, so the read does not drain
a pipeline that is not already drained.

Nothing here changes what the model computes. `recompute` is exact recomputation: the forward
it re-runs is the forward it saved, and the gradient is the same one.
"""
import argparse
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
for _p in (str(_ROOT / "perf" / "bcx_round"), str(_ROOT / "perf" / "bcx_predictor"),
           str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import meter as M                                                      # noqa: E402

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--out", required=True)
_ap.add_argument("--first", default="ckpt", choices=["ckpt", "nockpt"],
                 help="which arm round 2 runs; round 1 always runs the default")
_known, _rest = _ap.parse_known_args()
FIRST = _known.first
sys.argv = [sys.argv[0], "--out", _known.out] + _rest

SPLICES = []          # every EvoformerOnDevice / ExtraMsaOnDevice built in this process
DEV = [None]


def dram():
    """Bytes allocated in DRAM across all banks, the idiom `perf/ptx_integrate/step.py` uses."""
    import ttnn
    if DEV[0] is None:
        return None
    mv = ttnn.get_memory_view(DEV[0], ttnn.BufferType.DRAM)
    return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)


class ArmMeter(M.Meter):
    """`M.Meter` with the arm flipped at the round boundary."""

    arm = "ckpt"

    def on_sequence_gradients_enter(self):
        r = self.entries + 1
        # Round 1 carries the jit compile and the first trunk load; it runs the default and is
        # dropped by the analysis either way. From round 2 the arms alternate.
        if r == 1:
            arm = "ckpt"
        else:
            first = (r % 2 == 0)
            arm = FIRST if first else ("nockpt" if FIRST == "ckpt" else "ckpt")
        ArmMeter.arm = arm
        want = (arm == "ckpt")
        for s in SPLICES:
            s.recompute = want
        M.EVENTS.append({"kind": "arm", "phase": arm, "t0": time.time(), "round": r,
                         "recompute": want, "dram_bytes": dram()})
        super().on_sequence_gradients_enter()          # may raise StopAfterRounds


def install(bindcraft2, meter):
    for module, cls in (("evoformer", bindcraft2.EvoformerOnDevice),
                        ("extra_msa", bindcraft2.ExtraMsaOnDevice)):
        orig_init = cls.__init__

        def make_init(orig_init):
            def __init__(self, *a, **kw):
                orig_init(self, *a, **kw)
                SPLICES.append(self)
            return __init__
        cls.__init__ = make_init(orig_init)

        for name in ("_taped", "_backward"):
            orig = getattr(cls, name)

            def make(name, orig, module):
                def wrapper(self, *a, **kw):
                    if DEV[0] is None:
                        for t in self.pool._trunks.values():
                            DEV[0] = t.device
                            break
                    t0 = time.time()
                    try:
                        return orig(self, *a, **kw)
                    finally:
                        if DEV[0] is None:
                            for t in self.pool._trunks.values():
                                DEV[0] = t.device
                                break
                        M.EVENTS.append({
                            "kind": "mem", "phase": name.lstrip("_"), "module": module,
                            "t0": t0, "t1": time.time(),
                            "dt": round(time.time() - t0, 6),
                            "arm": ArmMeter.arm, "recompute": self.recompute,
                            "dram_bytes": dram(), "round": meter.entries})
                return wrapper
            setattr(cls, name, make(name, orig, module))


def main():
    import run_round
    from tt_bio import bindcraft2

    M.Meter = ArmMeter
    real_install = M.install

    def patched(meter, splice_mod, predictor_cls, trajectory_mod, seqopt_mod):
        install(bindcraft2, meter)
        real_install(meter, splice_mod, predictor_cls, trajectory_mod, seqopt_mod)
    M.install = patched

    real_dump = M.dump

    def dump(path, stamp):
        stamp["arms"] = {"first_from_round_2": FIRST}
        stamp["splices"] = len(SPLICES)
        return real_dump(path, stamp)
    M.dump = dump

    run_round.main()


if __name__ == "__main__":
    main()
