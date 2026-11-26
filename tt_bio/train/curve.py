"""The matched-compute curve leg: one driver contract both sides read, and its reader.

G3's second half compares our loss curve with upstream's. That only means anything if the two
runs did the same work in the same order: same examples, same order, same ``t`` and noise per
step, same recycle depth, same number of steps. Upstream's own ``rfd4-train`` decides its data
order from its dataloader, which moves with any config change, so neither side may simply "run
its own training" and compare. Both sides read a :class:`CurvePlan` instead, and the plan's
digest is written into every log line's header: a comparison across two different plans is a
contract error, not a FAIL.

Recycle depth is upstream's own synchronized draw (``Proteina.handle_recycling`` with
``n_recycle_sync``): a Poisson(lambda) clipped to [1, n_max], seeded on the step index. Taking
it from the plan rather than from each process's RNG is what lets a replay land on upstream's
depth sequence without reaching into upstream's global RNG state.

The log is JSONL: one header object, then one object per step. Flat, append-only and readable
while a run is in flight, so a 2,000-step leg can be graded the moment it ends and inspected
before that.

:func:`grade_curve_files` applies the pre-registered bars in :mod:`tt_bio.train.parity`. It
refuses, rather than passes, when the seed-B control is absent: a cross-arch gap with no
same-arch seed control has no scale to be judged against.
"""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from tt_bio.train.parity import CURVE_MIN_STEPS, CurveMatch, curve_match

__all__ = ["CurvePlan", "StepSpec", "CurveLog", "CurveContractError", "read_curve",
           "grade_curve_files", "SIDES"]

#: The three logs a graded curve leg needs. ``gpu_b`` is upstream re-run at a second seed on the
#: same hardware, and is not optional.
SIDES = ("ours", "gpu_a", "gpu_b")


class CurveContractError(Exception):
    """The logs cannot be compared at all (different plans, missing side, broken log).

    Distinct from a FAIL: a FAIL is a measured verdict against the bars, this is "there is no
    measurement here". The grader exits 2 on it, never 0.
    """


@dataclass(frozen=True)
class CurvePlan:
    """What both sides execute, verbatim.

    ``example_ids`` is the data order: step *i* takes ``batch_size`` consecutive ids starting at
    ``i * batch_size``, wrapping, so the order is fixed by the plan and survives any change to
    upstream's sampler. ``seed`` derives the per-step ``t`` and noise seeds; a step's seeds
    depend on the step index alone, so a run that is interrupted and resumed at step k draws
    exactly what an uninterrupted run drew there.
    """

    steps: int
    crop: int
    example_ids: tuple
    batch_size: int = 1
    exploration_k: int = 4
    seed: int = 0x47330000
    warmup: int = 200
    n_recycle_max: int = 9
    n_recycle_lambda: float = 3.0
    dataset: str = "test_dataset"
    #: Parameters this plan's data cannot drive, each with upstream's reason, as
    #: ``((name, reason), ...)``. The G2 guard refuses a run in which a trainable parameter gets
    #: no gradient, and upstream's own run over a small id set trips it honestly: a feature no
    #: example carries returns ``zeros + 0.0 * weight.sum()`` (feature_factory.py:770), and the
    #: insertion head is driven only by examples with unindexed tokens. An exemption belongs in
    #: the plan, where it is hashed into the digest before the run, never discovered mid-run.
    exempt: tuple = ()
    note: str = ""

    def __post_init__(self):
        if self.steps < 1 or self.batch_size < 1 or self.crop < 1:
            raise ValueError("steps, batch_size and crop are all >= 1")
        if not self.example_ids:
            raise ValueError("a plan with no example ids has no data order")
        if not 0 <= self.warmup <= self.steps:
            raise ValueError(f"warmup {self.warmup} outside [0, {self.steps}]")
        object.__setattr__(self, "example_ids", tuple(str(e) for e in self.example_ids))
        ex = tuple((str(n), str(r)) for n, r in self.exempt)
        if any(not r for _, r in ex):
            raise ValueError("every exemption carries upstream's reason, not a bare name")
        object.__setattr__(self, "exempt", ex)

    @property
    def graded(self) -> bool:
        """Is this plan long enough for a reportable CURVE field (PLAN.md G3: >= 2,000 steps)?

        A shorter plan is legal and useful (plumbing smoke runs), it just may not be reported.
        """
        return self.steps >= CURVE_MIN_STEPS

    def digest(self) -> str:
        """sha256 over the canonical plan. Two logs with different digests are not comparable."""
        body = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()[:16]

    def step(self, i: int) -> "StepSpec":
        if not 0 <= i < self.steps:
            raise IndexError(f"step {i} outside the plan's 0..{self.steps - 1}")
        n = len(self.example_ids)
        ids = tuple(self.example_ids[(i * self.batch_size + j) % n] for j in range(self.batch_size))
        # Upstream's synchronized recycle draw, by step index (proteina.py handle_recycling).
        rng = np.random.default_rng(0x5EED0000 + i)
        depth = int(np.clip(rng.poisson(self.n_recycle_lambda), 1, self.n_recycle_max))
        return StepSpec(step=i, example_ids=ids, t_seed=self.seed + 2 * i,
                        noise_seed=self.seed + 2 * i + 1, n_recycle=depth)

    @classmethod
    def from_dict(cls, d: Mapping) -> "CurvePlan":
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(d) - known)
        if unknown:
            raise CurveContractError(f"plan carries unknown fields {unknown}: a field the writer "
                                     f"had and the reader does not would change the run silently")
        def conv(k, v):
            if k == "example_ids":
                return tuple(v)
            if k == "exempt":
                return tuple(tuple(e) for e in v)
            return v
        return cls(**{k: conv(k, v) for k, v in d.items()})


@dataclass(frozen=True)
class StepSpec:
    step: int
    example_ids: tuple
    t_seed: int
    noise_seed: int
    n_recycle: int


@dataclass
class CurveLog:
    """Writer. One header line, then one line a step, flushed as it goes."""

    path: Path
    plan: CurvePlan
    side: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.side not in SIDES:
            raise ValueError(f"side={self.side!r}, expected one of {SIDES}")
        self.path = Path(self.path)
        self._f = self.path.open("a", buffering=1)
        header = {"kind": "r4t-curve", "side": self.side, "plan_digest": self.plan.digest(),
                  "plan": asdict(self.plan), "host": socket.gethostname(),
                  "platform": platform.platform(), "started_unix": time.time(), **self.meta}
        self._f.write(json.dumps(header, sort_keys=True) + "\n")
        self._next = 0

    def record(self, step: int, loss: float, terms: Optional[Mapping[str, float]] = None,
               n_recycle: Optional[int] = None, **extra) -> None:
        """Record one step. ``step`` must be the next one: a gap is a contract error here, not
        a surprise for the reader half an hour later."""
        if step != self._next:
            raise CurveContractError(f"{self.path}: step {step} written after {self._next - 1}")
        spec = self.plan.step(step)
        if n_recycle is not None and n_recycle != spec.n_recycle:
            raise CurveContractError(f"step {step}: ran at recycle depth {n_recycle}, the plan "
                                     f"says {spec.n_recycle}; the two sides are not matched")
        row = {"step": step, "loss": float(loss), "n_recycle": spec.n_recycle,
               "example_ids": list(spec.example_ids), "unix": time.time()}
        if terms:
            row["terms"] = {k: float(v) for k, v in terms.items()}
        row.update(extra)
        self._f.write(json.dumps(row, sort_keys=True) + "\n")
        self._next += 1

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_curve(path) -> tuple:
    """A curve log -> (header, losses). Raises :class:`CurveContractError` on a broken log."""
    path = Path(path)
    try:
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    except OSError as e:
        raise CurveContractError(f"{path}: {e}") from e
    if not lines:
        raise CurveContractError(f"{path}: empty")
    try:
        header = json.loads(lines[0])
        rows = [json.loads(ln) for ln in lines[1:]]
    except json.JSONDecodeError as e:
        raise CurveContractError(f"{path}: not JSONL ({e})") from e
    if header.get("kind") != "r4t-curve" or header.get("side") not in SIDES:
        raise CurveContractError(f"{path}: first line is not a curve-log header")
    plan = CurvePlan.from_dict(header["plan"])
    if plan.digest() != header.get("plan_digest"):
        raise CurveContractError(f"{path}: header plan does not match its own digest")
    steps = [r.get("step") for r in rows]
    if steps != list(range(len(rows))):
        missing = sorted(set(range(len(rows) + 1)) - set(s for s in steps if s is not None))
        raise CurveContractError(f"{path}: steps are not 0..{len(rows) - 1} in order "
                                 f"(first gap or repeat near step {missing[0] if missing else '?'})")
    for r in rows:
        if r.get("n_recycle") != plan.step(r["step"]).n_recycle:
            raise CurveContractError(f"{path}: step {r['step']} logged recycle depth "
                                     f"{r.get('n_recycle')}, the plan says "
                                     f"{plan.step(r['step']).n_recycle}")
    header["plan_obj"] = plan
    return header, [float(r["loss"]) for r in rows]


def _check_reference_legs(headers, paths) -> None:
    """The two upstream legs are the scale the bar is read against, so they have to be the same
    run twice at a different seed: same device, fp32, TF32 off.

    TF32 is the one that would pass unnoticed. It is on by default for matmuls on an Ampere or
    later card, costs the mantissa 13 bits, and a leg that silently ran at 10-bit precision
    widens the A-B gap the bar divides by, so a TT leg would be graded against a looser bar than
    the one registered. A header that does not carry the fields (a log written before they
    existed) is refused too: the grader cannot confirm what it cannot see.
    """
    for side in ("gpu_a", "gpu_b"):
        meta = headers[side]
        for field in ("device", "dtype", "tf32"):
            if field not in meta:
                raise CurveContractError(
                    f"{paths[side]}: header has no {field!r}; a reference leg must say what it "
                    f"ran on and in what precision before it can be the scale for a bar")
        if meta["dtype"] != "float32":
            raise CurveContractError(f"{paths[side]}: reference leg ran in {meta['dtype']}, the "
                                     f"registered legs are float32")
        if meta["tf32"]:
            raise CurveContractError(f"{paths[side]}: reference leg ran with TF32 on, so it is "
                                     f"not the fp32 leg the bar was registered against")
    if headers["gpu_a"]["device"] != headers["gpu_b"]["device"]:
        raise CurveContractError(
            f"the seed control ran on different hardware than seed A "
            f"({headers['gpu_a']['device']} vs {headers['gpu_b']['device']}): "
            f"their gap then carries a hardware difference and is no longer a seed scale")


def grade_curve_files(ours, gpu_a, gpu_b=None) -> CurveMatch:
    """Grade three curve logs against the pre-registered G3 bars.

    Refuses (rather than grading) when the seed-B control is missing, when the three logs do not
    carry the same plan, when a side is mislabelled, or when a log is short of its own plan.
    """
    if gpu_b is None:
        raise CurveContractError(
            "no seed-B control: the curve bar is max(2 % of upstream, 1.5 x the seed-to-seed "
            "gap), so without upstream re-run at a second seed on the same hardware there is "
            "no scale for a cross-arch gap. Refusing to grade.")
    given = {"ours": ours, "gpu_a": gpu_a, "gpu_b": gpu_b}
    headers, curves = {}, {}
    for side, path in given.items():
        headers[side], curves[side] = read_curve(path)
        if headers[side]["side"] != side:
            raise CurveContractError(f"{path}: logged as side {headers[side]['side']!r}, passed "
                                     f"as {side!r}")
    digests = {side: h["plan_digest"] for side, h in headers.items()}
    if len(set(digests.values())) != 1:
        raise CurveContractError(f"the three logs ran different plans: {digests}")
    plan = headers["ours"]["plan_obj"]
    short = {side: len(c) for side, c in curves.items() if len(c) != plan.steps}
    if short:
        raise CurveContractError(f"log(s) do not cover the plan's {plan.steps} steps: {short}")
    _check_reference_legs(headers, given)
    return curve_match(curves["ours"], curves["gpu_a"], curves["gpu_b"], plan.warmup)


def summarise(losses: Sequence[float], window: int = 100) -> dict:
    """Trailing-mean summary of a curve, for a log tail or a state doc line."""
    x = np.asarray(losses, dtype=np.float64)
    if x.size == 0:
        return {"steps": 0}
    w = min(window, x.size)
    return {"steps": int(x.size), "first": float(x[0]), "last": float(x[-1]),
            f"mean_last_{w}": float(x[-w:].mean()),
            "finite": bool(np.isfinite(x).all())}


def read_dump_dir(path) -> tuple["CurvePlan", list[Path]]:
    """A directory of plan-step inputs (``curve_upstream.py --dump-dir``) -> its plan and its
    step files in step order.

    Every file is checked against ``plan.json``'s digest, so a leg can never train on one plan's
    tensors while logging another's: mixing two dumps in a directory, or pointing a run at a
    plan it was not dumped for, raises here rather than producing a log that grades.
    """
    path = Path(path)
    plan = CurvePlan.from_dict(json.loads((path / "plan.json").read_text()))
    files = sorted(path.glob("step*.npz"))
    if not files:
        raise CurveContractError(f"{path}: no step*.npz")
    want = plan.digest()
    for f in files:
        got = str(np.load(f, allow_pickle=False)["plan_digest"])
        if got != want:
            raise CurveContractError(f"{f.name} was dumped for plan {got}, plan.json is {want}")
    missing = [i for i in range(len(files)) if files[i].name != f"step{i:05d}.npz"]
    if missing:
        raise CurveContractError(f"{path}: step files are not a gapless run from 0 "
                                 f"(first bad index {missing[0]}: {files[missing[0]].name})")
    return plan, files
