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
    #: The seed-control knob, and the ONE field the digest does not cover. The seed-B leg is the
    #: same plan -- same data order, same steps, same recycle depths -- re-drawn at a different
    #: ``t`` and noise sequence, so it has to compare equal to seed A as a plan while running
    #: different draws. Everything else about a leg is hashed; this is enforced instead, by
    #: :func:`grade_curve_files`: ``ours`` and ``gpu_a`` must carry 0 (matched compute) and
    #: ``gpu_b`` must not (or there is no control). Measured on 2026-11-26: without it, upstream's
    #: own ``--seed-offset`` changed nothing a step reads, and the two A100 legs came out
    #: byte-identical for 111 steps.
    seed_offset: int = 0
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
        """sha256 over the canonical plan, ``seed_offset`` excluded (see the field's note). Two
        logs with different digests are not comparable."""
        body = asdict(self)
        body.pop("seed_offset", None)
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    def step(self, i: int) -> "StepSpec":
        if not 0 <= i < self.steps:
            raise IndexError(f"step {i} outside the plan's 0..{self.steps - 1}")
        n = len(self.example_ids)
        ids = tuple(self.example_ids[(i * self.batch_size + j) % n] for j in range(self.batch_size))
        # Upstream's synchronized recycle draw, by step index (proteina.py handle_recycling).
        rng = np.random.default_rng(0x5EED0000 + i)
        depth = int(np.clip(rng.poisson(self.n_recycle_lambda), 1, self.n_recycle_max))
        base = self.seed + 0x01000000 * self.seed_offset
        return StepSpec(step=i, example_ids=ids, t_seed=base + 2 * i,
                        noise_seed=base + 2 * i + 1, n_recycle=depth)

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
    start: int = 0

    def __post_init__(self):
        if self.side not in SIDES:
            raise ValueError(f"side={self.side!r}, expected one of {SIDES}")
        self.path = Path(self.path)
        if self.start:
            self._resume()
            return
        self._f = self.path.open("a", buffering=1)
        header = {"kind": "r4t-curve", "side": self.side, "plan_digest": self.plan.digest(),
                  "plan": asdict(self.plan), "host": socket.gethostname(),
                  "platform": platform.platform(), "started_unix": time.time(), **self.meta}
        self._f.write(json.dumps(header, sort_keys=True) + "\n")
        self._next = 0

    def _resume(self) -> None:
        """Continue a log from step ``start`` (a run resumed from the checkpoint after step
        ``start - 1``). Rows at or past ``start`` were written after that checkpoint and are
        recomputed by the resumed run, so they are dropped; the file must already hold every
        step before ``start`` for this plan and side, or the resume is refused."""
        header, losses = read_curve(self.path)
        if header["plan_digest"] != self.plan.digest() or header["side"] != self.side:
            raise CurveContractError(f"{self.path}: resuming {self.side} plan {self.plan.digest()} "
                                     f"onto a {header['side']} log of plan {header['plan_digest']}")
        if len(losses) < self.start:
            raise CurveContractError(f"{self.path}: resume at step {self.start} but the log ends "
                                     f"at step {len(losses) - 1}")
        lines = [ln for ln in self.path.read_text().splitlines() if ln.strip()]
        last = max(i for i, ln in enumerate(lines)
                   if json.loads(ln).get("kind") == "r4t-curve")
        keep = lines[:last + self.start + 1]  # dead attempts stay, they are the run's record
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text("\n".join(keep) + "\n")
        tmp.replace(self.path)
        self._f = self.path.open("a", buffering=1)
        self._next = self.start

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


def _segment(path, lines) -> tuple:
    """Split a log into attempts and return (header, row objects) of the LAST one.

    A leg that died and was relaunched into the same file leaves its dead attempt in place:
    header, a few rows, then the live attempt's own header. The grader reads the last attempt
    and reports the dead ones, but only once every attempt agrees on the plan and the side --
    two different runs concatenated is a contract error, not a log with a dead prefix.
    """
    try:
        objs = [json.loads(ln) for ln in lines]
    except json.JSONDecodeError as e:
        raise CurveContractError(f"{path}: not JSONL ({e})") from e
    heads = [i for i, o in enumerate(objs)
             if isinstance(o, dict) and o.get("kind") == "r4t-curve"]
    if not heads or heads[0] != 0:
        raise CurveContractError(f"{path}: first line is not a curve-log header")
    bounds = heads + [len(objs)]
    attempts = [(objs[bounds[k]], objs[bounds[k] + 1:bounds[k + 1]]) for k in range(len(heads))]
    first = attempts[0][0]
    for h, _ in attempts[1:]:
        if (h.get("plan_digest"), h.get("side")) != (first.get("plan_digest"), first.get("side")):
            raise CurveContractError(
                f"{path}: attempt {h.get('side')!r}/{h.get('plan_digest')} appended to a log of "
                f"{first.get('side')!r}/{first.get('plan_digest')}; two runs in one file")
    header, rows = attempts[-1]
    header["superseded"] = [len(r) for _, r in attempts[:-1]]
    return header, rows


def read_curve(path) -> tuple:
    """A curve log -> (header, losses). Raises :class:`CurveContractError` on a broken log."""
    path = Path(path)
    try:
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    except OSError as e:
        raise CurveContractError(f"{path}: {e}") from e
    if not lines:
        raise CurveContractError(f"{path}: empty")
    header, rows = _segment(path, lines)
    if header.get("side") not in SIDES:
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


#: What every curve leg must declare, and the matched leg must share with seed A.
UPDATE_RULE = ("optimizers", "clip")


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


def _check_seed_control(headers, curves, paths) -> None:
    """The seed-B leg has to be a different draw of the same plan, and has to show it.

    Two ways it silently is not. The offset can be absent, which makes B a second copy of A; and
    the offset can be set on a leg that must not have one, which breaks the matched compute
    ``ours`` and ``gpu_a`` are compared on. Both are refusals, not FAILs. The identity check at
    the end is the backstop: whatever the headers claim, two curves that agree to the bit are one
    run reported twice, and a bar divided by their gap would be the 2 %-of-upstream branch alone.
    """
    off = {side: h["plan_obj"].seed_offset for side, h in headers.items()}
    for side in ("ours", "gpu_a"):
        if off[side]:
            raise CurveContractError(
                f"{paths[side]}: ran at seed offset {off[side]}, but {side} carries the matched "
                f"compute and must draw the plan's own t and noise (offset 0)")
    if not off["gpu_b"]:
        raise CurveContractError(
            f"{paths['gpu_b']}: seed offset 0, the same draw as seed A, so it is a copy of the "
            f"A leg and not a seed control. Re-run it at a non-zero offset.")
    a, b = np.asarray(curves["gpu_a"]), np.asarray(curves["gpu_b"])
    n = min(a.size, b.size)
    if n and np.array_equal(a[:n], b[:n]):
        raise CurveContractError(
            f"{paths['gpu_b']}: identical to seed A over all {n} logged steps, so the seed-to-"
            f"seed gap the bar is scaled by is exactly zero: there is no control here")


def _check_update_rule(headers, paths) -> None:
    """All three legs must train with the same update rule, or the curve compares optimizers.

    "Matched compute" is the same tensors AND the same step: a leg that runs plain AdamW against
    upstream's Muon + Adam descends a different path from the same start, and its curve gap
    then measures the optimizer, not the port. That is exactly how the first graded TT leg
    (2026-11-26, AdamW at 1e-4 against GroupedMuon + Adam) read 11 % high from step 7 on. So
    every leg declares its optimizers and its clip, and the grader refuses a mismatch or a leg
    that does not say. The names are upstream's (``type(opt).__name__``): a TT leg that
    implements Muon + Adam declares ``["GroupedMuon", "Adam"]``.
    """
    for side, meta in headers.items():
        for field in UPDATE_RULE:
            if field not in meta:
                raise CurveContractError(
                    f"{paths[side]}: header has no {field!r}; a leg must declare its update rule "
                    f"before its curve can be compared at matched compute")
    for side in ("ours", "gpu_b"):
        for field in UPDATE_RULE:
            if headers[side][field] != headers["gpu_a"][field]:
                raise CurveContractError(
                    f"{paths[side]}: {field} {headers[side][field]!r} is not seed A's "
                    f"{headers['gpu_a'][field]!r}, so the curves compare two update rules")


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
    _check_seed_control(headers, curves, given)
    plan = headers["ours"]["plan_obj"]
    short = {side: len(c) for side, c in curves.items() if len(c) != plan.steps}
    if short:
        raise CurveContractError(f"log(s) do not cover the plan's {plan.steps} steps: {short}")
    _check_reference_legs(headers, given)
    _check_update_rule(headers, given)
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
