"""``Mesh`` -- named axes over the chips in one box, and the reducer ``step()`` consumes.

Distribution appears in user code exactly twice: a ``Mesh`` built once, and
``data_parallel=mesh.axis("dp")`` handed to the optimizer. Everything else is the
optimizer's problem, and that placement is the whole design rather than a convenience.

tt-train is the argument, from its own inconsistency: ``SFTTrainer`` requires the user to
remember a ``DDPCallback`` while ``GRPOTrainer`` reduces inside its own step. Forget the
callback and you train N diverging replicas behind a loss curve that still looks right,
because each replica's loss is real -- it is just a different model's loss. There is no
warning available for that at the callback layer, because a missing callback is
indistinguishable from a user who meant single-chip. So the reduce moves into ``step()``,
and the mesh's width is what tells ``step()`` whether one was required:
``opt.step()`` raises :class:`UnreducedGradients` when the DP axis is wider than one chip and
no reducer was given.

**Global batch is an explicit required argument and is never derived from device count.**
``accelerate``'s data loader silently redefines it in terms of device count
(``data_loader.py:347-348``) and its scheduler follows (``scheduler.py:69-76``), so the same
script trains a different recipe on a different box. It is the axis a published recipe pins,
which makes it the last thing that should move on its own.

Single box, up to 4 chips. Multi-host is out of scope and it is blocked on cabling rather than
on code: 20 MB/s over WiFi to qb2 makes a per-step gradient exchange cost more than the step.
``Mesh.auto()`` reports ``hosts=1`` today, and the interface does not change shape when that
number changes -- an axis is an axis whether its chips share a host or not.

Tensor parallelism is deliberately absent. A full training replica is 14.8 % of one chip's
measured 34.23 GB, so there is no memory argument for it; it returns only if something later
forces it, and only as an explicit exception to Moritz's standing data-parallel direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

__all__ = ["Mesh", "Axis", "UnreducedGradients", "MAX_CHIPS_ONE_BOX"]


# One box. Not a hardware limit, a measured-scope limit: 1.87x on two chips is the only
# multi-chip scaling point we have, and beyond one box the wire is the blocker.
MAX_CHIPS_ONE_BOX = 4


class UnreducedGradients(RuntimeError):
    """Raised by ``step()`` when a wide DP axis had no reducer.

    The failure this exists to prevent is silent: N replicas each taking their own gradient's
    step diverge immediately, and every replica's loss curve looks healthy on the way down.
    """


@dataclass(frozen=True)
class Axis:
    """One named axis of a mesh: a name, the chips on it, and how to reduce across it.

    Hashable and frozen so it can be compared and logged. ``width == 1`` is the honest
    single-chip case and ``reduce`` is then a no-op, which is what lets the same user code run
    on one chip and on four without a branch.
    """

    name: str
    device_ids: tuple
    mesh: "Mesh" = None

    @property
    def width(self) -> int:
        return len(self.device_ids)

    def reduce(self, tensors: Sequence) -> list:
        """Sum ``tensors`` across the axis, one entry per chip, in place of each.

        Sum rather than mean: the divisor is the global batch, which the caller pinned, and
        dividing here would hide it. ``ttnn.all_reduce`` is Sum-only at our pin anyway
        (``all_reduce.cpp:47``), so mean would be a second op with a number in it.
        """
        if self.width == 1:
            return list(tensors)
        import ttnn
        if len(tensors) != self.width:
            raise ValueError(f"axis {self.name!r} is {self.width} chips wide, got "
                             f"{len(tensors)} tensors to reduce")
        # bf16 is the only dtype straight through the fast path; fp32 takes a separate route
        # and bfloat8_b round-trips through bf16 and is lossy by construction (train-r5
        # COLLECTIVES). A gradient arriving here in another dtype is a caller bug worth
        # naming, not something to cast around silently.
        bad = [str(t.dtype) for t in tensors if t.dtype != ttnn.bfloat16]
        if bad:
            raise ValueError(
                f"all_reduce across axis {self.name!r} wants bfloat16; got {sorted(set(bad))}. "
                f"bfloat8_b round-trips through bfloat16 and loses bits by construction, and "
                f"fp32 takes a different route -- cast deliberately rather than here")
        return [ttnn.all_reduce(t) for t in tensors]

    def reduce_all(self, per_param: Dict[str, Sequence]) -> dict:
        """Sum every parameter's gradient across the axis. Name in, one summed gradient out.

        The per-name loop over :meth:`reduce` is the whole of it here, because a mesh-device
        ``all_reduce`` is already one call per tensor. It exists as its own method so an axis
        whose width spans PROCESSES rather than chips a single process holds can override it
        and exchange the whole parameter set in one message: across processes the cost is the
        barrier and not the bytes, and an adapter has dozens of parameters per step.

        A sum comes back wherever the axis summed it -- on the device from an on-device
        collective, on the host from one that reduces there. The optimizer reads it with
        ``to_host`` either way, so an axis that already has the number on the host hands it
        over as it is instead of paying a PCIe round trip to be uniform.
        """
        return {n: self.reduce(list(v))[0] for n, v in per_param.items()}

    def __str__(self) -> str:
        return f"{self.name}[{self.width}]"


class Mesh:
    """Named axes over a set of chips in one box.

    ``Mesh({"dp": [0, 1]})`` is the whole of it. Names are yours; ``"dp"`` is the only one the
    optimizer knows about and it learns it by being handed the axis, not by looking the name
    up, so nothing here reserves a vocabulary.
    """

    def __init__(self, axes: Dict[str, Sequence[int]], *, hosts: int = 1):
        if not axes:
            raise ValueError("a mesh needs at least one axis, e.g. Mesh({'dp': [0]})")
        if hosts != 1:
            raise NotImplementedError(
                "multi-host is out of scope: the blocker is 20 MB/s over WiFi to qb2, which "
                "needs a cable and a wired NIC rather than a code change. The interface does "
                "not change shape when that lands")
        seen: Dict[int, str] = {}
        self._axes: Dict[str, Axis] = {}
        for name, ids in axes.items():
            ids = tuple(int(i) for i in ids)
            if not ids:
                raise ValueError(f"axis {name!r} has no chips")
            if len(set(ids)) != len(ids):
                raise ValueError(f"axis {name!r} repeats a chip: {ids}")
            for i in ids:
                if i in seen:
                    raise ValueError(
                        f"chip {i} is on both axis {seen[i]!r} and axis {name!r}. Axes "
                        f"partition the chips; a chip on two axes would be reduced twice")
                seen[i] = name
            self._axes[name] = Axis(name=name, device_ids=ids, mesh=self)
        if len(seen) > MAX_CHIPS_ONE_BOX:
            raise ValueError(f"{len(seen)} chips is beyond one box ({MAX_CHIPS_ONE_BOX}); "
                             f"multi-host is out of scope, see the module docstring")
        self.hosts = hosts

    @classmethod
    def auto(cls, *, axis: str = "dp") -> "Mesh":
        """Every visible chip on one axis. Reports ``hosts=1``, because that is what we have.

        Convenience, not policy: it does not decide the global batch and it cannot, because
        global batch is the user's to pin.
        """
        import ttnn
        n = min(int(ttnn.GetNumAvailableDevices()), MAX_CHIPS_ONE_BOX)
        return cls({axis: list(range(max(n, 1)))})

    def axis(self, name: str) -> Axis:
        try:
            return self._axes[name]
        except KeyError:
            raise KeyError(f"no axis {name!r}; this mesh has "
                           f"{sorted(self._axes)}") from None

    @property
    def device_ids(self) -> tuple:
        return tuple(sorted(i for a in self._axes.values() for i in a.device_ids))

    @property
    def chips(self) -> int:
        return len(self.device_ids)

    def __str__(self) -> str:
        return f"Mesh({', '.join(str(a) for a in self._axes.values())}, hosts={self.hosts})"

    __repr__ = __str__
