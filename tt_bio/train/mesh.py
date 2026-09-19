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

Four chips a box is measured, not a claim: ABodyBuilder3 steps in 7.647 s on four qb1 chips
against 28.189 s on one, 3.686x at 92.2 %, with one master weight hash across the four ranks.
The 7.8 % that does not scale is host torch and not the exchange, which moves 127.87 MB in
0.293 s. It is conditional on ``launcher.host_threads`` dividing the host's cores across the
ranks: at torch's own width four ranks take 64 threads on 16 cores and the step is 931 s.

Up to 4 chips a box, and more than one box. ``Mesh({"dp": [0, 1]}, hosts=["ttuser@tt-quietbox2"])``
names the other hosts, and :meth:`Mesh.rendezvous` turns that into the spec
``tt_bio/train/hostreduce.py`` already takes -- ``/dev/shm/abb3-dp+ttuser@tt-quietbox2``. The peer
list rides in an argument that was being passed anyway, so the host axis adds no second transport
and no second protocol: ``tt_bio/train/xhost.py`` mirrors the rendezvous and every rank still sums
0..world-1 over local files, which is why the per-rank masters stay bit-identical across the
boundary. Measured on ABodyBuilder3, two chips split across qb1 and qb2: 13.545 s a step against
11.154 s in one box, 1.554x on one chip's 21.043 s where one box gives 1.887x. The gap is the link
and qb2 is on WiFi.

``Mesh.auto()`` reports ``hosts=1`` and always will: it can only count the chips on the box it runs
on, so a second host is something you name rather than something we detect.

Tensor parallelism is deliberately absent. A full training replica is 14.8 % of one chip's
measured 34.23 GB, so there is no memory argument for it; it returns only if something later
forces it, and only as an explicit exception to Moritz's standing data-parallel direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Union

from .xhost import PEER_SEP

__all__ = ["Mesh", "Axis", "UnreducedGradients", "MAX_CHIPS_ONE_BOX"]


# Per box, and not a hardware limit: a QuietBox holds four chips and that is how many one
# process can be given. The host count multiplies it, because a rank on another box is another
# process on another four.
MAX_CHIPS_ONE_BOX = 4

#: Separates peer destinations inside the rendezvous spec, the way ``PEER_SEP`` separates the
#: directory from the peers. Both belong to ``tt_bio/train/xhost.py``'s format; only the first
#: is exported, so the second is named here and the round trip through
#: :func:`~tt_bio.train.xhost.parse_rendezvous` is asserted in the tests rather than assumed.
_PEER_LIST_SEP = ","


def _peers_and_count(hosts) -> tuple:
    """``hosts`` is either how many boxes, or which ones. Returns ``(peers, count)``.

    A bare count is the honest state of a mesh whose peers are not named yet -- ``auto()``
    returns one -- and it constructs. Refusing to construct is what put this capability out of
    reach while it was measured and working; it is :meth:`Mesh.rendezvous` that needs a
    destination, and that is where saying so belongs.
    """
    if isinstance(hosts, bool):
        raise TypeError("hosts is a count or a list of ssh destinations, not a bool")
    if isinstance(hosts, int):
        if hosts < 1:
            raise ValueError(f"hosts={hosts}: a mesh runs on at least one host")
        return (), hosts
    peers = tuple(str(h).strip() for h in ([hosts] if isinstance(hosts, str) else hosts))
    for peer in peers:
        # The two separators are the format, so a destination carrying one would compose a
        # spec that parses back as a different peer list -- silently, and on the far side.
        bad = [c for c in (PEER_SEP, _PEER_LIST_SEP, " ") if c in peer]
        if not peer or bad:
            raise ValueError(
                f"peer {peer!r} is not an ssh destination this can put in a rendezvous spec: "
                f"the spec is dir{PEER_SEP}peer{_PEER_LIST_SEP}peer, so a destination cannot be "
                f"empty or contain {bad or [PEER_SEP, _PEER_LIST_SEP]}")
    if len(set(peers)) != len(peers):
        raise ValueError(f"hosts repeats a peer: {peers}. Every file would be mirrored to it "
                         f"twice and the second put would race the reader on the first")
    return peers, len(peers) + 1


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
    """Named axes over a set of chips, on one host or several.

    ``Mesh({"dp": [0, 1]})`` is the whole of it. Names are yours; ``"dp"`` is the only one the
    optimizer knows about and it learns it by being handed the axis, not by looking the name
    up, so nothing here reserves a vocabulary.

    ``hosts`` is either how many boxes the axis spans or, better, which ones:
    ``Mesh({"dp": [0, 1]}, hosts=["ttuser@tt-quietbox2"])`` from qb1 names the one peer, and the
    same mesh on qb2 names qb1. Each host names the others, the way each rank's rendezvous spec
    already does. An id on an axis is a rank's logical index once more than one host is
    involved, because two boxes both offer a chip 0 and the physical chip is picked per rank by
    ``TT_VISIBLE_DEVICES`` on the process that opens it.
    """

    def __init__(self, axes: Dict[str, Sequence[int]], *,
                 hosts: Union[int, Sequence[str], str] = 1):
        if not axes:
            raise ValueError("a mesh needs at least one axis, e.g. Mesh({'dp': [0]})")
        self.peers, self.hosts = _peers_and_count(hosts)
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
        cap = MAX_CHIPS_ONE_BOX * self.hosts
        if len(seen) > cap:
            raise ValueError(
                f"{len(seen)} chips across {self.hosts} host(s) is beyond {cap}: a box holds "
                f"{MAX_CHIPS_ONE_BOX}. Name more hosts -- "
                f"hosts=['ttuser@otherbox'] -- or take fewer chips")

    def rendezvous(self, dir) -> str:
        """The rendezvous spec for THIS host: the directory, plus the peers that mirror it.

        ``/dev/shm/abb3-dp`` on one host, ``/dev/shm/abb3-dp+ttuser@tt-quietbox2`` on two. That
        string is what :class:`~tt_bio.train.hostreduce.HostReduce` takes and what
        :func:`~tt_bio.train.xhost.parse_rendezvous` reads, so the host axis reaches the
        transport through an argument that was already being passed rather than through
        anything new. One host gives back the bare path, which is the same code path the
        single-host caller had before this method existed.
        """
        if self.hosts > 1 and not self.peers:
            raise ValueError(
                f"this mesh spans {self.hosts} hosts but none of them is named, so there is "
                f"no destination to mirror the rendezvous to. Build it with the peers instead "
                f"of the count -- Mesh(axes, hosts=['ttuser@otherbox']) -- naming every host "
                f"except this one")
        return (f"{dir}{PEER_SEP}{_PEER_LIST_SEP.join(self.peers)}" if self.peers
                else str(dir))

    @classmethod
    def auto(cls, *, axis: str = "dp") -> "Mesh":
        """Every visible chip on one axis. Reports ``hosts=1``, because that is what it can see.

        Convenience, not policy: it does not decide the global batch and it cannot, because
        global batch is the user's to pin. It cannot discover a second box either -- ttnn
        counts the chips on this one -- so a multi-host mesh is one you build yourself with the
        peers named.
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
        where = (f"hosts={self.hosts} ({_PEER_LIST_SEP.join(self.peers)} + this one)"
                 if self.peers else f"hosts={self.hosts}")
        return f"Mesh({', '.join(str(a) for a in self._axes.values())}, {where})"

    __repr__ = __str__
