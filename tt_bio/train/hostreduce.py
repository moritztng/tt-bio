"""Cross-process all-reduce for data parallelism, over a shared directory.

One process per chip is forced rather than chosen: a ttnn process that can see four chips
brings up all four, so each replica opens exactly one card and the replicas are separate
processes. That puts the gradient exchange outside any one device context, so
``ttnn.all_reduce`` is unavailable and the reduction happens on the host.

**Which costs nothing, because the gradient is already on the host.** ABodyBuilder3's step
pulls every gradient down to a float32 mirror before the optimizer runs
(``tt_bio/train/abodybuilder3_step.py``), so a host-side reduce adds no transfer that the step
was not already paying. 7,111,515 parameters at float32 is 28.4 MB per rank per step against a
38.5 s step, and the default directory is ``/dev/shm``, so the exchange is a memcpy.

**The sum is taken in rank order on every rank, and that is the load-bearing detail.** Floating
point addition is not associative, so summing the same four arrays in two different orders gives
two different results, and two ranks that disagree in the last mantissa bit diverge from the next
step onward while both loss curves look healthy. Every rank sums ``0, 1, ..., world-1``, so every
rank computes the same bits and the per-rank master hashes stay equal. That equality is asserted
each step by the caller rather than assumed here.

**Two hosts run the same protocol over a longer wire.** A rendezvous of
``/dev/shm/abb3-dp+ttuser@tt-quietbox2`` keeps the local directory and mirrors every file that
lands in it to that peer, so each host has its own copy of the rendezvous and every rank still
sums 0..world-1 over local files. ``tt_bio/train/xhost.py`` is the transport and it is the only
thing that changes; the reduction order, the retirement and the hash check are the same code on
one host or two. Without a peer in the rendezvous nothing here is reached, so the single-host
path is unchanged.

Failure is a timeout, not a hang: a rank killed by a watchdog reset stops writing, the survivors
raise after ``timeout`` seconds, and the supervisor restarts the whole world from the last
checkpoint. A rendezvous that waits forever converts one dead rank into a silently stalled run.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .xhost import SshPeers, parse_rendezvous

__all__ = ["HostReduce", "master_hash"]


def master_hash(arrays) -> bytes:
    """A digest of the float32 masters, in parameter order. The DP equality invariant.

    Hashes the raw bytes rather than a rounded summary: the failure this catches -- one rank
    stepping on its own gradient, or a resume that restored three ranks and re-initialised the
    fourth -- shows up in the low mantissa bits first and in the loss curve much later, or never.
    """
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        a = np.ascontiguousarray(a, dtype=np.float32)
        h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
        h.update(a.tobytes())
    return h.digest()


class HostReduce:
    """A rendezvous over ``dir`` for ``world`` ranks. ``world == 1`` makes every call a no-op.

    The single-chip case is not a special branch in the caller: ``allreduce`` returns its input
    and ``check_equal`` passes, so the same run code is what gets measured at 1, 2 and 4 chips.
    """

    def __init__(self, dir, rank: int, world: int, *, timeout: float = 900.0,
                 poll: float = 0.01, transport=None):
        if not (0 <= rank < world):
            raise ValueError(f"rank {rank} is outside a world of {world}")
        self.rank, self.world = int(rank), int(world)
        self.dir, self.peers = parse_rendezvous(dir)
        self.timeout, self.poll = float(timeout), float(poll)
        self.waited = 0.0
        self.bytes_moved = 0
        self.transport = transport
        if self.world > 1:
            self.dir.mkdir(parents=True, exist_ok=True)
            if self.peers and self.transport is None:
                self.transport = SshPeers(self.peers, self.dir)

    # ------------------------------------------------------------------ the primitive

    def allgather(self, tag: str, payload) -> list:
        """Every rank's ``payload`` for ``tag``, in rank order. ``bytes`` or a float32 array.

        Written to a temporary name and renamed, so a reader never sees a partial file: rename
        within a directory is atomic and a reader that polls for existence would otherwise load
        a half-written array as a valid one.
        """
        if self.world == 1:
            return [payload]
        raw = isinstance(payload, (bytes, bytearray))
        d = self.dir / tag
        d.mkdir(parents=True, exist_ok=True)
        ext = "bin" if raw else "npy"
        final = d / f"{self.rank}.{ext}"
        # The temporary name ends in the same extension on purpose: `np.save` appends `.npy`
        # to anything that does not already end in it, so a `.tmp` suffix would leave the
        # written file somewhere other than where the rename looks for it.
        tmp = d / f".{self.rank}.tmp.{ext}"
        if raw:
            tmp.write_bytes(payload)
        else:
            np.save(tmp, np.ascontiguousarray(payload, dtype=np.float32))
        os.replace(tmp, final)
        if self.transport is not None:
            # Read back rather than kept in hand, so the bytes on the wire are the bytes on
            # disk and the local write path is the one it always was.
            self.transport.put(f"{tag}/{final.name}", final.read_bytes())
        want = [d / f"{r}.{ext}" for r in range(self.world)]
        t0 = time.perf_counter()
        while True:
            missing = [p for p in want if not p.exists()]
            if not missing:
                break
            if self.transport is not None:
                self.transport.check()
            if time.perf_counter() - t0 > self.timeout:
                raise TimeoutError(
                    f"rank {self.rank} waited {self.timeout:.0f}s at {tag!r} for "
                    f"{[p.name for p in missing]}. A rank is gone -- on this host that is a "
                    f"watchdog reset, and the supervisor's job is to restart the whole world "
                    f"from the last checkpoint rather than let the survivors stall")
            time.sleep(self.poll)
        self.waited += time.perf_counter() - t0
        out = [p.read_bytes() if raw else np.load(p) for p in want]
        self.bytes_moved += sum(p.stat().st_size for p in want)
        return out

    # ------------------------------------------------------------------ what the run uses

    def allreduce(self, vec: np.ndarray, *, step: int) -> np.ndarray:
        """Sum ``vec`` across the world. Sum, not mean: the divisor is the pinned global batch.

        Dividing by the chip count here is the substitution that turns one recipe into a
        different recipe per box, which is exactly what ``tt_bio/train/mesh.py`` refuses.
        """
        if self.world == 1:
            return vec
        parts = self.allgather(f"grad-{step:09d}", vec)
        total = parts[0].astype(np.float32, copy=True)
        for p in parts[1:]:
            total += p
        self._retire(f"grad-{step - 1:09d}")
        return total

    def check_equal(self, digest: bytes, *, step: int, what: str = "master weights") -> None:
        """Fail the run unless every rank's digest matches. Never a warning.

        N replicas each stepping on their own gradient produce N different models behind N loss
        curves that all fall, so there is no metric a human watching the run could notice it in.
        """
        if self.world == 1:
            return
        got = self.allgather(f"hash-{step:09d}", digest)
        self._retire(f"hash-{step - 1:09d}")
        if len(set(got)) != 1:
            raise RuntimeError(
                f"{what} diverged at step {step}: {len(set(got))} distinct digests across "
                f"{self.world} ranks ({[g.hex()[:12] for g in got]}). Every rank must apply the "
                f"same reduced gradient to the same master, so this is either an unreduced step "
                f"or a resume that restored some ranks and re-initialised others. Both train "
                f"something other than one model and both look healthy in the loss")

    def _retire(self, tag: str) -> None:
        """Delete this rank's file from the previous step, once the current one has closed.

        Safe at exactly this point and not earlier. Reaching here means every rank wrote its
        file for the current step, which means every rank finished the previous step's
        exchange, so nobody is still reading the file being removed.
        """
        for ext in ("npy", "bin"):
            (self.dir / tag / f"{self.rank}.{ext}").unlink(missing_ok=True)
            if self.transport is not None:
                # The copy on the peer is this rank’s file too, and nobody else will remove
                # it. Skipping this leaks 28.4 MB per rank per step onto every peer host.
                self.transport.remove(f"{tag}/{self.rank}.{ext}")

    def cleanup(self) -> None:
        if self.world > 1 and self.rank == 0 and self.dir.exists():
            shutil.rmtree(self.dir, ignore_errors=True)
        if self.transport is not None:
            if self.rank == 0:
                self.transport.remove_tree()
            self.transport.close()
            self.transport = None

    def __str__(self) -> str:
        where = f"{self.dir}" + (f" + {','.join(self.peers)}" if self.peers else "")
        return (f"HostReduce(rank {self.rank}/{self.world}, {where}, "
                f"{self.bytes_moved / 1e6:.0f} MB moved, {self.waited:.1f}s waiting)")
