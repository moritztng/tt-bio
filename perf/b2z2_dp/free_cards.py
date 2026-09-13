#!/usr/bin/env python3
"""The first N cards nothing currently holds, comma-joined, for a ladder about to start.

Asked of the LEASE, not of lsof. ``TT_VISIBLE_DEVICES=0`` is a UMD logical id, not a device
node: on whglx logical 0 opens ``/dev/tenstorrent/16``. So a free-card scan that reads node
numbers out of ``lsof`` and hands them back as card ids is reading one namespace and answering
in another -- it called chips 0-7 free while this worker's own eight children were folding on
them. ``tt_bio.device_lease`` keys its flock on the card id the fleet uses, so a non-blocking
flock on that same file answers the question in the right namespace and agrees with what the
device open will actually enforce.
"""
import fcntl
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.device_lease import lease_dir, lease_host  # noqa: E402


def free_cards(n_cards=32):
    d = Path(lease_dir())
    d.mkdir(parents=True, exist_ok=True)
    host = lease_host()
    free = []
    for c in range(n_cards):
        if not os.path.exists(f"/dev/tenstorrent/{c}"):
            continue
        p = d / f"{host}-card{c}.json"
        # The lease file may already exist and belong to another UNIX account -- the fleet runs
        # this galaxy as tt-admin and a worker shelling in under its own login is not in that
        # group. O_RDWR then fails with EACCES on a 0664 file, and treating that as "held" made
        # every one of 32 idle cards read busy. flock needs only an open descriptor, so a
        # read-only fallback asks the real question. If even reading fails the lease is genuinely
        # opaque to us: say so rather than guessing in either direction.
        try:
            fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o664)
        except OSError:
            try:
                fd = os.open(p, os.O_RDONLY)
            except OSError as e:
                print(f"card {c}: lease unreadable ({e.strerror}), assuming held", file=sys.stderr)
                continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        free.append(str(c))
    return free


if __name__ == "__main__":
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    got = free_cards()
    if len(got) < want:
        sys.exit(f"only {len(got)} free cards ({','.join(got)}), need {want}")
    print(",".join(got[:want]))
