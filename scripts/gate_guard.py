"""Card-grant and host-load guards shared by the two release gates.

Both gates can be launched by a fleet worker that holds exactly ONE card while a sibling
worker holds another card on the same box. Nothing in either gate used to check that: the
worker pool came from ``--workers``, the delegated legs came from whatever the ambient
environment happened to say, and a gate asked for four cards on a four-card box took all
four regardless of what its own lease covered. On 2026-08-21 qb1 ran at loadavg 30-62 with
six jobs on it and then dropped off the network entirely.

The grant is ambient ``TT_VISIBLE_DEVICES``. That is not a new convention: it is what the
fleet dispatcher exports for a worker's leased card, what ``tt_bio/device_lease.py`` keys its
per-card flock on, and what ttnn itself reads at device open. Unset means the caller was
granted the box (a release run on an idle host), which is unbounded by design and must behave
exactly as it did before this module existed.
"""

import os

GRANT_ENV = "TT_VISIBLE_DEVICES"

#: Refuse to start above this multiple of nproc. 1.5x leaves room for a gate's own fan-out on
#: a box doing ordinary work, and still refuses the 2-4x readings that preceded the qb1 outage.
DEFAULT_LOAD_CEILING = 1.5


def card_grant(env=None):
    """The physical cards this process may open, or ``None`` when it was granted the box.

    Tokens may be UMD indices or PCI BDFs (ttnn accepts either); BDFs resolve through
    ``tt_bio.runtime``, and a value that resolves to nothing is treated as unbounded rather
    than as an empty grant, so a malformed pin can never silently skip every leg.
    """
    env = os.environ if env is None else env
    raw = (env.get(GRANT_ENV) or "").strip()
    if not raw:
        return None
    try:
        from tt_bio.runtime import visible_device_indices
        cards = set(visible_device_indices(raw))
    except Exception:
        cards = {int(t) for t in raw.split(",") if t.strip().isdigit()}
    return cards or None


def grant_label(grant):
    """One-word rendering of a grant for a log line."""
    return "the whole box (unpinned)" if grant is None else f"card(s) {sorted(grant)}"


def load_ceiling_problem(multiplier=DEFAULT_LOAD_CEILING):
    """A reason to refuse to start on host load, or ``None``. Reads ``/proc/loadavg`` only.

    Even a correctly pinned gate fans subprocesses out, so N workers each within their lease
    can still overcommit the host. This is the second line of defence and costs nothing.
    """
    if multiplier <= 0:
        return None
    nproc = os.cpu_count() or 1
    load1 = os.getloadavg()[0]
    ceiling = multiplier * nproc
    if load1 <= ceiling:
        return None
    return (f"host load: 1-min loadavg {load1:.2f} is above {multiplier:g}x nproc "
            f"({nproc}) = {ceiling:.1f}. A gate measured on a box this loaded is noise, and a "
            f"gate's own fan-out on top of it is how a QuietBox stops answering. Wait for the "
            f"box to settle, move to a quieter host, or pass --load-ceiling 0 to override.")


def worker_pool_problems(local_cards, grant, host_label="this host"):
    """Local ``--workers`` cards the caller does not hold. Refusing beats narrowing silently:
    a pool trimmed behind the caller's back changes what got measured without saying so."""
    if grant is None:
        return []
    outside = sorted(c for c in dict.fromkeys(local_cards) if c not in grant)
    if not outside:
        return []
    return [f"--workers asks for {host_label} card(s) {outside}, but this process holds only "
            f"{grant_label(grant)}. Opening a card this gate was not granted overcommits a box a "
            f"sibling worker is using. Drop those slots, or run unpinned on an idle host for "
            f"full coverage."]


def leg_grant_skip(cards_required, grant):
    """Why a leg cannot run under this grant, or ``None`` if it fits."""
    if grant is None or cards_required <= len(grant):
        return None
    return (f"leg requires {cards_required} cards, granted {len(grant)} "
            f"({sorted(grant)}), not run")


def pin_target(local_cards, grant, fallback=None):
    """The card a same-process/delegated leg must be pinned to.

    Prefers the first LOCAL worker slot, so a delegated leg lands where the gate said it would
    instead of on card 0; falls back to the caller's own value, then to the grant.
    """
    for card in local_cards:
        if grant is None or card in grant:
            return card
    if fallback is not None and (grant is None or fallback in grant):
        return fallback
    if grant:
        return sorted(grant)[0]
    return fallback
