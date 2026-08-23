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
import re

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


# ── the gate interpreter itself ────────────────────────────────────────────────
#
# A gate scores the checkout's code on whatever interpreter it was launched with. Nothing
# used to check that interpreter satisfies tt-bio's OWN declared dependencies, and on
# 2026-08-23 that cost a full UX gate run: qb2's shared env predated the RF3 dependency
# additions, so `--model rf3` died in featurization on `ModuleNotFoundError: toolz` — a
# package declared in pyproject.toml since 2026-08-22. The gate reported `rf3 FAIL`, which
# reads exactly like a product regression and blocks a tag. Same env was 9 requirements off
# in total, two of them version bounds (transformers, huggingface_hub).
#
# Version bounds are checked, not just presence: RELEASING.md has always opened with "the
# gate interpreter must carry the pinned ttnn" because 0.67.4 and 0.68.0 disagree by several
# angstrom on a boltz2 no-MSA target. That was a line in a document a human had to remember.
# It is a machine check now.

_DEP_ARRAYS = ("dependencies", "optional-dependencies")


def _pyproject_requirements(path, extras=()):
    """The requirement strings tt-bio declares: base dependencies plus the named extras.

    Uses ``tomllib`` where it exists. The gate also runs on 3.10 interpreters, which have no
    ``tomllib`` and where adding a TOML dependency to read the file that declares the
    dependencies would be circular, so the fallback scans the two arrays directly.
    ``test_both_pyproject_readers_agree`` pins the two against each other on the repo's own
    file, so the fallback cannot drift away from the real parser unnoticed.
    """
    text = open(path, encoding="utf-8").read()
    try:
        import tomllib
    except ImportError:
        return _scan_requirements(text, extras)
    proj = tomllib.loads(text)["project"]
    reqs = list(proj.get("dependencies") or ())
    optional = proj.get("optional-dependencies") or {}
    for extra in extras:
        reqs += list(optional.get(extra) or ())
    return reqs


def _strip_comment(line):
    """The line up to its first ``#`` outside a double-quoted string."""
    quoted = False
    for i, ch in enumerate(line):
        if ch == '"':
            quoted = not quoted
        elif ch == "#" and not quoted:
            return line[:i]
    return line


def _scan_requirements(text, extras=()):
    """``tomllib``-free reader for the requirement arrays.

    Line-based rather than a regex over the whole file: the arrays are long, commented, and
    interleaved with other tables, and a regex that spans them is the kind of thing that
    quietly returns one element instead of forty-four. Comments are cut first: tt-bio's own
    dependency array carries the sentence `zstandard is a different distribution from the
    "zstd" above`, and reading quoted strings without cutting comments turns that into a
    forty-fifth requirement.
    """
    wanted = {("project", "dependencies")}
    wanted |= {("project.optional-dependencies", e) for e in extras}
    out, table, collecting = [], None, False
    for raw in text.splitlines():
        line = _strip_comment(raw)
        stripped = line.strip()
        if not collecting and stripped.startswith("[") and stripped.endswith("]") \
                and "=" not in stripped:
            table = stripped[1:-1]
            continue
        if collecting:
            out += re.findall(r'"([^"]+)"', line)
            if stripped.startswith("]"):
                collecting = False
            continue
        key, sep, rest = stripped.partition("=")
        if not sep or (table, key.strip()) not in wanted:
            continue
        out += re.findall(r'"([^"]+)"', rest)
        collecting = "]" not in rest
    return out


def declared_dependency_problems(pyproject, extras=("tenstorrent",), env=None):
    """Reasons this interpreter cannot be trusted to score the checkout, or ``[]``.

    ``env`` is an override map of ``{dist_name: version_or_None}`` for tests; ``None`` means
    read the running interpreter.
    """
    try:
        from packaging.requirements import Requirement
    except ImportError:
        return ["`packaging` is not installed in this interpreter, so tt-bio's declared "
                "dependencies cannot be checked — and `packaging` is itself one of them."]
    try:
        reqs = _pyproject_requirements(pyproject, extras)
    except (OSError, KeyError) as exc:
        return [f"cannot read declared dependencies from {pyproject}: {exc}"]
    if not reqs:
        return [f"{pyproject} declared no dependencies, which cannot be right — the reader "
                f"is broken, so treat this interpreter as unverified rather than clean."]

    def installed(name):
        if env is not None:
            return env.get(name, False)
        from importlib.metadata import distribution, PackageNotFoundError
        try:
            return distribution(name).version
        except PackageNotFoundError:
            return False

    missing, wrong = [], []
    for spec in reqs:
        req = Requirement(spec)
        version = installed(req.name)
        if version is False:
            missing.append(req.name)
        elif version and req.specifier and not req.specifier.contains(
                version, prereleases=True):
            wrong.append(f"{req.name} {version} violates the declared {req.specifier}")
    problems = []
    if missing:
        problems.append(
            f"this interpreter is missing {len(missing)} of tt-bio's {len(reqs)} declared "
            f"runtime dependencies: {', '.join(sorted(missing))}. A gate cannot score a "
            f"package on an interpreter that could not install it — the legs that reach a "
            f"missing import report FAIL, which is indistinguishable from a real regression.")
    if wrong:
        problems.append(
            "declared version bounds this interpreter violates: " + "; ".join(sorted(wrong))
            + ". These pins exist because the versions differ in results, so a leg measured "
              "outside them did not measure what a user gets.")
    return problems
