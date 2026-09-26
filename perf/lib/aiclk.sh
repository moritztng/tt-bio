# Read a card's AICLK from sysfs the way tt_bio.aiclk does. Source it, do not execute it:
#
#     . "${0%/perf/*}/perf/lib/aiclk.sh"     # resolves at any depth under perf/
#     aiclk 3                                 # -> "1350", or "DEAD", or "NA"
#     aiclk "$NODE"                           # a node dir or a full tt_aiclk path works too
#     aiclk_alive 3 || echo "no clock on card 3"
#
# `cat .../tt_aiclk 2>/dev/null || echo NA` catches an unreadable node and NOT a lying one.
# A Blackhole whose ARC firmware has died still enumerates, still opens and still computes;
# it just answers every telemetry read with a sentinel and raises nothing. On 2026-09-26 a
# five-minute anchor on qb1 card 3 banked a whole run of those as MHz and nothing noticed.
#
#   NA    the node would not read at all: missing, permissions, no card.
#   DEAD  the node answered with something that is not a clock.
#
# Both are non-numeric, so a consumer that already handled NA handles DEAD unchanged; DEAD
# only says which of the two it was. `aiclk` always exits 0 so it is safe under `set -e` in
# the command substitutions it replaces; `aiclk_alive` is the one that carries a status.
#
# The bound mirrors tt_bio.aiclk.MAX_PLAUSIBLE_MHZ and tests/test_aiclk_sentinel.py fails if
# the two drift. Deliberately a literal rather than a call into python: a guard that shells
# out for its own threshold fails open on the host where the thing that broke is python.
AICLK_MAX_PLAUSIBLE_MHZ=3000

aiclk() {
    _aiclk_p=${1:-0}
    case $_aiclk_p in
        *[!0-9]*) ;;
        *) _aiclk_p="/sys/class/tenstorrent/tenstorrent!$_aiclk_p" ;;
    esac
    case $_aiclk_p in
        */tt_aiclk) ;;
        *) _aiclk_p="$_aiclk_p/tt_aiclk" ;;
    esac

    if ! _aiclk_v=$(cat "$_aiclk_p" 2>/dev/null); then
        echo NA
        return 0
    fi
    case ${_aiclk_v:-x} in
        *[!0-9]*|"") echo DEAD; return 0 ;;
    esac
    if [ "$_aiclk_v" -gt 0 ] && [ "$_aiclk_v" -le "$AICLK_MAX_PLAUSIBLE_MHZ" ]; then
        echo "$_aiclk_v"
    else
        echo DEAD
    fi
    return 0
}

aiclk_alive() {
    case $(aiclk "$@") in
        NA|DEAD) return 1 ;;
    esac
    return 0
}
