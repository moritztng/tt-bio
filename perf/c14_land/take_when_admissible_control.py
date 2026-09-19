#!/usr/bin/env python3
"""Known-answer control for take_when_admissible.sh's queue arithmetic.

The waiter it controls exists because its predecessor exec'd into benchlock, so when the harness
preflight refused a stale window with exit 75 there was no loop left to retry. `bash -n` proves
syntax and would have passed the broken version too. What needs controlling is the decision the
broken version could not make: a non-zero rc keeps the candidate at the head, and only rc=0 pops it.

So this stubs the two things that touch hardware -- the admissibility guard and the benchlock'd
session -- and drives the real script's real loop with a scripted rc sequence 75, 75, 0, 0:

    call 1  APB       rc=75   APB must stay at the head
    call 2  APB       rc=75   APB must stay at the head
    call 3  APB       rc=0    APB pops, MM_SHORT_M_BW becomes the head
    call 4  MM_SHORT  rc=0    queue empties, exit 0

The negative control is the predecessor itself, transcribed below and run through the same
assertions: it exec's, so it must fail to reach call 2 at all. If both scripts passed, this file
would be testing nothing.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVE = HERE / "take_when_admissible.sh"

STUB_RUN = """run_one() {
  n=$(cat "$TMP/n"); n=$((n+1)); echo $n > "$TMP/n"
  echo "CALL $n $1"
  if [ $n -le 2 ]; then return 75; fi
  return 0
}
"""

# The predecessor's shape, reduced to the one line that decided its fate. It exec's, so the loop
# is gone after the first attempt and no second call can ever happen.
PRED = """#!/usr/bin/env bash
set -u
TMP=$TMP
QUEUE="TT_BIO_APB_CONCAT_HEADS:apb TT_BIO_MM_SHORT_M_BW:mmshort"
""" + STUB_RUN + """
for i in $(seq 1 4); do
  head=${QUEUE%% *}; flag=${head%%:*}; tag=${head##*:}
  exec_run() { run_one "$flag" "$tag"; }
  exec bash -c "$(declare -f run_one); TMP=$TMP; run_one $flag $tag"
done
echo "UNREACHABLE"
"""


def _stub(tmp: Path) -> Path:
    src = LIVE.read_text()
    src = re.sub(r"run_one\(\) \{.*?\n\}\n", STUB_RUN, src, flags=re.S)
    # the guard is the only other thing that would open a device or read the live box
    src = re.sub(r"^  if out=\$\(python3 \"\$G\".*$", "  if out=$(echo STUB); then",
                 src, flags=re.M)
    src = src.replace("sleep 30", "sleep 0")
    p = tmp / "stub.sh"
    p.write_text(src)
    return p


def _calls(text: str) -> list[str]:
    return re.findall(r"^CALL \d+ (\S+)$", text, flags=re.M)


def main() -> int:
    tmp = Path(subprocess.run(["mktemp", "-d"], capture_output=True, text=True,
                              check=True).stdout.strip())
    (tmp / "n").write_text("0\n")
    env = {"PATH": "/usr/bin:/bin", "TMP": str(tmp),
           "DEADLINE": str(int(subprocess.run(["date", "+%s"], capture_output=True, text=True,
                                              check=True).stdout) + 30)}
    out = subprocess.run(["bash", str(_stub(tmp))], capture_output=True, text=True,
                         env=env, timeout=120).stdout

    want = ["TT_BIO_APB_CONCAT_HEADS", "TT_BIO_APB_CONCAT_HEADS",
            "TT_BIO_APB_CONCAT_HEADS", "TT_BIO_MM_SHORT_M_BW"]
    got = _calls(out)
    ok = True
    if got != want:
        print(f"FAIL queue order: want {want}, got {got}")
        ok = False
    if "QUEUE EMPTY" not in out:
        print("FAIL: an emptied queue must exit on QUEUE EMPTY, not run to the deadline")
        ok = False
    if out.count("stays at the head") != 2:
        print(f"FAIL: want 2 refusals kept at the head, saw {out.count('stays at the head')}")
        ok = False

    # negative control: the predecessor's exec cannot reach a second call
    (tmp / "n").write_text("0\n")
    pred = tmp / "pred.sh"
    pred.write_text(PRED.replace("$TMP", str(tmp)))
    pout = subprocess.run(["bash", str(pred)], capture_output=True, text=True,
                          env=env, timeout=120).stdout
    pcalls = _calls(pout)
    if len(pcalls) != 1:
        print(f"FAIL negative control: the exec'ing predecessor should make exactly one call "
              f"and then be gone, but made {len(pcalls)}. This control is not testing anything.")
        ok = False

    print(f"live script calls : {got}")
    print(f"predecessor calls : {pcalls}  (exec replaced the loop, so it never retried)")
    print("CONTROL PASS" if ok else "CONTROL FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
