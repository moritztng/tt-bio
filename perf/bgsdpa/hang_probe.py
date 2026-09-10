"""Log every ttnn.multiply_ call a BoltzGen design makes, so the one that never returns names
its own operands.

py-spy puts the 2100-residue hang in `ttnn.multiply_` at tt_bio/tenstorrent.py:6571, but a stack
carries no shapes and a hung process cannot be asked for its locals. This wraps the op in the
fold's own process and flushes a line BEFORE each call, so the last line in the file is the call
that hung, with the operands an upstream repro needs.

    python3 perf/bgsdpa/hang_probe.py --target-res 2100 --timeout 360
"""
import argparse
import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]

HOOK = '''import os, sys, threading
LOG = os.environ.get("MULTIPLY_LOG")

def _patch():
    # tt_bio.tenstorrent imports ttnn at module level and calls ttnn.multiply_ by attribute, so
    # rebinding the name here is enough. Poll rather than use an import hook: model construction
    # takes tens of seconds, and the call that hangs is deep in the run.
    for _ in range(6000):
        m = sys.modules.get("ttnn")
        if m is not None and hasattr(m, "multiply_"):
            break
        threading.Event().wait(0.05)
    else:
        return
    orig = m.multiply_
    fh = open(LOG, "a", buffering=1)
    n = [0]

    def desc(t):
        try:
            return f"{list(t.shape)}/{t.dtype}/{t.layout}/{t.memory_config().buffer_type}"
        except Exception as e:                                        # noqa: BLE001
            return f"<{type(t).__name__}:{e}>"

    def wrapper(a, b, *args, **kw):
        n[0] += 1
        fh.write(f"{n[0]} ENTER a={desc(a)} b={desc(b)}\\n")
        r = orig(a, b, *args, **kw)
        fh.write(f"{n[0]} exit\\n")
        return r

    m.multiply_ = wrapper
    fh.write(f"# wrapped ttnn.multiply_ in pid {os.getpid()}\\n")

if LOG:
    threading.Thread(target=_patch, daemon=True).start()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-res", type=int, default=2100)
    ap.add_argument("--timeout", type=int, default=360)
    ap.add_argument("--card", default="2")
    ap.add_argument("--binder", type=int, default=80)
    ap.add_argument("--tag", default="hang")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("bh_ladder", ROOT / "perf" / "bhdesign" / "ladder.py")
    ladder = importlib.util.module_from_spec(spec)
    sys.modules["bh_ladder"] = ladder
    spec.loader.exec_module(ladder)

    work = ROOT / "perf" / "bgsdpa" / "work"
    work.mkdir(parents=True, exist_ok=True)
    fx, atoms, tres = ladder.boltzgen_fixture(
        work, args.target_res, ROOT / "perf" / "bhdesign" / "targets" / "big_7324.cif", args.binder)
    hook = work / f"hook_{args.tag}"
    hook.mkdir(exist_ok=True)
    (hook / "sitecustomize.py").write_text(HOOK)
    mlog = work / f"multiply_{args.tag}.log"
    mlog.unlink(missing_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{hook}:{ROOT}"
    env["MULTIPLY_LOG"] = str(mlog)
    env["TT_VISIBLE_DEVICES"] = args.card
    env["TT_BIO_LEASE_CARDS"] = args.card
    env["TT_BIO_LEASE_HOLDER"] = "worker:bh-boltzgen-sdpa-circbuf"
    env["TT_BIO_SIZE_LIMIT"] = "0"
    env["TT_BIO_LEASE_TIMEOUT"] = str(args.timeout)

    out_dir = work / f"out_{args.tag}"
    subprocess.run(["rm", "-rf", str(out_dir)], check=False)
    cmd = [ladder.PY, "-u", "-m", "tt_bio.main", "design", str(fx), "--model", "boltzgen",
           "--out_dir", str(out_dir), "--num_designs", "1", "--steps", "design", "--debug"]
    log = work / f"log_{args.tag}.txt"
    t0 = time.time()
    with log.open("w") as fh:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            for sig, w in ((signal.SIGINT, 20), (signal.SIGKILL, 30)):
                try:
                    os.killpg(proc.pid, sig)
                    rc = proc.wait(timeout=w)
                    break
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    rc = -9
    lines = mlog.read_text().splitlines() if mlog.exists() else []
    enters = [l for l in lines if " ENTER " in l]
    exits = [l for l in lines if l.endswith(" exit")]
    print(json.dumps({"target_res": tres, "target_atoms": atoms, "rc": rc,
                      "wall_s": round(time.time() - t0, 1), "calls_entered": len(enters),
                      "calls_returned": len(exits), "log": str(mlog)}, indent=2))
    print("\n--- last 3 entered ---")
    for l in enters[-3:]:
        print(l)
    if len(enters) > len(exits):
        print("\nHUNG CALL (entered, never returned):")
        print(enters[-1])


main()
