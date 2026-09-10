"""Log the ttnn calls a BoltzGen design makes, so the one that never returns names its own
operands and its own call site.

py-spy puts the 2100-residue hang in `ttnn.multiply_` at tt_bio/tenstorrent.py:6571, but a stack
carries no shapes and a hung process cannot be asked for its locals. This wraps the op in the
fold's own process and flushes a line BEFORE each call, so the last line in the file is the call
that hung, with the operands an upstream repro needs.

    python3 perf/bgsdpa/hang_probe.py --target-res 2100 --timeout 360

`--sync` drains the device before each logged call, which is what turns "where the host ran out
of run-ahead" into "the op the device is wedged in". With `--sync` alone the wedge landed between
two `multiply_` calls, i.e. in an op this probe was not watching, so `--ops` widens the wrapped
set to every callable ttnn exposes and `--trace-from` keeps the log to the window that matters.
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

HOOK = '''import os, sys, threading, types
LOG = os.environ.get("MULTIPLY_LOG")

# Rebinding these would recurse (the wrapper syncs), or fires before/around the device the
# wrapper wants to ask about.
SKIP = {"synchronize_device", "open_device", "close_device", "manage_device",
        "CreateDevice", "CloseDevice", "GetNumAvailableDevices", "SetDefaultDevice",
        "GetDefaultDevice", "DumpDeviceProfiler", "get_memory_config"}


def _patch():
    # tt_bio.tenstorrent imports ttnn at module level and calls its ops by attribute, so
    # rebinding the names here is enough. Poll rather than use an import hook: model
    # construction takes tens of seconds, and the call that hangs is deep in the run.
    for _ in range(6000):
        m = sys.modules.get("ttnn")
        if m is not None and hasattr(m, "multiply_"):
            break
        threading.Event().wait(0.05)
    else:
        return
    fh = open(LOG, "a", buffering=1)
    n = [0]                       # multiply_ calls only, so indices stay comparable across passes
    seq = [0]                     # every wrapped call
    sync_all = os.environ.get("MULTIPLY_SYNC")
    trace_from = int(os.environ.get("TRACE_FROM") or 0)
    sync_from = int(os.environ.get("TRACE_SYNC_FROM") or 0)
    frames = int(os.environ.get("TRACE_FRAMES") or 0)
    want = (os.environ.get("TRACE_OPS") or "").split(",")
    want = [s for s in want if s]
    sync_dev = m.synchronize_device

    def desc(t):
        try:
            return f"{list(t.shape)}/{t.dtype}/{t.layout}/{t.memory_config().buffer_type}"
        except Exception as e:                                        # noqa: BLE001
            return f"<{type(t).__name__}:{e}>"

    def operands(a):
        out = []
        for x in a:
            if hasattr(x, "shape"):
                out.append(desc(x))
            elif isinstance(x, (list, tuple)):
                out.extend(desc(y) for y in x if hasattr(y, "shape"))
        return " ".join(out)

    def site():
        # frame 0 is site(), 1 is the wrapper, so the model's own call site is 2.
        out, f = [], sys._getframe(2)
        while f is not None and len(out) < frames:
            out.append(f"{f.f_code.co_filename.rsplit('/', 1)[-1]}:{f.f_lineno}:{f.f_code.co_name}")
            f = f.f_back
        return (" <- " + " ".join(out)) if out else ""

    def device_of(a):
        for x in a:
            try:
                return x.device()
            except Exception:                                         # noqa: BLE001
                continue
        return None

    def make(name, orig, is_mul):
        def wrapper(*a, **kw):
            if is_mul:
                n[0] += 1
            seq[0] += 1
            if n[0] < trace_from:
                return orig(*a, **kw)
            i = seq[0]
            # A ttnn op is an enqueue, so without a drain the host runs ahead of a wedged
            # device and the call that blocks is wherever the run-ahead ran out, not the call
            # the device is stuck in. Draining here makes "drained" the last line when the
            # wedge is upstream of this call, and "ENTER" the last line when this call is
            # itself the wedge.
            if sync_all or (sync_from and n[0] >= sync_from):
                d = device_of(a)
                if d is not None:
                    try:
                        sync_dev(d)
                    except Exception as e:                            # noqa: BLE001
                        fh.write(f"{n[0]} #{i} {name} sync-failed {e}\\n")
                fh.write(f"{n[0]} #{i} {name} drained\\n")
            fh.write(f"{n[0]} #{i} {name} ENTER {operands(a)}{site()}\\n")
            r = orig(*a, **kw)
            fh.write(f"{n[0]} #{i} {name} exit\\n")
            return r
        return wrapper

    def targets(mod, prefix):
        for k in dir(mod):
            if k.startswith("_") or k in SKIP:
                continue
            try:
                v = getattr(mod, k)
            except Exception:                                         # noqa: BLE001
                continue
            if callable(v) and not isinstance(v, (type, types.ModuleType)):
                yield mod, k, v, prefix + k

    wrapped = 0
    if want == ["all"] or want:
        mods = [(m, "")]
        exp = getattr(m, "experimental", None)
        if exp is not None:
            mods.append((exp, "experimental."))
        for mod, k, v, label in [t for mo, pre in mods for t in targets(mo, pre)]:
            if want != ["all"] and label not in want and k not in want:
                continue
            setattr(mod, k, make(label, v, k == "multiply_"))
            wrapped += 1
    if not hasattr(m.multiply_, "__wrapped_by_probe__"):
        # multiply_ is always watched, so the index in this log lines up with earlier passes.
        if want != ["all"] and "multiply_" not in want:
            m.multiply_ = make("multiply_", m.multiply_, True)
            wrapped += 1
    fh.write(f"# wrapped {wrapped} ttnn callables in pid {os.getpid()}\\n")


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
    ap.add_argument("--sync", action="store_true",
                    help="drain the device queue before each wrapped call, so the stall "
                         "names the op the device is stuck in and not the host run-ahead")
    ap.add_argument("--ops", default="",
                    help="also wrap these ttnn callables (comma-separated), or 'all'. "
                         "multiply_ is always wrapped so its index stays comparable.")
    ap.add_argument("--trace-from", type=int, default=0,
                    help="log nothing until this many multiply_ calls have happened; keeps "
                         "weight load and the early fold out of an 'all' trace")
    ap.add_argument("--sync-from", type=int, default=0,
                    help="like --sync but only from this multiply_ index on")
    ap.add_argument("--frames", type=int, default=0,
                    help="record this many caller frames per logged call")
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
    if args.sync:
        env["MULTIPLY_SYNC"] = "1"
    env["TRACE_OPS"] = args.ops
    env["TRACE_FROM"] = str(args.trace_from)
    env["TRACE_SYNC_FROM"] = str(args.sync_from)
    env["TRACE_FRAMES"] = str(args.frames)
    env["TT_VISIBLE_DEVICES"] = args.card
    env["TT_BIO_LEASE_CARDS"] = args.card
    env["TT_BIO_LEASE_HOLDER"] = os.environ.get("TT_BIO_LEASE_HOLDER", "worker:bh-boltzgen-sdpa-circbuf")
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
    print("\n--- last 5 lines ---")
    for l in lines[-5:]:
        print(l)
    if len(enters) > len(exits):
        print("\nHUNG CALL (entered, never returned):")
        print(enters[-1])


main()
