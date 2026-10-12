"""Record which files of the tree under test a gate leg's Python processes read.

scripts/gate_host_prep.sh installs this file into every gate venv as `_gate_reads` with a .pth
line that imports it, so every interpreter of the venv loads it whatever PYTHONPATH a gate script
hands its children. It does nothing unless GATE_READS (a directory) and GATE_TREE (the tree) are
set, which gate_fanout.py does per leg. Each Python process then appends each tree file it opens
for reading, imports (a .pyc maps back to its source) or lists, once, to GATE_READS/<pid>.txt.
The ledger keeps that set with the leg's PASS, and a later commit reuses the PASS when none of
those files changed. Lines are written as they are seen, not at exit: a multiprocessing child
leaves through os._exit and would lose an exit hook's write.

Not seen here: files read by something other than Python (the JIT compiling a C++ kernel, a shell
script). gate_fanout keys every tracked non-Python file instead, so a change to one invalidates
every leg.
"""
import os
import sys

_out, _tree = os.environ.get("GATE_READS"), os.environ.get("GATE_TREE")
if _out and _tree:
    from importlib.util import source_from_cache

    _tree = os.path.realpath(_tree) + os.sep
    _site = os.sep + "site-packages" + os.sep
    _seen, _busy, _fd = set(), [False], {}         # _fd: pid -> this process's file

    def _rel(path):
        p = os.path.abspath(os.fsdecode(path))
        if p.startswith(_tree):
            r = p[len(_tree):]
        elif _site in p and p.split(_site, 1)[1].startswith("tt_bio" + os.sep):
            r = p.split(_site, 1)[1]          # the leg imports tt_bio from the commit's wheel
        else:
            return None
        return source_from_cache(r) if r.endswith(".pyc") else r

    def _note(entry):
        pid = os.getpid()
        if pid not in _fd:                    # first read, or a forked child: its own file
            os.makedirs(_out, exist_ok=True)
            _fd[pid] = os.open(os.path.join(_out, f"{pid}.txt"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            _seen.clear()
        if entry not in _seen:
            _seen.add(entry)
            os.write(_fd[pid], (entry + "\n").encode())

    def _hook(event, args):
        if _busy[0] or event not in ("open", "os.listdir", "os.scandir"):
            return
        _busy[0] = True
        try:
            path = args[0] if args and args[0] is not None else "."
            if isinstance(path, int) or (r := _rel(path)) is None:
                return
            if event == "open":
                mode, flags = (list(args[1:3]) + [None, 0])[:2]
                if (("r" in mode and "+" not in mode) if isinstance(mode, str)
                        else (flags or 0) & os.O_ACCMODE == os.O_RDONLY):
                    _note(r)                      # what it writes is output, not input
            # The import system lists every package directory to find modules; that names
            # nothing the leg's result depends on. Code that lists a directory itself does.
            elif not sys._getframe(1).f_code.co_filename.startswith("<frozen importlib"):
                _note("dir:" + r.rstrip(os.sep))
        except Exception:
            pass
        finally:
            _busy[0] = False

    sys.addaudithook(_hook)
