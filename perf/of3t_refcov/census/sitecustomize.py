"""Import census for of3t-refcov, taken in EVERY process of a real fold.

`tt_bio.main predict` spawns a local worker and the model runs THERE, so a census taken in the
CLI process would answer about the wrong process. `sitecustomize` is imported by every
interpreter start, spawned children included, which is why this is not a wrapper: a wrapper
that replaces `__main__` breaks multiprocessing spawn outright (it did, first attempt).
"""
import atexit, json, os, sys


def _dump():
    d = os.environ.get("OF3T_CENSUS_DIR")
    if not d:
        return
    mods = sorted(m for m in sys.modules if m.startswith("tt_bio"))
    if not mods:
        return
    try:
        with open(os.path.join(d, "census_%d.json" % os.getpid()), "w") as f:
            json.dump({"pid": os.getpid(), "argv": sys.argv[:4], "n_tt_bio_modules": len(mods),
                       "tt_bio_train_imported": [m for m in mods
                                                 if m.startswith("tt_bio.train")],
                       "modules": mods}, f, indent=1)
    except OSError:
        pass


atexit.register(_dump)
