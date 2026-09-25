import atexit, json, os, sys

def _dump():
    T = sys.modules.get("tt_bio.tenstorrent")
    if T is None:
        return
    d = os.environ.get("TT_HIFI_DUMP")
    if not d:
        return
    try:
        rec = {"pid": os.getpid(),
               "flag": os.environ.get("TT_BIO_TRIATT_DIVIDING_K"),
               "stats": dict(getattr(T, "TRIATT_FUSED_HIFI_STATS", {})),
               "picks": {str(k): v for k, v in
                         getattr(T, "TRIATT_FUSED_HIFI_PICKS", {}).items()}}
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"hifi_{os.getpid()}.json"), "w") as fh:
            json.dump(rec, fh, indent=1)
    except Exception:
        pass

atexit.register(_dump)
