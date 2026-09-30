"""Dump the fused-HiFi triangle-attention counters at exit, so a CLI fold says which route it took."""
import atexit, json, os, sys


def _dump():
    T = sys.modules.get("tt_bio.tenstorrent")
    out = os.environ.get("B2P_PADUP_DUMP")
    if not out:
        return
    rec = {"pid": os.getpid(), "pad_up_env": os.environ.get("TT_BIO_TRIATT_HIFI_PAD_UP"),
           "tenstorrent_imported": T is not None}
    if T is not None:
        rec.update(pad_up_tiles=T._TRIATT_HIFI_PAD_UP_TILES,
                   stats=dict(T.TRIATT_FUSED_HIFI_STATS),
                   padded={str(k): v for k, v in T.TRIATT_FUSED_HIFI_PADDED.items()},
                   picks={str(k): v for k, v in T.TRIATT_FUSED_HIFI_PICKS.items()})
    with open(out, "w") as fh:
        json.dump(rec, fh, indent=1)


atexit.register(_dump)
