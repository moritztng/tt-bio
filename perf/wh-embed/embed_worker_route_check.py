#!/usr/bin/env python3
"""Regression check for the embed-family dispatch refactor, without opening a device.

Lever 4 moved worker.predict_one / _ensure_local_artifacts off _is_esmc_model and onto
_is_embed_model, and made _predict_embed_one choose its loader and embedder per family. Those
are three routing changes on the path JapanFold's worker takes for every embed job, and a
device run is a slow and contended way to find out they still work.

This stubs the model and the embedder, so it exercises the real predict_one dispatch, the real
_predict_embed_one body (loader choice, both writers, the metrics dict) and the real artifact
gate -- and nothing else. A device is never opened.
"""
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import yaml

from tt_bio import esmc as esmc_mod
from tt_bio import saprot as saprot_mod
from tt_bio.worker import _WorkerState, _ensure_local_artifacts

out = {}
work = Path(tempfile.mkdtemp(prefix="route-check-"))


def fake_embedding(sid, seq, d=8):
    per = np.arange(len(seq) * d, dtype=np.float32).reshape(len(seq), d)
    return esmc_mod.ESMCEmbedding(id=sid, sequence=seq, per_residue=per,
                                  pooled=per.mean(axis=0), logits=None)


def run_one(model_id, shard, out_format="npz"):
    struct_dir = work / f"out-{model_id}-{out_format}"
    struct_dir.mkdir(parents=True, exist_ok=True)
    shard_path = work / f"shard-{model_id}.yaml"
    shard_path.write_text(yaml.safe_dump(shard))

    seen = {}

    def fake_embed(model, sequences, **kw):
        seen["sequences"] = sequences
        seen["kwargs"] = kw
        return [fake_embedding(sid, v[0] if isinstance(v, tuple) else v)
                for sid, v in sequences.items()]

    old_e, old_s = esmc_mod.embed_sequences, saprot_mod.embed_sequences
    esmc_mod.embed_sequences = fake_embed
    saprot_mod.embed_sequences = fake_embed
    try:
        st = _WorkerState("cpu")
        st.model = object()
        st.pfn = lambda *a, **k: None
        cfg = {"model": model_id, "struct_dir": str(struct_dir), "job_id": "j1",
               "output_format": out_format, "batch_size": 4, "pool": "mean",
               "return_logits": False}
        metrics, _b, _f = st.predict_one(shard_path, cfg)
    finally:
        esmc_mod.embed_sequences, saprot_mod.embed_sequences = old_e, old_s

    files = sorted(p.name for p in struct_dir.glob("*"))
    return metrics, seen, files


# --- ESM-C: the path that already worked, must be unchanged --------------------------------
m, seen, files = run_one("esmc-300m", {"a": "MQIF", "b": "KTLTG"})
out["esmc_routed_to_embed"] = m is not None and "n_sequences" in m
out["esmc_n_sequences"] = m["n_sequences"]
out["esmc_ids"] = m["ids"]
out["esmc_lengths"] = m["lengths"]
out["esmc_files"] = files
out["esmc_payload_is_plain_str"] = all(isinstance(v, str) for v in seen["sequences"].values())
out["esmc_reports_device_and_write"] = "device_s" in m and "write_s" in m

# --- SaProt: the path that could not run at all before --------------------------------------
m, seen, files = run_one("saprot-650m", {"a": ["MQIF", "dvqa"], "b": "KTLTG"})
out["saprot_routed_to_embed"] = m is not None and "n_sequences" in m
out["saprot_n_sequences"] = m["n_sequences"]
out["saprot_files"] = files
out["saprot_payload_is_pair"] = all(isinstance(v, tuple) and len(v) == 2
                                    for v in seen["sequences"].values())
out["saprot_3di_preserved"] = seen["sequences"]["a"][1] == "dvqa"
out["saprot_bare_string_filled"] = seen["sequences"]["b"][1] == "#####"
out["saprot_reports_device_and_write"] = "device_s" in m and "write_s" in m

# --- parquet branch still reachable for both -------------------------------------------------
m, _s, files = run_one("esmc-300m", {"a": "MQIF"}, out_format="parquet")
out["esmc_parquet_files"] = files
m, _s, files = run_one("saprot-650m", {"a": ["MQIF", "dvqa"]}, out_format="parquet")
out["saprot_parquet_files"] = files

# --- the artifact gate must skip Boltz-2 downloads for BOTH families -------------------------
for mid in ("esmc-300m", "saprot-650m", "saprot-1.3b"):
    cfg = {"model": mid}
    _ensure_local_artifacts(cfg)
    out[f"artifacts_skipped_{mid}"] = "conf_ckpt" not in cfg

checks = [
    out["esmc_routed_to_embed"], out["esmc_n_sequences"] == 2,
    out["esmc_files"] == ["a.npz", "b.npz"], out["esmc_payload_is_plain_str"],
    out["esmc_reports_device_and_write"],
    out["saprot_routed_to_embed"], out["saprot_n_sequences"] == 2,
    out["saprot_files"] == ["a.npz", "b.npz"], out["saprot_payload_is_pair"],
    out["saprot_3di_preserved"], out["saprot_bare_string_filled"],
    out["saprot_reports_device_and_write"],
    out["esmc_parquet_files"] == ["j1.parquet"],
    out["saprot_parquet_files"] == ["j1.parquet"],
    out["artifacts_skipped_esmc-300m"], out["artifacts_skipped_saprot-650m"],
    out["artifacts_skipped_saprot-1.3b"],
]
out["ALL_PASS"] = all(checks)
print(json.dumps(out, indent=2, default=str))
sys.exit(0 if out["ALL_PASS"] else 1)
