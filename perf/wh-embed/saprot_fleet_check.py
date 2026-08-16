#!/usr/bin/env python3
"""Defect A regression check: SaProt can reach a fleet worker.

Three failures the planning pass reproduced at b1a3fe61d, all host-side and none needing a device:

  1. `tt-bio saprot <f> --controller <url>` exits 2, "No such option '--controller'", so the
     platform's own command builder produces a line that dies at argument parse.
  2. worker.load_model dispatches on _is_esmc_model, so a saprot-* id falls through to the
     Boltz-2 branch.
  3. worker.predict_one routes only ESMC ids to the embed path.

Also checks the shard round-trip, since SaProt's payload is an (aa, 3di) pair and yaml.safe_dump
cannot represent a tuple.
"""
import base64
import json
import sys
import tempfile
from pathlib import Path

import yaml
from click.testing import CliRunner

from tt_bio import esmc, saprot
from tt_bio.main import cli
from tt_bio.worker import _is_embed_model, _is_esmc_model, _is_saprot_model

out = {}

# --- 1. the CLI option -------------------------------------------------------------------
def opts(name):
    cmd = cli.commands[name]
    return {p.name for p in cmd.params}

out["saprot_has_controller"] = "controller" in opts("saprot")
out["saprot_has_owner"] = "owner" in opts("saprot")
out["embed_has_controller"] = "controller" in opts("embed")

work = Path(tempfile.mkdtemp(prefix="saprot-check-"))
fa = work / "x.fasta"
fa.write_text(">a|protein\nMQIFVKTLTGKTITLEVEPSD\n")
r = CliRunner().invoke(cli, ["saprot", str(fa), "--controller", "http://127.0.0.1:1"])
out["saprot_controller_exit_code"] = r.exit_code
out["saprot_controller_output_tail"] = (r.output or "").strip().splitlines()[-1:] or [""]
# exit 2 == click usage error (the old "No such option"). Anything else means the option parsed
# and the command got as far as trying to reach a controller, which is the fixed behaviour.
out["parses_controller_option"] = "No such option" not in (r.output or "")

# --- 2/3. worker dispatch ----------------------------------------------------------------
out["is_saprot_model"] = {m: _is_saprot_model(m) for m in saprot.MODELS}
out["is_embed_model"] = {m: _is_embed_model(m) for m in list(saprot.MODELS) + list(esmc.MODELS)}
out["esmc_unchanged"] = {m: _is_esmc_model(m) for m in esmc.MODELS}
out["boltz2_not_embed"] = not _is_embed_model("boltz2")

# load_model must reach a saprot branch, not the Boltz-2 else. Assert by source inspection of
# the dispatch chain rather than by loading 1.3 GB of weights.
import inspect

from tt_bio.worker import _WorkerState
src = inspect.getsource(_WorkerState.load_model)
out["load_model_has_saprot_branch"] = "_is_saprot_model(model_id)" in src and "load_saprot" in src
src_embed = inspect.getsource(_WorkerState._predict_embed_one)
out["predict_embed_handles_saprot"] = "read_shard_yaml" in src_embed

# --- 4. shard round-trip ------------------------------------------------------------------
seqs = saprot.load_sequences_with_structure(str(fa), None)
out["pair_payload"] = {k: [v[0][:8], v[1][:8]] for k, v in seqs.items()}
items = list(seqs.items())
key = (lambda it: len(it[1][0]))
shards = esmc._shard_by_length(items, 1, key=key)
blob = yaml.safe_dump({k: list(v) for k, v in shards[0]})
enc = base64.b64encode(blob.encode()).decode()
shard_path = work / "shard.yaml"
shard_path.write_text(base64.b64decode(enc).decode())
back = saprot.read_shard_yaml(shard_path)
out["shard_roundtrip_equal"] = back == seqs

# a bare-string shard must still load, read as sequence-only
p2 = work / "shard2.yaml"
p2.write_text(yaml.safe_dump({"b": "MQIF"}))
out["bare_string_shard"] = saprot.read_shard_yaml(p2) == {"b": ("MQIF", "####")}

checks = [out["saprot_has_controller"], out["saprot_has_owner"], out["parses_controller_option"],
          all(out["is_saprot_model"].values()), all(out["is_embed_model"].values()),
          all(out["esmc_unchanged"].values()), out["boltz2_not_embed"],
          out["load_model_has_saprot_branch"], out["predict_embed_handles_saprot"],
          out["shard_roundtrip_equal"], out["bare_string_shard"]]
out["ALL_PASS"] = all(checks)
print(json.dumps(out, indent=2))
sys.exit(0 if out["ALL_PASS"] else 1)
