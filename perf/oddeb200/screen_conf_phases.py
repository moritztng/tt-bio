#!/usr/bin/env python3
"""Sub-decompose the confidence head's 1.1383 s of host torch into what the device path can
actually take and what it cannot.

screen_confidence.json measured the region at 1.6286 s warm: 0.4903 s device Pairformer,
1.1383 s host. That is NOT all addressable. `ConfidenceHead.z_base_device` (protenix.py:1410)
builds `z_trunk + s1 + s2` in fp32 ON HOST and only then uploads, exactly as the host path does,
"precision-safe -- bf16-accumulating it on device regresses the pairformer input at small N". So
that term survives the switch and must come out of the prediction before anything is built.

Phases, and their fate under confidence_device:
  zbase      s_t layer_norm + z = z_trunk + s1 + s2          STAYS on host (z_base_device)
  distembed  mask/cdist/one-hot/linear(oh)/linear(d)         MOVES to device
  upload     T(s_t), T(z) -> device                          STAYS (same bytes, once per fold)
  pf         the device Pairformer                           ALREADY device
  download   to_torch(so), to_torch(zf)                      MOVES (z never round-trips)
  heads      pae/pde layer_norm+linear, plddt einsum, post   MOVES to device

Method: replace ConfidenceHead.confidence with a line-for-line replica of the real body with
timers between phases. The replica is only trustworthy if it computes the same thing, so the run
asserts the fold still returns plDDT 0.75411 and CIF 357c67003bb738ac -- if the replica drifted,
the digest moves and the numbers are thrown away.
"""
import json, sys, time
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/opendde-beat-b200")
sys.path.insert(0, str(WT))
sys.path.insert(0, str(WT / "scripts" / "gpu_vs_tt"))

import tt_baseline as TB
import torch
import torch.nn.functional as F
import ttnn
import tt_bio.tenstorrent as T
from tt_bio.protenix import ConfidenceHead

REC = []
REF_PLDDT = 0.75411
REF_CIF = "357c67003bb738ac"


def phased_confidence(self, s_inputs, s_trunk, z_trunk, coords, feats):
    ph = {}
    dev = T.get_device()

    def mark(k, t0):
        ph[k] = round(time.perf_counter() - t0, 4)

    t = time.perf_counter()
    N = s_trunk.shape[0]
    s_t = F.layer_norm(torch.clamp(s_trunk, -512, 512), (384,)) * self._g("input_strunk_ln.weight") + self._bias("input_strunk_ln.bias")
    z = (z_trunk + F.linear(s_inputs, self._g("linear_no_bias_s1.weight")).unsqueeze(1)
         + F.linear(s_inputs, self._g("linear_no_bias_s2.weight")).unsqueeze(0))
    mark("zbase", t)

    t = time.perf_counter()
    mask = feats["distogram_rep_atom_mask"].bool()
    xr = coords.reshape(-1, 3)[mask]
    d = torch.cdist(xr, xr)
    oh = ((d.unsqueeze(-1) >= self._g("lower_bins")) & (d.unsqueeze(-1) < self._g("upper_bins"))).float()
    z = z + F.linear(oh, self._g("linear_no_bias_d.weight")) + F.linear(d.unsqueeze(-1), self._g("linear_no_bias_d_wo_onehot.weight"))
    mark("distembed", t)

    t = time.perf_counter()
    Tn = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=self.dev, dtype=ttnn.bfloat16)
    s_dev, z_dev = Tn(s_t.unsqueeze(0)), Tn(z.unsqueeze(0))
    ttnn.synchronize_device(dev)
    mark("upload", t)

    t = time.perf_counter()
    so, zo = self.pf(s_dev, z_dev)
    ttnn.synchronize_device(dev)
    mark("pf", t)

    t = time.perf_counter()
    s_single = torch.Tensor(ttnn.to_torch(so)).float().reshape(N, 384)
    zf = torch.Tensor(ttnn.to_torch(zo)).float().reshape(N, N, -1)
    mark("download", t)

    t = time.perf_counter()
    pae_logits = F.linear(F.layer_norm(zf, (zf.shape[-1],)) * self._g("pae_ln.weight") + self._bias("pae_ln.bias"),
                          self._g("linear_no_bias_pae.weight"))
    pde_logits = F.linear(F.layer_norm(zf + zf.transpose(0, 1), (zf.shape[-1],)) * self._g("pde_ln.weight") + self._bias("pde_ln.bias"),
                          self._g("linear_no_bias_pde.weight"))
    a2t = feats["atom_to_token_idx"].long(); a2ta = feats["atom_to_tokatom_idx"].long()
    a = s_single[a2t]
    aln = F.layer_norm(a, (384,)) * self._g("plddt_ln.weight") + self._bias("plddt_ln.bias")
    plddt_logits = torch.einsum("nc,ncb->nb", aln, self._g("plddt_weight")[a2ta])
    out = self._postprocess(pae_logits, pde_logits, plddt_logits, feats)
    mark("heads", t)

    ph["total"] = round(sum(ph.values()), 4)
    REC.append(ph)
    return out


ConfidenceHead.confidence = phased_confidence

FIX = WT / "perf" / "size512" / "fixtures"
one_fold, meta, state = TB.build_fold(
    "opendde", WT / ".msa_om512_512", FIX / "cdk2x2_512.yaml", FIX / "cdk2x2_512.a3m")

import hashlib
folds = []
for i in range(3):
    t, m = one_fold()
    cifs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(meta["struct_dir"]).glob("*.cif"))}
    folds.append({"fold": i, "kind": "cold" if i == 0 else "warm", "fold_s": round(t, 3),
                  "plddt": m.get("plddt"), "cif_sha256_full": cifs,
                  "phases": REC[-1] if REC else None})
    print(f"fold {i} {folds[-1]['kind']:4} {t:8.3f}s plddt {m.get('plddt')} {REC[-1]}", flush=True)

# Faithfulness gate: the replica must not have changed the computation.
bad = [f for f in folds if abs((f["plddt"] or 0) - REF_PLDDT) > 1e-9
       or not all(v.startswith(REF_CIF) for v in f["cif_sha256_full"].values())]
warm = [f for f in folds if f["kind"] == "warm"]
keys = ["zbase", "distembed", "upload", "pf", "download", "heads", "total"]
avg = {k: round(sum(f["phases"][k] for f in warm) / len(warm), 4) for k in keys}

MOVES = ["distembed", "download", "heads"]
STAYS = ["zbase", "upload"]
summary = {
    "host": "tt-quietbox2", "model": "opendde", "n_tokens": 512,
    "replica_faithful": not bad,
    "ref": {"plddt": REF_PLDDT, "cif_sha256_prefix": REF_CIF},
    "folds": folds, "warm_phase_avg_s": avg,
    "addressable_s": round(sum(avg[k] for k in MOVES), 4),
    "not_addressable_s": round(sum(avg[k] for k in STAYS), 4),
    "already_device_s": avg["pf"],
    "note": ("addressable = distembed+download+heads, the work confidence_device moves. "
             "not_addressable = zbase+upload, which z_base_device does on host in fp32 too."),
}
out = WT / "perf" / "oddeb200" / "screen_conf_phases.json"
out.write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({k: summary[k] for k in
                  ("replica_faithful", "warm_phase_avg_s", "addressable_s",
                   "not_addressable_s", "already_device_s")}, indent=1))
if bad:
    print("REPLICA DRIFTED -- phase numbers are NOT valid", file=sys.stderr)
T.cleanup()
