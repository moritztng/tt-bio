"""ESM C forward on a GPU, the tt-bio.com embed protocol: one 512-residue sequence copied to fill the batch, bf16 autocast,
the model's own forward (no masked-LM logits read), synced, cold rep discarded. The mode comes from ESMC_OPT (unset = stock);
under exact the kit prints its APPLIED line, which this script echoes into the record.
usage: python esmc_gpu.py --variant 300m|600m|6b --batch 7 --warm 15 --seq prot512.seq --out FILE.jsonl
"""
import argparse, hashlib, json, os, time

import torch
from esm.models.esmc import ESMC

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
ap.add_argument("--batch", type=int, default=7)
ap.add_argument("--warm", type=int, default=15)
ap.add_argument("--seq", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
raw = open(a.seq, "rb").read()
seq = "".join(raw.decode().split()[-1:]) if raw.startswith(b">") else raw.decode().strip()
client = ESMC.from_pretrained(f"esmc_{a.variant}", device=torch.device("cuda"))
batch = client.tokenizer([seq] * a.batch, return_tensors="pt", padding=True).to("cuda")
times, digest = [], None
for rep in range(a.warm + 1):
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = client.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    emb = out.last_hidden_state  # client.model is the native EsmcForMaskedLM; ESMCOutput.embeddings is this tensor
    digest = hashlib.sha256(emb.float().mean(1).cpu().numpy().tobytes()).hexdigest()[:16]
warm = sorted(times[1:])
rec = dict(variant=a.variant, mode=os.environ.get("ESMC_OPT") or "off", batch=a.batch, tokens=int(batch["input_ids"].shape[1]),
           seq_sha256=hashlib.sha256(raw).hexdigest()[:16], cold_s=times[0], warm_median_s=warm[len(warm) // 2],
           warm_min_s=warm[0], warm_max_s=warm[-1], n=len(warm), pooled_digest=digest, gpu=torch.cuda.get_device_name())
open(a.out, "a").write(json.dumps(rec) + "\n")
print(json.dumps(rec), flush=True)
