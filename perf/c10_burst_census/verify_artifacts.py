"""Verify the committed manifest without importing device libraries."""
import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent
rows=json.loads((p/"manifest.json").read_text())["artifacts"]
for r in rows:
 f=p/r["path"]
 if f.stat().st_size!=r["bytes"] or hashlib.sha256(f.read_bytes()).hexdigest()!=r["sha256"]:raise SystemExit("Artifact mismatch: "+str(f))
print(f"Verified {len(rows)} committed artifacts")
