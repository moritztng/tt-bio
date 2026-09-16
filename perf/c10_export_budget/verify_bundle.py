"""Verify the committed CPU evidence and reviewed source without device imports."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
rows = json.loads((root / 'manifest.json').read_text())
for row in rows:
    p = root / row['path']
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if p.stat().st_size != row['bytes'] or h != row['sha256']:
        raise SystemExit('Evidence mismatch: ' + str(p))
print(f'Verified {len(rows)} CPU evidence/source files')
