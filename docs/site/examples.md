# Examples

End-to-end, copy-pasteable examples in **curl** and **Python**, for the three
common tasks: a plain fold, a co-fold with affinity, and a binder design. All use
the async submit → poll → download flow.

```
BASE = https://api.japanfold.com
```

---

## Fold a protein

### curl

```bash
BASE=https://api.japanfold.com

JOB=$(curl -s -X POST $BASE/v1/predictions \
  -H 'Content-Type: application/json' \
  -d '{"model":"boltz2","name":"myprotein","sequence":"MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')

# poll (Prefer: wait blocks up to 60s per call)
until curl -s -H 'Prefer: wait=60' $BASE/v1/jobs/$JOB \
  | grep -qE '"status":"(succeeded|failed|canceled)"'; do :; done

curl -s $BASE/v1/jobs/$JOB/results
curl -s $BASE/v1/jobs/$JOB/archive -o myprotein.zip && unzip -oq myprotein.zip -d myprotein
```

### Python — stdlib only (no dependencies)

```python
import json, time, urllib.request

BASE = "https://api.japanfold.com"
# The edge blocks urllib's default User-Agent as a bot — send a browser-like one.
HEADERS = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}

def api(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=HEADERS)
    with urllib.request.urlopen(req) as r:
        return json.load(r)

def wait(job_id):
    while True:
        job = api("GET", f"/v1/jobs/{job_id}")
        if job["status"] in ("succeeded", "failed", "canceled"):
            return job
        time.sleep(5)

job = api("POST", "/v1/predictions",
          {"model": "boltz2", "name": "myprotein",
           "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"})
job = wait(job["id"])
assert job["status"] == "succeeded", job.get("error")

results = api("GET", f"/v1/jobs/{job['id']}/results")
for row in results["rows"]:
    print(row["id"], "plddt=", row.get("plddt"), "ptm=", row.get("ptm"))

# download the zip bundle (urlretrieve doesn't take headers, so open it directly)
req = urllib.request.Request(BASE + results["archive_url"], headers=HEADERS)
with urllib.request.urlopen(req) as r, open("myprotein.zip", "wb") as f:
    f.write(r.read())
```

### Python — httpx

```python
import time, httpx

BASE = "https://api.japanfold.com"

with httpx.Client(base_url=BASE, timeout=120) as c:
    job = c.post("/v1/predictions", json={
        "model": "boltz2", "name": "myprotein",
        "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"}).json()

    while job["status"] not in ("succeeded", "failed", "canceled"):
        time.sleep(5)
        job = c.get(f"/v1/jobs/{job['id']}").json()

    results = c.get(f"/v1/jobs/{job['id']}/results").json()
    print(results["rows"])
    with open("myprotein.zip", "wb") as f:
        f.write(c.get(results["archive_url"]).content)
```

---

## Co-fold a protein + ligand, with affinity

Only **Boltz-2** does affinity. Provide the complex as a Boltz YAML `input`
string with a `ligand` chain and a `properties: affinity` block.

### curl

```bash
BASE=https://api.japanfold.com

read -r -d '' PAYLOAD <<'JSON'
{
  "model": "boltz2",
  "name": "prot-ligand",
  "input": "sequences:\n  - protein: {id: A, sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ}\n  - ligand: {id: L, smiles: \"CC(=O)Oc1ccccc1C(=O)O\"}\nproperties:\n  - affinity: {binder: L}\n",
  "params": {"use_msa_server": true}
}
JSON

JOB=$(curl -s -X POST $BASE/v1/predictions -H 'Content-Type: application/json' \
  -d "$PAYLOAD" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')

curl -s -H 'Prefer: wait=120' $BASE/v1/jobs/$JOB          # affinity runs take longer
curl -s $BASE/v1/jobs/$JOB/results
curl -s $BASE/v1/jobs/$JOB/archive -o prot-ligand.zip
```

### Python — httpx

```python
import time, httpx

BASE = "https://api.japanfold.com"
YAML = (
    "sequences:\n"
    "  - protein: {id: A, sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ}\n"
    "  - ligand: {id: L, smiles: \"CC(=O)Oc1ccccc1C(=O)O\"}\n"
    "properties:\n"
    "  - affinity: {binder: L}\n"
)

with httpx.Client(base_url=BASE, timeout=180) as c:
    job = c.post("/v1/predictions", json={
        "model": "boltz2", "name": "prot-ligand",
        "input": YAML, "params": {"use_msa_server": True}}).json()
    while job["status"] not in ("succeeded", "failed", "canceled"):
        time.sleep(5)
        job = c.get(f"/v1/jobs/{job['id']}").json()
    print(c.get(f"/v1/jobs/{job['id']}/results").json()["rows"])
```

The result `rows` include affinity fields alongside the structure/confidence
scores.

---

## Binder design

De-novo binders with BoltzGen via `POST /v1/designs`. Poll and download like a
prediction; the results carry ranked `designs`.

### curl

```bash
BASE=https://api.japanfold.com

read -r -d '' PAYLOAD <<'JSON'
{
  "protocol": "nanobody-anything",
  "name": "my-nanobodies",
  "spec": "sequences:\n  - protein: {id: A, sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ}\n",
  "params": {"num_designs": 10, "budget": 10, "fast": true}
}
JSON

JOB=$(curl -s -X POST $BASE/v1/designs -H 'Content-Type: application/json' \
  -d "$PAYLOAD" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')

until curl -s $BASE/v1/jobs/$JOB \
  | grep -qE '"status":"(succeeded|failed|canceled)"'; do sleep 10; done

curl -s $BASE/v1/jobs/$JOB/results
curl -s $BASE/v1/jobs/$JOB/archive -o designs.zip && unzip -oq designs.zip -d designs
```

### Python — httpx

```python
import time, httpx

BASE = "https://api.japanfold.com"

with httpx.Client(base_url=BASE, timeout=300) as c:
    job = c.post("/v1/designs", json={
        "protocol": "nanobody-anything", "name": "my-nanobodies",
        "spec": "sequences:\n  - protein: {id: A, sequence: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ}\n",
        "params": {"num_designs": 10, "budget": 10, "fast": True}}).json()
    while job["status"] not in ("succeeded", "failed", "canceled"):
        time.sleep(10)
        job = c.get(f"/v1/jobs/{job['id']}").json()
    results = c.get(f"/v1/jobs/{job['id']}/results").json()
    print(results.get("designs"))
    with open("designs.zip", "wb") as f:
        f.write(c.get(results["archive_url"]).content)
```
