#!/usr/bin/env bash
# p25_harvest.sh -- pull successful galaxy deep-N folds into the qb1 analysis arm tree.
#
# Runs ON qb1. Reads the run's fleet results.jsonl over ssh, rsyncs every (model,target)
# with an ok record at the requested rung(s) into ~/abag_xm/deepn/galaxy/<mdir>/<t>_n<N>/,
# verifies the harvested fold is complete on disk (results.json ok + one CIF per all_runs
# entry), and drops incomplete harvests (e.g. all-attempts-OOM targets whose on-disk state
# is the last failed attempt). Also refreshes galaxy/fleet_results.jsonl (line-deduped).
#
# usage: p25_harvest.sh <run_id> [rung ...]     e.g.: p25_harvest.sh p25 64
# Chunk-aware (p27+): records with chunk/chunks>1 land as <t>_n<N>_c<j> from galaxy <t>_c<j>
# dirs and partial chunk pools are reported at the end (the analysis pools measured-N).
# On qb1 run it directly; qb1 cannot reach the galaxy (no cloudflared), so the relay is:
#   pc:  bash scripts/abag_xm/p25_harvest.sh p25 64 && bash scripts/abag_xm/p25_harvest.sh p25b 64
#   pc:  rsync -az ~/abag_xm/deepn/galaxy/ qb1:abag_xm/deepn/galaxy/
set -uo pipefail
RUN=${1:?run id, e.g. p25 or p25b}
shift
RUNGS=${*:-64}
DEST=${DEST:-$HOME/abag_xm/deepn/galaxy}
GAL=${GAL:-japanfold-ssh}
GB=/home/cust-team/mthuening/$RUN
mkdir -p "$DEST"

timeout 120 ssh -o BatchMode=yes "$GAL" "cat $GB/results.jsonl" > "$DEST/.fleet.$RUN.jsonl" \
  || { echo "WARN: no results.jsonl for $RUN"; exit 0; }
# skip-and-link manifest (p28+): attests the engine tree is identical across windows,
# which licenses the nested-rung repair below (a rung-R chunk j IS rung R/2's chunk j).
timeout 60 ssh -o BatchMode=yes "$GAL" "cat $GB/link_manifest.json 2>/dev/null" \
  > "$DEST/.link_manifest.$RUN.json" || true
{ [ -f "$DEST/fleet_results.jsonl" ] && cat "$DEST/fleet_results.jsonl"; cat "$DEST/.fleet.$RUN.jsonl"; } \
  | sort -u > "$DEST/.fleet.all" && mv "$DEST/.fleet.all" "$DEST/fleet_results.jsonl"
rm -f "$DEST/.fleet.$RUN.jsonl"

python3 - "$DEST" "$RUN" $RUNGS <<'PY'
import json, pathlib, subprocess, sys
dest = pathlib.Path(sys.argv[1]); run = sys.argv[2]
rungs = {int(x) for x in sys.argv[3:]}
MD = {"boltz2": "boltz2", "opendde-abag": "opendde",
      "protenix-v2": "protenix", "esmfold2": "esmfold2"}
ok = {}
for line in (dest / "fleet_results.jsonl").read_text().splitlines():
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    if r.get("rc") == 0 and r.get("cifs", 0) > 0 and r.get("rung") in rungs:
        # chunked runs (p27+) key by (model,target,rung,chunk); unchunked get chunk None
        ok[(r["model"], r["target"], r["rung"], r.get("chunk"))] = r.get("chunks", 1)
print(f"harvest: {len(ok)} ok folds at rungs {sorted(rungs)}")
gb = f"/home/cust-team/mthuening/{run}"
chunk_tot = {}

# Upward materialization: the inverse of the link phase -- a rung-R chunk j that the
# fleet hardlinked from rung R/2's chunk j can be re-materialized LOCALLY from the
# already-harvested R/2 slot instead of re-rsyncing identical bytes through the relay.
# Same license as the downward repair below: the run's link_manifest.json attesting
# commit_equal (same engine tree => same numerics). Existing slots are re-verified
# (incomplete ones are rebuilt); never disturbs a complete slot.
mani = None
mp = dest / f".link_manifest.{run}.json"
try:
    mani = json.loads(mp.read_text()) if mp.exists() and mp.stat().st_size else None
except Exception:
    mani = None
def slot_complete(d, mdir, t):
    rjs = list(d.glob(f"{mdir}_results_{t}/results.json"))
    if not rjs:
        return False
    try:
        rec = json.loads(rjs[0].read_text())[0]
        n = len(rec.get("all_runs") or [])
        return rec.get("status") == "ok" and n > 0 \
            and len(list((rjs[0].parent / "structures").glob("*.cif"))) == n
    except Exception:
        return False
if mani and mani.get("commit_equal"):
    made = rebuilt = 0
    for (model, t, rung, chunk), chunks in sorted(ok.items()):
        if chunk is None or chunks <= 1:
            continue
        mdir = MD[model]
        r2 = rung // 2
        while r2 >= 128 and chunk < r2 // 64:
            src = dest / mdir / f"{t}_n{r2}_c{chunk}"
            slot = dest / mdir / f"{t}_n{rung}_c{chunk}"
            if not slot.exists() and slot_complete(src, mdir, t):
                subprocess.run(["cp", "-al", str(src), str(slot)], check=True)
                made += 1
            r2 //= 2
    if made:
        print(f"upward materialization (commit_equal manifest): {made} slots hardlinked locally")

for (model, t, rung, chunk), chunks in sorted(ok.items()):
    mdir = MD[model]
    suffix = f"{t}_n{rung}" + (f"_c{chunk}" if chunk is not None and chunks > 1 else "")
    srcdir = f"{t}_c{chunk}" if chunk is not None and chunks > 1 else t
    out = dest / mdir / suffix
    rd = out / f"{mdir}_results_{t}"
    rj = rd / "results.json"
    if rj.exists():
        try:
            rec = json.loads(rj.read_text())[0]
            n = len(rec.get("all_runs") or [])
            if rec.get("status") == "ok" and n > 0 \
               and len(list((rd / "structures").glob("*.cif"))) == n:
                chunk_tot[(model, t, rung, chunks)] = chunk_tot.get((model, t, rung, chunks), 0) + 1
                continue  # already harvested and complete
        except Exception:
            pass
    out.mkdir(parents=True, exist_ok=True)
    src = f"japanfold-ssh:{gb}/{mdir}/{srcdir}/{mdir}_results_{t}/"
    r = subprocess.run(["rsync", "-az", "--timeout=300", src, str(rd) + "/"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  {model} {t} n{rung} c{chunk}: rsync FAILED {r.stderr.strip()[:120]}")
        continue
    try:
        rec = json.loads(rj.read_text())[0]
        n = len(rec.get("all_runs") or [])
        n_cif = len(list((rd / "structures").glob("*.cif")))
        if rec.get("status") == "ok" and n > 0 and n_cif == n:
            chunk_tot[(model, t, rung, chunks)] = chunk_tot.get((model, t, rung, chunks), 0) + 1
            print(f"  {model} {t} n{rung} c{chunk}: ok ({n} cifs)")
        else:
            print(f"  {model} {t} n{rung} c{chunk}: INCOMPLETE on galaxy "
                  f"(status={rec.get('status')} runs={n} cifs={n_cif}) -- dropped")
            subprocess.run(["rm", "-rf", str(out)])
    except Exception as e:
        print(f"  {model} {t} n{rung} c{chunk}: verify error {e} -- dropped")
        subprocess.run(["rm", "-rf", str(out)])
partials = [(m, t, r, k, c) for (m, t, r, k), c in sorted(chunk_tot.items()) if k > 1 and c != k]
if partials:
    print("PARTIAL chunk pools (pool is honest measured-N but not the full rung):")
    for m, t, r, k, c in partials:
        print(f"  {m} {t} n{r}: {c}/{k} chunks")

# Nested-rung repair: on a seed-nested ladder a rung-R chunk j carries the same seed block
# as rung R/2's chunk j, so a fresh rung-R fold also fills a missing R/2 (and R/4, ...)
# slot. Same commit_equal manifest license as the upward materialization above;
# never overwrites an existing slot.
if mani and mani.get("commit_equal"):
    repaired = []
    for (model, t, rung, chunk), chunks in sorted(ok.items()):
        if chunk is None or chunks <= 1 or rung < 256:
            continue
        mdir = MD[model]
        srcdir = dest / mdir / f"{t}_n{rung}_c{chunk}"
        if not (srcdir / f"{mdir}_results_{t}" / "results.json").exists():
            continue  # source slot not staged locally; nothing safe to link from
        r2 = rung // 2
        while r2 >= 128 and chunk < r2 // 64:
            slot = dest / mdir / f"{t}_n{r2}_c{chunk}"
            if not slot.exists():
                subprocess.run(["cp", "-al", str(srcdir), str(slot)], check=True)
                repaired.append(f"{model} {t} n{r2} c{chunk} <- n{rung} c{chunk}")
            r2 //= 2
    if repaired:
        print(f"nested-rung repair (commit_equal manifest): {len(repaired)} slots materialized")
        for x in repaired:
            print(f"  REPAIR {x}")
PY
