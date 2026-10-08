"""Prune a BindCraft 2 project folder back to its fully-complete trajectories so a resume is exact.

Why this exists. A vast.ai box stops under us with status_msg success; contract 54845453 did it four
times in five hours, never holding 45 minutes against a campaign that needs about three. Restarting
the whole 8-trajectory campaign on every stop cannot converge, so the host arm of #17 never got past
trajectory 2.

Resuming naively is wrong, and the reason is specific. `CampaignProgress.claim_trajectory`
(campaign_output.py:171) increments `trajectories` in .campaign_state.json BEFORE the work runs, so a
box that dies mid-trajectory leaves a claim no row was ever written for. Resume reads that file back,
counts the phantom claim as spent, and silently runs fewer than 8. That is the real content of "never
re-enter a project folder with resume: true".

But the campaign already carries its own recovery path: `recovered_state()` rebuilds the counters from
what is on disk -- `trajectories` from the row count of 1_Trajectories/!_Trajectories.csv, `attempted`
from the hashes in it. Delete .campaign_state.json and the phantom claim is gone. Trajectory keys are
`jax.random.fold_in(key, trajectory_number)` (campaign.py:163), derived from the index alone, so the
retried trajectory draws the identical key and the identical design hash. The resume is not an
approximation of the lost trajectory, it is the same trajectory.

One subtlety decides correctness. The trajectory's CSV row is appended as soon as the gradient design
returns, BEFORE ProteinMPNN redesign and validation -- the stages that decide acceptance. So a row on
disk does not mean the trajectory finished. A trajectory is complete only if either:
  * a per-stage gate terminated it (`terminated` non-empty) -- there is no redesign for it, or
  * it reached redesign and has rows in 2_Refolded.
Anything else was in flight and must be dropped entirely, row and directory, or it is counted as spent
with no candidate ever scored and the accepted count comes out low.
"""
import csv, glob, json, os, shutil, sys

TRAJ, REFOLD = "1_Trajectories", "2_Refolded"


def refolded_designs(proj):
    """Design names that reached redesign. Matched against the whole refold table as text: the
    column name has moved between versions, the design name has not."""
    names = set()
    for path in glob.glob(os.path.join(proj, REFOLD, "*.csv")):
        with open(path, newline="") as handle:
            blob = handle.read()
        for row in csv.DictReader(blob.splitlines()):
            for value in row.values():
                if value and value.startswith("design_"):
                    names.add(value)
        names.update(part for part in blob.replace(",", " ").split() if part.startswith("design_"))
    return names


def prune(proj):
    table = os.path.join(proj, TRAJ, "!_Trajectories.csv")
    if not os.path.exists(table):
        return 0, 0
    with open(table, newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    reached = refolded_designs(proj)
    keep = [r for r in rows if (r.get("terminated") or "").strip() or r.get("design") in reached]
    dropped = [r for r in rows if r not in keep]
    orphans = []
    for row in dropped:
        directory = os.path.join(proj, TRAJ, row.get("design", ""))
        if row.get("design") and os.path.isdir(directory):
            shutil.rmtree(directory)
        print(f"  dropped in-flight trajectory {row.get('trajectory')} {row.get('design')}"
              f" (terminated={row.get('terminated')!r}, reached redesign=False)")
    # A folder with no row at all is the same fault one step earlier. TrajectoryRecorder creates the
    # folder when the trajectory starts; the row is appended only when the gradient design returns.
    # A box that dies mid-design leaves a folder the table never mentions. Pruning by row alone never
    # saw it, so it survived every resume -- and the retry, which draws the same key and the same
    # design hash, hit TrajectoryRecorder's own 'already has trajectory output' guard and killed the
    # arm seconds after launch. Six relaunches between 19:21Z and 21:01Z did exactly that.
    banked = {r.get("design") for r in keep}
    for directory in sorted(glob.glob(os.path.join(proj, TRAJ, "design_*"))):
        if os.path.isdir(directory) and os.path.basename(directory) not in banked:
            shutil.rmtree(directory)
            orphans.append(os.path.basename(directory))
            print(f"  removed orphan folder {os.path.basename(directory)}:"
                  f" started, no row ever written")
    if dropped:
        tmp = table + ".partial"
        with open(tmp, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(keep)
        os.replace(tmp, table)
    # The phantom claim lives here; recovered_state() rebuilds it from the rows left above.
    for name in (".campaign_state.json", ".campaign_state.json.partial", ".redesigned_sequences.txt"):
        path = os.path.join(proj, name)
        if os.path.exists(path):
            os.remove(path)
    return len(keep), len(dropped), orphans


def complete_count(proj):
    kept = 0
    table = os.path.join(proj, TRAJ, "!_Trajectories.csv")
    if not os.path.exists(table):
        return 0
    reached = refolded_designs(proj)
    with open(table, newline="") as handle:
        for row in csv.DictReader(handle):
            if (row.get("terminated") or "").strip() or row.get("design") in reached:
                kept += 1
    return kept


def main():
    proj = sys.argv[1]
    # Earlier stops left their work in .dead.* folders. Promote whichever folder carries the most
    # complete trajectories; banking one per box-life is what makes the campaign converge at all.
    candidates = [p for p in [proj] + sorted(glob.glob(proj + ".dead.*")) if os.path.isdir(p)]
    if not candidates:
        print("no project folder yet; this is a fresh campaign")
        return 0
    best = max(candidates, key=complete_count)
    print(f"folders: " + ", ".join(f"{os.path.basename(p)}={complete_count(p)}" for p in candidates))
    if best != proj:
        if os.path.isdir(proj):
            shutil.move(proj, proj + ".superseded." + str(os.getpid()))
        shutil.move(best, proj)
        print(f"promoted {os.path.basename(best)} to the live folder")
    kept, dropped, orphans = prune(proj)
    print(f"resuming with {kept} complete trajectories banked, {dropped} in-flight dropped for rerun,"
          f" {len(orphans)} orphan folder(s) removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
