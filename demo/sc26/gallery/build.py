#!/usr/bin/env python3
"""Expand the gallery store into protocol recordings, and write the manifest.

    python3 demo/sc26/gallery/build.py            # store/ -> trajectories/*.jsonl, manifest.json, engine/attract.json
    python3 demo/sc26/gallery/build.py --check    # also replay every file through the parser

The recordings are what engine/server.py replays (PROTOCOL.md, "Replay"). They are generated, not
committed: a 200-step trajectory is 3x larger as base64 JSON than in store/. Run this once
after a checkout; it takes a few seconds and the output is the same every time.

It also writes engine/attract.json, the list the chips fold when no visitor is waiting: the same
picks, with the same chains, so the chips fold live what the stage replays.
"""
import argparse
import base64
import json
import lzma
import sys
from pathlib import Path

import numpy as np

from record import yaml_for

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
import trajectory  # noqa: E402

HERE = Path(__file__).resolve().parent


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, "<f4").tobytes()).decode()


def unshuffle(buf, shape):
    n = int(np.prod(shape))
    return np.frombuffer(buf, np.uint8).reshape(2, n).T.copy().view(np.int16).reshape(shape)


def load(meta):
    raw = lzma.decompress((HERE / "store" / meta["bin"]["file"]).read_bytes())
    n, fx, f0 = meta["n_atoms"], len(meta["frame_steps"]), len(meta["frame_steps"]) - 1
    bx, b0 = fx * n * 3 * 2, f0 * n * 3 * 2
    qx = unshuffle(raw[:bx], (fx, n, 3)).astype(np.float32) * np.float32(meta["quantum_xyz"])[:, None, None]
    q0 = unshuffle(raw[bx:bx + b0], (f0, n, 3)).astype(np.float32) * np.float32(meta["quantum_x0"])[:, None, None]
    final = np.frombuffer(raw[bx + b0:], "<f4").reshape(n, 3)
    qx[-1] = final  # the last state is stored exactly: the scored structure, bit for bit
    return qx, q0, final


def messages(meta, pick):
    xyz, x0, final = load(meta)
    fid = meta["id"]
    base = dict(id=fid, chip=meta["chip"], kind="replay", source="live", model=meta["model"],
                recorded_chip=meta["chip"])
    seq = ":".join(c["sequence"] for c in pick["chains"])
    yield dict(type="fold_start", **base, sequence=seq, n_res=meta["n_res"], n_atoms=meta["n_atoms"],
               steps=meta["steps"], loops=meta["recycling_steps"], seed=meta["seed"],
               rg_expected=meta["rg_final"], name=pick["name"], story=pick["story"], pdb=pick["pdb"],
               chains=[dict(id=c["id"], role=c["role"], n_res=len(c["sequence"])) for c in pick["chains"]],
               ligands=[l for l in pick["ligands"] if l["id"] in set(meta["atoms"].get("chain", [l["id"]]))],
               atoms=meta["atoms"], t=0.0)
    # Stage events: the recorded ones, each at the time the chip finished the work before it
    # (record.py). An older recording without them gets one per stage at the frame times.
    stages = meta.get("stage_events") or [["trunk", 0, meta["recycling_steps"], 0.0],
                                          ["diffusion", 0, meta["steps"], meta["frame_t"][0]],
                                          ["confidence", 0, 1, meta["frame_t"][-1]]]
    of = meta["steps"]
    timeline = [(t, 0, dict(type="stage", id=fid, chip=meta["chip"], stage=st, step=k, total=n, t=t))
                for st, k, n, t in stages]
    timeline += [(t, 1, i) for i, t in enumerate(meta["frame_t"])]
    for t, kind, ev in sorted(timeline, key=lambda e: (e[0], e[1])):
        if kind == 0:
            yield ev
            continue
        yield dict(type="frame", id=fid, chip=meta["chip"], step=meta["frame_steps"][ev], of=of, t=t)
    # every state onto the final structure, fitted on its x0 (the noise state on itself), and packed
    # as the chip worker packs a live fold
    states = [(s, t, xyz[i], x0[i - 1] if s >= 0 else None)
              for i, (s, t) in enumerate(zip(meta["frame_steps"], meta["frame_t"]))]
    yield dict(type="fold_done", **base, n_res=meta["n_res"], seconds=meta["seconds"], stages=meta["stages"],
               aiclk_mhz=meta["aiclk_mhz"], xyz=b64(final), plddt=meta["plddt"],
               ptm=meta["confidence"]["ptm"], iptm=meta["confidence"].get("iptm"), t=meta["seconds"],
               recorded_utc=meta["recorded_utc"], host=meta["host"], frames=trajectory.pack(states))


def check(path, meta):
    """Read a recording back the way the server and the browser do."""
    lines = path.read_text().splitlines()
    assert any('"type":"fold_done"' in l for l in lines), "server.Replay would skip it"
    evs = [json.loads(l) for l in lines]
    frames = [e for e in evs if e["type"] == "frame"]
    done = evs[-1]
    assert len(frames) == meta["steps"] + 1 and frames[0]["step"] == -1 and frames[-1]["step"] == meta["steps"] - 1
    states = trajectory.unpack(done["frames"], done["xyz"])
    assert done["frames"]["step"] == [f["step"] for f in frames]
    assert all(x.shape == (meta["n_atoms"], 3) for x in states)
    assert np.array_equal(states[-1], load(meta)[2]), "last state is not the scored structure"
    assert len(done["plddt"]) == max(evs[0]["atoms"]["residue"]) + 1
    ts = [f["t"] for f in frames]
    assert ts == sorted(ts)
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    picks = {p["id"]: p for p in json.load(open(HERE / "picks.json"))}
    acc_path = HERE / "accuracy.json"
    acc = json.loads(acc_path.read_text()) if acc_path.exists() else {}
    out = HERE / "trajectories"
    out.mkdir(exist_ok=True)
    entries = []
    for pid, pick in picks.items():
        mp = HERE / "store" / f"{pid}.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        path = out / f"{pid}.jsonl"
        with open(path, "w") as f:
            for m in messages(meta, pick):
                f.write(json.dumps(m, separators=(",", ":")) + "\n")
        n_frames = check(path, meta) if a.check else len(meta["frame_steps"])
        xyz = load(meta)[0]
        spread = np.sqrt(((xyz - xyz.mean(1, keepdims=True)) ** 2).sum(2).mean(1))
        steps = meta["frame_steps"]
        first = lambda k: next(s for s, v in zip(steps, spread) if v < k * spread[-1])
        store_bytes = mp.stat().st_size + (HERE / "store" / meta["bin"]["file"]).stat().st_size
        entries.append(dict(
            id=pid, name=pick["name"], story=pick["story"], why=pick["why"], pdb=pick["pdb"],
            uniprot=pick["uniprot"], chains=[dict(id=c["id"], role=c["role"], n_res=len(c["sequence"]))
                                             for c in pick["chains"]],
            ligands=[l["ccd"] for l in pick["ligands"] if l["ccd"] not in meta.get("ligands_omitted", [])],
            ligands_omitted=meta.get("ligands_omitted", []), n_res=meta["n_res"], n_atoms=meta["n_atoms"],
            model=meta["model"], steps=meta["steps"], frames=n_frames, msa=meta["msa"],
            seconds=meta["seconds"], seconds_first_fold_with_compile=meta["seconds_compile_fold"],
            stages=meta["stages"], aiclk_mhz=meta["aiclk_mhz"], chip=meta["chip"], host=meta["host"],
            recorded_utc=meta["recorded_utc"], confidence=meta["confidence"],
            mean_plddt=round(float(np.mean(meta["plddt"])), 3),
            share_plddt_70=round(float(np.mean(np.array(meta["plddt"]) >= 0.7)), 3), accuracy=acc.get(pid),
            rg_final=meta["rg_final"], visible_from_step=first(4.0), settled_at_step=first(1.1),
            spread_rms_A=[round(float(v), 1) for v in spread],
            recording=f"trajectories/{pid}.jsonl", recording_bytes=path.stat().st_size,
            store=[f"store/{pid}.json", f"store/{meta['bin']['file']}"], store_bytes=store_bytes))
        print(f"{pid:16s} {meta['n_res']:4d} res {meta['n_atoms']:5d} atoms {n_frames} frames "
              f"{meta['seconds']:6.1f} s  store {store_bytes / 1e6:5.2f} MB  jsonl {path.stat().st_size / 1e6:6.2f} MB")
    for stale in set(out.glob("*.jsonl")) - {out / f"{e['id']}.jsonl" for e in entries}:
        stale.unlink()  # the server replays every file here, so a dropped pick must go
    manifest = dict(
        protocol=2, generated_by="demo/sc26/gallery/build.py",
        about="Real OpenFold3 folds recorded on qb2. Every frame is a sampler state; see README.md.",
        entries=entries)
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    attract = [dict(name=pick["name"], story=pick["story"], pdb=pick["pdb"],
                    chains=[dict(id=c["id"], role=c["role"], n_res=len(c["sequence"])) for c in pick["chains"]],
                    sequence=":".join(c["sequence"] for c in pick["chains"]), yaml=yaml_for(pick))
               for pick in sorted((picks[e["id"]] for e in entries), key=lambda p: p["n_res"])]
    (HERE.parent / "engine" / "attract.json").write_text(json.dumps(attract, indent=1) + "\n")
    print(f"{len(entries)} entries, store {sum(e['store_bytes'] for e in entries) / 1e6:.1f} MB, "
          f"recordings {sum(e['recording_bytes'] for e in entries) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
