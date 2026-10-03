#!/usr/bin/env python3
"""Expand the gallery store into protocol recordings, and write the manifest.

    python3 demo/sc26/gallery/build.py            # store/ -> trajectories/*.jsonl + manifest.json
    python3 demo/sc26/gallery/build.py --check    # also replay every file through the parser

The recordings are what engine/server.py replays (PROTOCOL.md, "Replay"). They are generated, not
committed: a 200-step Boltz-2 trajectory is 3x larger as base64 JSON than in store/. Run this once
after a checkout; it takes a few seconds and the output is the same every time.
"""
import argparse
import base64
import json
import lzma
from pathlib import Path

import numpy as np

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
               rg_expected=meta["rg_final"], title=pick["name"], story=pick["story"], pdb=pick["pdb"],
               chains=[dict(id=c["id"], role=c["role"], n_res=len(c["sequence"])) for c in pick["chains"]],
               ligands=pick["ligands"], atoms=meta["atoms"], t=0.0)
    yield dict(type="stage", id=fid, chip=meta["chip"], stage="trunk", step=0, total=meta["recycling_steps"], t=0.0)
    t_diff = meta["frame_t"][0]
    yield dict(type="stage", id=fid, chip=meta["chip"], stage="diffusion", step=0, total=meta["steps"], t=t_diff)
    of = meta["steps"]
    for i, (s, t) in enumerate(zip(meta["frame_steps"], meta["frame_t"])):
        yield dict(type="frame", id=fid, chip=meta["chip"], step=s, of=of, t=t, xyz=b64(xyz[i]),
                   x0=b64(x0[i - 1]) if s >= 0 else None, R=meta["R"][i], T=meta["T"][i])
    yield dict(type="stage", id=fid, chip=meta["chip"], stage="confidence", step=0, total=1, t=meta["frame_t"][-1])
    yield dict(type="fold_done", **base, n_res=meta["n_res"], seconds=meta["seconds"], stages=meta["stages"],
               aiclk_mhz=meta["aiclk_mhz"], xyz=b64(final), plddt=meta["plddt"],
               ptm=meta["confidence"]["ptm"], iptm=meta["confidence"]["iptm"], t=meta["seconds"],
               recorded_utc=meta["recorded_utc"], host=meta["host"])


def check(path, meta):
    """Read a recording back the way the server and the browser do."""
    lines = path.read_text().splitlines()
    assert any('"type":"fold_done"' in l for l in lines), "server.Replay would skip it"
    evs = [json.loads(l) for l in lines]
    frames = [e for e in evs if e["type"] == "frame"]
    done = evs[-1]
    dec = lambda s: np.frombuffer(base64.b64decode(s), "<f4").reshape(-1, 3)
    assert len(frames) == meta["steps"] + 1 and frames[0]["step"] == -1 and frames[-1]["step"] == meta["steps"] - 1
    assert all(dec(f["xyz"]).shape == (meta["n_atoms"], 3) for f in frames)
    assert frames[-1]["xyz"] == done["xyz"], "last frame is not the scored structure"
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
            ligands=[l["ccd"] for l in pick["ligands"]], n_res=meta["n_res"], n_atoms=meta["n_atoms"],
            model="boltz2", steps=meta["steps"], frames=n_frames, msa=meta["msa"],
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
    manifest = dict(
        protocol=1, generated_by="demo/sc26/gallery/build.py",
        about="Real Boltz-2 folds recorded on qb2. Every frame is a sampler state; see README.md.",
        entries=entries)
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(entries)} entries, store {sum(e['store_bytes'] for e in entries) / 1e6:.1f} MB, "
          f"recordings {sum(e['recording_bytes'] for e in entries) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
