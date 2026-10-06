"""A fold's sampler states, packed for the wire (PROTOCOL.md "Coordinates").

The chip worker keeps every state of a fold and packs them once, at fold_done:

* every state is rigidly superposed onto the final structure, the rotation fitted on the state's x0
  (the network's denoised estimate, which shares the state's random frame and already has the
  protein's shape), so x0 itself never has to leave the box;
* every state but the last is quantised to int16 around its own centroid, with a step of its own
  extent / 32767, so the error is at most 1/65534 of the cloud's size: under 0.003 A from the
  sampler's 100th step on, at 833 residues;
* the last state is the scored structure. It is not repeated here: it is fold_done.xyz, float32,
  bit for bit.

    python3 demo/booth/engine/trajectory.py convert gallery/trajectories/*.jsonl

rewrites protocol-1 recordings (float32 xyz and x0 in every frame) in this format, in place.
Needs numpy; the server only slices what this module packed.
"""
import base64
import json
import sys

import numpy as np

Q = 32767


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def decode_f32(s):
    return np.frombuffer(base64.b64decode(s), "<f4").reshape(-1, 3)


def superpose(mov, fit_on, ref):
    """`mov` moved by the rigid transform that best puts `fit_on` onto `ref` (Kabsch, float64)."""
    p, q = fit_on.mean(0), ref.mean(0)
    u, _, vt = np.linalg.svd((fit_on - p).T @ (ref - q))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (mov - p) @ r.T + q


def pack(states):
    """states: [(step, t, xyz, x0 or None)] in sampler order, coordinates (n_atoms, 3); the last is
    the final structure (fold_done.xyz). Returns the fold_done `frames` field: every state's step
    and time, and all states but the last, packed."""
    ref = np.asarray(states[-1][2], np.float64)
    origin, scale, q16 = [], [], []
    for _, _, x, x0 in states[:-1]:
        x = np.asarray(x, np.float64)
        a = superpose(x, np.asarray(x0, np.float64) if x0 is not None else x, ref)
        c = a.mean(0)
        s = max(float(np.abs(a - c).max()) / Q, 1e-6)
        q16.append(np.round((a - c) / s).astype("<i2"))
        origin.append([round(float(v), 4) for v in c])
        scale.append(s)
    return {"step": [s[0] for s in states], "t": [s[1] for s in states], "origin": origin, "scale": scale,
            "q16": b64(np.concatenate(q16) if q16 else np.zeros(0, "<i2"))}


def unpack(frames, xyz):
    """fold_done's `frames` and `xyz` back to float32 coordinates, one (n_atoms, 3) array per state."""
    q = np.frombuffer(base64.b64decode(frames["q16"]), "<i2").reshape(len(frames["scale"]), -1, 3)
    out = [(q[i] * frames["scale"][i] + np.asarray(frames["origin"][i])).astype("<f4") for i in range(len(q))]
    return out + [decode_f32(xyz)]


def convert(lines):
    """Protocol-1 recording lines -> this format: frames lose their coordinates, fold_done gains
    `frames`. Lines already in this format come back unchanged."""
    lines = [l for l in lines if l.strip()]
    msgs = [json.loads(l) for l in lines]
    frames = sorted((m for m in msgs if m.get("type") == "frame"), key=lambda m: m["step"])
    if not frames or "xyz" not in frames[0]:
        return lines
    states = [(m["step"], m["t"], decode_f32(m["xyz"]), decode_f32(m["x0"]) if m.get("x0") else None) for m in frames]
    out = []
    for m in msgs:
        if m.get("type") == "frame":
            m = {k: v for k, v in m.items() if k not in ("xyz", "x0", "R", "T")}
        elif m.get("type") == "fold_done":
            m = m | {"frames": pack(states)}
        out.append(json.dumps(m, separators=(",", ":")))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] != ["convert"]:
        sys.exit(__doc__)
    for path in sys.argv[2:]:
        lines = open(path).read().splitlines()
        new = convert(lines)
        if new != [l for l in lines if l.strip()]:
            open(path, "w").write("\n".join(new) + "\n")
            print(path, "converted")
