"""Render the 3D layer of an SC26 concept mockup from real structure files.

Not the demo's renderer (that is sc26-render's WebGL engine). This draws the
look the concept asks for, offline, so the concept can be judged on pixels:

  points   every atom a soft light, sized and dimmed by depth, with bloom
  surface  a Gaussian molecular surface (the isosurface of a sum of atom
           Gaussians), ray-marched, lit by one key light, a rim and cheap AO
  reveal   surface grows outward from the most confident core, low-confidence
           atoms never get a surface and stay points

Every frame of a trajectory is Kabsch-aligned to the final frame and the
camera is fixed from the final frame, so the noise spills past the screen.
"""
import argparse
import json
from pathlib import Path

import gemmi
import numpy as np
from PIL import Image
from scipy import ndimage

BG = np.array([0x08, 0x09, 0x0C]) / 255.0
# colour means molecule: the protein is ice, its partner is ember
ICE = np.array([0.62, 0.86, 1.00])
ICE_CORE = np.array([0.90, 0.97, 1.00])
EMBER = np.array([1.00, 0.66, 0.36])
EMBER_CORE = np.array([1.00, 0.88, 0.74])
SEAM = np.array([0.40, 0.86, 1.00])  # Tenstorrent blue, lifted for light
DIM = np.array([0.42, 0.48, 0.56])  # low confidence, never condenses
PEARL = np.array([0.93, 0.95, 0.97])
PEARL_EMBER = np.array([1.00, 0.62, 0.40])


def load(path):
    s = gemmi.read_structure(str(path))
    xyz, chain, b, bb = [], [], [], []
    for ch in s[0]:
        for r in ch:
            if r.is_water():
                continue
            for a in r:
                if a.element.name == "H":
                    continue
                xyz.append(a.pos.tolist())
                chain.append(ch.name)
                b.append(a.b_iso)
                bb.append(a.name in ("N", "CA", "C"))
    load.backbone = np.array(bb)
    return np.array(xyz), np.array(chain), np.array(b)


def kabsch(P, Q):
    """Rotation R, translation t so that P @ R.T + t best fits Q."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, qc - pc @ R.T


class Camera:
    """Fixed from the final structure: principal axis horizontal, fit to a stage box."""

    def __init__(self, final, W, H, stage, fov=26.0, yaw=0.0, pitch=0.0):
        c = final.mean(0)
        X = final - c
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        R = Vt.copy()
        if np.linalg.det(R) < 0:
            R[2] *= -1
        cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        self.R = Rx @ Ry @ R
        self.c = c
        self.W, self.H = W, H
        x0, y0, x1, y1 = stage  # fractions of the screen
        self.cx, self.cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
        self.f = H / 2 / np.tan(np.radians(fov / 2))
        Y = self.view(final)
        ext_x = np.percentile(np.abs(Y[:, 0]), 99.5) + 4
        ext_y = np.percentile(np.abs(Y[:, 1]), 99.5) + 4
        zx = self.f * ext_x / ((x1 - x0) / 2 * W)
        zy = self.f * ext_y / ((y1 - y0) / 2 * H)
        self.dist = max(zx, zy) + np.abs(Y[:, 2]).max() * 0.0

    def view(self, P):
        return (P - self.c) @ self.R.T

    def project(self, P):
        Y = self.view(P)
        z = self.dist - Y[:, 2]
        u = self.cx + self.f * Y[:, 0] / z
        v = self.cy - self.f * Y[:, 1] / z
        return u, v, z


def splat_points(cam, P, rgb, inten, size_scale=0.38, glow=1.0, gain=0.55):
    """Atoms as soft lights. Depth decides size and brightness; bloom adds the air."""
    W, H = cam.W, cam.H
    u, v, z = cam.project(P)
    ok = (z > 1) & (u > -50) & (u < W + 50) & (v > -50) & (v < H + 50) & (inten > 1e-3)
    u, v, z, rgb, inten = u[ok], v[ok], z[ok], rgb[ok], inten[ok]
    out = np.zeros((H, W, 3))
    if len(z) == 0:
        return out
    # apparent radius of a 1.0 A ball at that depth, in px
    r_px = np.clip(cam.f * 1.0 / z * size_scale, 0.6, 40)
    zn = (z - z.min()) / max(np.ptp(z), 1e-6)
    depth_gain = 1.3 - 0.95 * zn  # near is brighter, far fades into the ground
    w = (inten * depth_gain)[:, None] * rgb
    bins = np.quantile(r_px, np.linspace(0, 1, 7))
    idx = np.clip(np.searchsorted(bins, r_px, side="right") - 1, 0, 5)
    for k in range(6):
        m = idx == k
        if not m.any():
            continue
        acc = np.zeros((H, W, 3))
        x, y = u[m], v[m]
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - x0, y - y0
        for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                           (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            xi, yi = x0 + dx, y0 + dy
            inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            for c in range(3):
                np.add.at(acc[:, :, c], (yi[inb], xi[inb]), (w[m, c] * ww)[inb])
        sig = float(np.median(r_px[m])) * 0.55
        sig = max(sig, 0.5)
        # gaussian_filter keeps the sum; scale back so one isolated point peaks at `gain`
        core = ndimage.gaussian_filter(acc, (sig, sig, 0)) * (2 * np.pi * sig ** 2) * gain
        halo = ndimage.gaussian_filter(acc, (sig * 6, sig * 6, 0)) * (2 * np.pi * sig ** 2) * gain * 0.6
        out += core + glow * halo
    return out


def density_grid(P, weight, rgb, spacing=0.7, radius=2.1, pad=8.0):
    lo = P.min(0) - pad
    hi = P.max(0) + pad
    shape = np.ceil((hi - lo) / spacing).astype(int) + 1
    rho = np.zeros(shape, np.float32)
    col = np.zeros(tuple(shape) + (3,), np.float32)
    g = (P - lo) / spacing
    gi = np.round(g).astype(int)
    k = int(np.ceil(2.6 * radius / spacing))
    offs = np.arange(-k, k + 1)
    for dx in offs:
        for dy in offs:
            for dz in offs:
                q = gi + np.array([dx, dy, dz])
                ok = np.all((q >= 0) & (q < shape), axis=1) & (weight > 0)
                d2 = (((q[ok] - g[ok]) * spacing) ** 2).sum(1)
                val = weight[ok] * np.exp(-d2 / (radius ** 2) * 1.2)
                np.add.at(rho, tuple(q[ok].T), val)
                for c in range(3):
                    np.add.at(col[..., c], tuple(q[ok].T), val * rgb[ok, c])
    col /= np.maximum(rho[..., None], 1e-6)
    return rho, col, lo, spacing


def raymarch_surface(cam, P, rho, col, lo, spacing, iso=0.55, light=(-0.45, 0.6, 0.66),
                     rim_rgb=(0.35, 0.72, 0.87)):
    W, H = cam.W, cam.H
    # conservative near bound per pixel from 4 A spheres, also the hit mask
    u, v, z = cam.project(P)
    near = np.full((H, W), np.inf)
    rad = cam.f * 4.5 / z
    order = np.argsort(-z)
    for i in order:
        r = rad[i]
        x0, x1 = int(max(u[i] - r, 0)), int(min(u[i] + r + 1, W))
        y0, y1 = int(max(v[i] - r, 0)), int(min(v[i] + r + 1, H))
        if x0 >= x1 or y0 >= y1:
            continue
        near[y0:y1, x0:x1] = np.minimum(near[y0:y1, x0:x1], z[i] - 4.5)
    ys, xs = np.nonzero(np.isfinite(near))
    t = near[ys, xs] - 1.0
    # ray directions in view space (camera at z=dist looking toward -z)
    dvx = (xs - cam.cx) / cam.f
    dvy = -(ys - cam.cy) / cam.f
    dirs = np.stack([dvx, dvy, -np.ones_like(dvx)], 1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    # convert t from "z depth" to distance along the ray
    t = t / (-dirs[:, 2])
    origin_v = np.array([0, 0, cam.dist])
    Rinv = cam.R  # view = (P-c) @ R.T  ->  P = view @ R + c
    o_w = origin_v @ Rinv + cam.c
    d_w = dirs @ Rinv
    shape = np.array(rho.shape)

    def sample(grid, pts):
        g = ((pts - lo) / spacing).T
        return ndimage.map_coordinates(grid, g, order=1, mode="constant", cval=0.0)

    hit = np.zeros(len(t), bool)
    th = np.full(len(t), np.nan)
    active = np.arange(len(t))
    prev = sample(rho, o_w + d_w * t[:, None])
    step = 0.35
    for _ in range(140):
        if len(active) == 0:
            break
        tn = t[active] + step
        val = sample(rho, o_w + d_w[active] * tn[:, None])
        cross = val >= iso
        if cross.any():
            a = active[cross]
            pv, cv = prev[cross], val[cross]
            frac = (iso - pv) / np.maximum(cv - pv, 1e-6)
            th[a] = t[a] + step * np.clip(frac, 0, 1)
            hit[a] = True
        keep = ~cross
        t[active] = tn
        prev = val[keep]
        active = active[keep]
    hi = np.nonzero(hit)[0]
    p = o_w + d_w[hi] * th[hi, None]
    eps = spacing * 0.8
    grad = np.stack([sample(rho, p + e) - sample(rho, p - e) for e in np.eye(3) * eps], 1)
    n = -grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-6)
    base = np.stack([sample(col[..., c], p) for c in range(3)], 1)
    # lighting in view space
    nv = n @ cam.R.T
    L = np.array(light) / np.linalg.norm(light)
    V = -(dirs[hi])
    ndl = nv @ L
    wrap = np.clip((ndl + 0.35) / 1.35, 0, 1)
    Hh = (L + V) / np.linalg.norm(L + V, axis=1, keepdims=True)
    spec = np.clip((nv * Hh).sum(1), 0, 1) ** 48
    fres = (1 - np.clip((nv * V).sum(1), 0, 1)) ** 3
    ao1 = sample(rho, p + n * 2.0)
    ao2 = sample(rho, p + n * 5.0)
    ao = np.clip(1.0 - 0.55 * np.clip(ao1 / iso, 0, 1.6) - 0.30 * np.clip(ao2 / iso, 0, 1.5), 0.12, 1)
    shade = (0.035 + 1.08 * wrap ** 1.4) * ao
    rgb = base * shade[:, None] + 0.45 * spec[:, None] * ao[:, None] \
        + np.array(rim_rgb) * (fres * 0.85)[:, None] * (0.3 + 0.7 * ao)[:, None]
    out = np.zeros((H, W, 3))
    alpha = np.zeros((H, W))
    out[ys[hi], xs[hi]] = rgb
    alpha[ys[hi], xs[hi]] = 1.0
    # smooth the silhouette: 1-px supersample by blurring alpha edge only
    a_s = ndimage.gaussian_filter(alpha, 0.7)
    out_s = ndimage.gaussian_filter(out, (0.7, 0.7, 0)) / np.maximum(a_s[..., None], 1e-6)
    edge = (a_s > 0) & (a_s < 1)
    out[edge] = out_s[edge]
    zbuf = np.full((H, W), np.inf)
    zbuf[ys[hi], xs[hi]] = (p @ cam.R.T - cam.c @ cam.R.T)[:, 2] * -1 + cam.dist
    return out, a_s, zbuf


def tonemap(x):
    return 1.0 - np.exp(-x * 1.15)


def vignette(W, H):
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt(((xx - W * 0.42) / W) ** 2 + ((yy - H * 0.5) / H) ** 2 * 0.8)
    return np.clip(1.0 - 0.55 * r ** 2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", help="dir of fNNNN.cif, a real sampler trajectory")
    ap.add_argument("--cif", help="a single final structure")
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--reveal", type=float, default=0.0, help="0 points .. 1 full surface")
    ap.add_argument("--partner", default="", help="chain ids drawn as the partner (ember)")
    ap.add_argument("--size", default="3840x2160")
    ap.add_argument("--stage", default="0.06,0.16,0.66,0.90")
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--plddt-floor", type=float, default=70.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W, H = map(int, a.size.split("x"))
    stage = tuple(map(float, a.stage.split(",")))

    if a.frames:
        files = sorted(Path(a.frames).glob("f*.cif"))
        final, chain, b = load(files[-1])
        cur = load(files[a.frame])[0]
        R, tr = kabsch(cur, final)
        cur = cur @ R.T + tr
    else:
        final, chain, b = load(a.cif)
        cur = final
    cam = Camera(final, W, H, stage, yaw=a.yaw, pitch=a.pitch)
    partner = np.isin(chain, list(a.partner))
    conf = b >= a.plddt_floor
    pt_rgb = np.where(partner[:, None], EMBER, ICE)
    pt_rgb = np.where(conf[:, None], pt_rgb, DIM)
    sf_rgb = np.where(partner[:, None], PEARL_EMBER, PEARL)

    img = np.zeros((H, W, 3)) + 0.0
    # reveal wave: grows from the most confident core outward
    w = np.zeros(len(final))
    if a.reveal > 0:
        core = final[b >= np.percentile(b, 80)].mean(0)
        d = np.linalg.norm(final - core, axis=1)
        front = a.reveal * (d.max() + 6)
        w = np.clip((front - d) / 9.0, 0, 1) * conf
    # the backbone thread reads brighter than side chains once the cloud condenses
    thread = np.where(load.backbone, 1.0, 0.5)
    pts_int = (1.0 - w) * thread
    # the bloom seam: atoms inside the moving front light up in the live accent
    seam = (w > 0.02) & (w < 0.98)
    pt_rgb = np.where(seam[:, None], SEAM, pt_rgb)
    pts_int = np.where(seam, 3.0, pts_int)
    if w.any():
        rho, col, lo, sp = density_grid(final, w, sf_rgb)
        surf, alpha, zbuf = raymarch_surface(cam, final, rho, col, lo, sp)
    else:
        surf, alpha, zbuf = np.zeros((H, W, 3)), np.zeros((H, W)), np.full((H, W), np.inf)
    pts = splat_points(cam, cur, pt_rgb, pts_int)
    img = pts * (1 - alpha[..., None] * 0.92) + surf * alpha[..., None]
    img = tonemap(img) * vignette(W, H)[..., None] + BG * (1 - tonemap(img))
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(a.out)
    stats = dict(n_atoms=int(len(final)), frame=a.frame, reveal=a.reveal,
                 rms_spread=float(np.sqrt(((cur - cur.mean(0)) ** 2).sum(1).mean())),
                 plddt_mean=float(b.mean()))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
