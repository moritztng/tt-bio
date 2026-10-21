"""How far each atomworks get_random_rots map is from Haar-uniform on SO(3)."""
import numpy as np, torch
torch.manual_seed(0); N = 400_000
A = torch.randn(N, 3, 3)
Q, R = torch.linalg.qr(A)
old = Q * torch.det(Q).sign()[:, None, None]
new = Q * torch.where(torch.diagonal(R, dim1=-2, dim2=-1) < 0, -1.0, 1.0)[:, None, :]
new[:, :, -1] *= torch.where(torch.det(new) < 0, -1.0, 1.0)[:, None]
def stats(M, name):
    M = M.double().numpy()
    ang = np.degrees(np.arccos(np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    # Haar: angle density (1-cos t)/pi -> mean 90 + 2/pi*... ; compare histogram to analytic
    edges = np.linspace(0, 180, 19); h, _ = np.histogram(ang, edges)
    t = np.radians(edges); cdf = (t - np.sin(t)) / np.pi; exp = np.diff(cdf) * len(ang)
    tv_angle = 0.5 * np.abs(h - exp).sum() / len(ang)
    # where does each basis vector go: sign balance of every entry (Haar: mean 0, E[m^2]=1/3)
    mean = M.mean(0); sq = (M ** 2).mean(0)
    # rotated reference direction e1 -> octant occupancy (Haar: 1/8 each)
    v = M[:, :, 0]; occ = np.bincount(((v > 0) * [4, 2, 1]).sum(1), minlength=8) / len(v)
    print(f"{name}: TV(angle hist vs Haar) {tv_angle:.4f} | max|E[R_ij]| {np.abs(mean).max():.3f} "
          f"| E[R_ij^2] range {sq.min():.3f}..{sq.max():.3f} | Re1 octants {np.round(occ, 3)}")
    print("   E[R]:", np.round(mean, 3).tolist())
stats(old, "aw221 sign(det)")
stats(new, "aw304 Haar    ")
