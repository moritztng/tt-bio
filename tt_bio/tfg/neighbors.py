"""Exact short-range pair search for TFG guidance.

The expensive TFG terms are sums over atom pairs closer than about 3 A: the Vina steric term (inter-chain pairs with
d < 0.775 (ri + rj)) and the rigid-body clash (moving-fixed pairs with d < 0.85 (ra + rb)). The dense reference
evaluates every pair; a uniform grid with cells at least as large as the cutoff finds every pair within the cutoff,
so the sums below run over exactly the pairs the dense code counts, and only the fp32 summation order differs.

Grid / pairs_within   every pair within r (27-cell neighbourhood, direct-indexed cells).
PairList              a pair list built with a skin and reused while no point has moved by the skin.
BoundField            voxel bounds that prove, per placed point, "touches nothing" or "severe overlap", so a pose
                      search evaluates exact pairs only where the bounds cannot decide.
SparseClash           the rigid clash terms (energy, severe pairs, depth, gradient) on a PairList, and the batched
                      pose scoring of the coarse searches.
"""

import torch

SOFT = 0.85
HARD = 0.75

_OFF27 = torch.stack(torch.meshgrid(*(torch.arange(-1, 2),) * 3, indexing="ij"), -1).reshape(27, 3)


class Grid:
    """Uniform grid of cell size h over a per-sample point cloud P [S, N, 3]."""

    def __init__(self, P, h):
        S, N, _ = P.shape
        self.S, self.N, self.h = S, N, float(h)
        self.lo = P.min(1).values - self.h
        c = torch.floor((P - self.lo[:, None]) / self.h).long()
        self.dims = c.reshape(-1, 3).max(0).values + 2
        dx, dy, dz = (int(v) for v in self.dims)
        self.ncell = dx * dy * dz
        key = (torch.arange(S)[:, None] * self.ncell + (c[..., 0] * dy + c[..., 1]) * dz + c[..., 2]).reshape(-1)
        self.order = torch.argsort(key, stable=True)
        self.cnt = torch.bincount(key, minlength=S * self.ncell)
        self.start = torch.cumsum(self.cnt, 0) - self.cnt

    def query(self, Q, sample):
        """Candidate pairs (q, j) for points Q [n, 3] of samples `sample` [n]: every P[sample, j] within h of Q[q]."""
        c = torch.floor((Q - self.lo[sample]) / self.h).long()
        nb = c[:, None, :] + _OFF27[None]
        ok = ((nb >= 0) & (nb < self.dims)).all(-1)
        dy, dz = int(self.dims[1]), int(self.dims[2])
        cid = sample[:, None] * self.ncell + (nb[..., 0] * dy + nb[..., 1]) * dz + nb[..., 2]
        cid = torch.where(ok, cid, torch.zeros_like(cid))
        n = torch.where(ok, self.cnt[cid], torch.zeros_like(cid)).reshape(-1)
        st = self.start[cid].reshape(-1)
        q = torch.arange(Q.shape[0]).repeat_interleave(27).repeat_interleave(n)
        first = torch.cumsum(n, 0) - n
        within = torch.arange(int(n.sum())) - first.repeat_interleave(n)
        return q, self.order[st.repeat_interleave(n) + within] % self.N


def pairs_within(Q, P, r):
    """Every (s, i, j) with |Q[s, i] - P[s, j]| < r; Q [S, M, 3], P [S, N, 3]. Query points are binned into the same
    cells as P, so the 27-cell lookup runs once per occupied query cell and expands into its atom pairs."""
    S, M, _ = Q.shape
    g = Grid(P, r)
    dy, dz = int(g.dims[1]), int(g.dims[2])
    # query cells on the grid grown by one cell per side; points beyond it have no neighbour cell
    c = torch.floor((Q - g.lo[:, None]) / g.h).long().reshape(-1, 3) + 1
    ex, ey, ez = (int(v) + 2 for v in g.dims)
    inside = torch.nonzero(((c >= 0) & (c < g.dims + 2)).all(-1)).squeeze(1)
    c = c[inside]
    key = (inside // M) * (ex * ey * ez) + (c[:, 0] * ey + c[:, 1]) * ez + c[:, 2]
    o = torch.argsort(key, stable=True)
    qo = inside[o]
    cells, qcnt = torch.unique_consecutive(key[o], return_counts=True)
    qstart = torch.cumsum(qcnt, 0) - qcnt
    rem = cells % (ex * ey * ez)
    nb = torch.stack([rem // (ey * ez), (rem // ez) % ey, rem % ez], -1)[:, None] - 1 + _OFF27[None]
    ok = ((nb >= 0) & (nb < g.dims)).all(-1)
    cid = torch.where(ok, (cells // (ex * ey * ez))[:, None] * g.ncell + (nb[..., 0] * dy + nb[..., 1]) * dz + nb[..., 2], 0)
    npc = torch.where(ok, g.cnt[cid], 0).reshape(-1)
    nq = qcnt.repeat_interleave(27)
    blk = nq * npc
    live = blk > 0
    blk, npc = blk[live], npc[live]
    idx = torch.arange(int(blk.sum())) - (torch.cumsum(blk, 0) - blk).repeat_interleave(blk)
    npr = npc.repeat_interleave(blk)
    q = qo[qstart.repeat_interleave(27)[live].repeat_interleave(blk) + idx // npr]
    j = g.order[g.start[cid.reshape(-1)[live]].repeat_interleave(blk) + idx % npr] % g.N
    s, i = q // M, q % M
    keep = (Q[s, i] - P[s, j]).square().sum(-1) < r * r
    return s[keep], i[keep], j[keep]


def square_length(v):
    """|v|^2 over the last axis as two adds of three products: about twice as fast as norm() on a size-3 axis. For
    bounds and thresholds (rounding differs from norm/cdist), not for values that must match them bitwise."""
    return (v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1]) + v[..., 2] * v[..., 2]


def cdist_norm(diff):
    """|diff| in torch.cdist's (donot_use_mm) order: squares rounded separately, summed (x + z) + y."""
    sq = diff * diff
    return torch.sqrt((sq[..., 0] + sq[..., 2]) + sq[..., 1])


class PairList:
    """Pairs (s, i, j) with |Q[s, i] - P[s, j]| < cut + skin at the build pose of Q (P static). Valid for any Q whose
    points are all within `skin` of the build pose, which `valid` checks; `build` rebuilds at a given pose."""

    def __init__(self, P, cut, skin=2.0):
        self.P, self.cut, self.skin0, self.Q = P, float(cut), float(skin), None

    def valid(self, Q):
        if self.Q is None:
            return False
        ref = self.Q if Q.dim() == self.Q.dim() else self.Q[:, None]
        return float((Q - ref).norm(dim=-1).max()) < self.skin

    def build(self, Q, reach=0.0):
        """Build at Q [S, M, 3] with a skin that also covers moves of up to `reach` from Q."""
        self.skin = max(self.skin0, 1.25 * float(reach) + 0.1)
        self.s, self.i, self.j = pairs_within(Q, self.P, self.cut + self.skin)
        self.Q = Q.clone()
        return self


class SparseClash:
    """The rigid clash terms of tt_bio.tfg.rigid.clash_terms on exact pair lists.

    terms(x [S, M, 3] or [S, K, M, 3]) -> energy sum relu(0.85 rs - d)^2, severe-pair keys, depth
    max relu(0.75 rs - d), gradient of 10 * sum overlap^2. Severe pairs are returned as sorted int64 keys
    (s * M * N + i * N + j) so two poses can be compared whatever list they were evaluated on.
    """

    batched = True                     # rigid_descent may hand terms() several poses per sample at once

    def __init__(self, fixed, ra, rb):
        self.fixed, self.ra, self.rb = fixed, ra, rb
        S, self.N, _ = fixed.shape
        self.M = ra.shape[0]
        self.cut = SOFT * float(ra.max() + rb.max())
        self.list = PairList(fixed, self.cut)
        self._field = None

    def ensure(self, *poses):
        """Make the pair list valid for every pose given (the first is the build pose if a rebuild is needed)."""
        if all(self.list.valid(p) for p in poses):
            return
        ref = poses[0] if poses[0].dim() == 3 else poses[0][:, 0]
        reach = max(float((p - (ref if p.dim() == 3 else ref[:, None])).norm(dim=-1).max()) for p in poses)
        self.list.build(ref, reach)

    def terms(self, x, want_gradient=False):
        single = x.dim() == 3
        X = x[:, None] if single else x
        self.ensure(X)
        S, K, M, _ = X.shape
        s, i, j = self.list.s, self.list.i, self.list.j
        diff = X[s, :, i] - self.fixed[s, j][:, None]                     # [P, K, 3]
        d = cdist_norm(diff).clamp_min(1e-6)
        rs = (self.ra[i] + self.rb[j])[:, None]
        overlap = torch.relu(SOFT * rs - d)
        energy = torch.zeros(S, K, dtype=X.dtype).index_add_(0, s, overlap.square())
        depth = torch.zeros(S, K, dtype=X.dtype).index_reduce_(0, s, torch.relu(HARD * rs - d), "amax")
        sev = d < HARD * rs                                                 # [P, K]
        p_idx, k_idx = torch.nonzero(sev, as_tuple=True)
        keys = (s[p_idx] * M + i[p_idx]) * self.N + j[p_idx]
        severe = SevereSet(keys, k_idx, S, K, M * self.N)
        grad = None
        if want_gradient:
            coeff = torch.where(d > 1e-6, -20.0 * overlap / d, torch.zeros_like(d))
            g = torch.zeros(S * M, K, 3, dtype=X.dtype)
            g.index_add_(0, s * M + i, coeff[..., None] * diff)
            grad = g.view(S, M, K, 3).transpose(1, 2)
        if single:
            return energy[:, 0], severe.item(0), depth[:, 0], None if grad is None else grad[:, 0]
        return energy, severe, depth, grad

    def no_new(self, previous_x, previous, x, current):
        """True per sample when the pose x has no severe pair that previous_x lacked."""
        return ~current.new_since(previous)

    @staticmethod
    def count(severe):
        return severe.count()

    def score(self, X, samples):
        """Coarse-search clash of many placements: X [n, B, M, 3] for the samples `samples` [n] -> energy [n, B]
        (meaningful where not severe) and severe [n, B]. Points the bound field proves far are skipped and a
        placement the field proves severe is not evaluated further; every other point is evaluated exactly."""
        if self._field is None:
            self._grid = Grid(self.fixed, self.cut)
            self._field = BoundField(self.fixed, self.rb, float(self.ra.max()))
        n, B, M, _ = X.shape
        pts = X.reshape(-1, 3)
        item = torch.arange(n * B).repeat_interleave(M)
        sample = samples.repeat_interleave(B * M)
        ra = self.ra.repeat(n * B)
        far, sev_p = self._field.classify(pts, sample, ra)
        sev = torch.zeros(n * B, dtype=torch.long).index_add_(0, item, sev_p.long()) > 0
        live = torch.nonzero(~far & ~sev[item]).squeeze(1)
        q, j = self._grid.query(pts[live], sample[live])
        q = live[q]
        d = cdist_norm(pts[q] - self.fixed[sample[q], j]).clamp_min(1e-6)
        rs = ra[q] + self.rb[j]
        it = item[q]
        energy = torch.zeros(n * B, dtype=X.dtype).index_add_(0, it, torch.relu(SOFT * rs - d).square())
        sev |= torch.zeros(n * B, dtype=torch.long).index_add_(0, it, (d < HARD * rs).long()) > 0
        return energy.view(n, B), sev.view(n, B)


class SevereSet:
    """Severe pairs of S x K evaluated poses: sorted keys with the pose index k they belong to."""

    def __init__(self, keys, k, S, K, span):
        self.keys, self.k, self.S, self.K, self.span = keys, k, S, K, span

    def item(self, k):
        m = self.k == k
        return SevereSet(self.keys[m], torch.zeros_like(self.k[m]), self.S, 1, self.span)

    def count(self):
        """Severe pairs per sample (and per pose when K > 1)."""
        c = torch.zeros(self.S * self.K, dtype=torch.long)
        c.index_add_(0, (self.keys // self.span) * self.K + self.k, torch.ones_like(self.keys))
        return c.view(self.S, self.K) if self.K > 1 else c

    def new_since(self, previous):
        """True per sample (and pose) when this set holds a pair that `previous` (one pose per sample) does not."""
        new = ~torch.isin(self.keys, previous.keys)
        c = torch.zeros(self.S * self.K, dtype=torch.long)
        c.index_add_(0, (self.keys // self.span) * self.K + self.k, new.long())
        c = c > 0
        return c.view(self.S, self.K) if self.K > 1 else c


class BoundField:
    """Voxel bounds on min_j (|x - b_j| - f rb_j), per sample: a lower bound for f = 0.85, an upper bound for f = 0.75.
    For a point x in a voxel with centre c and half diagonal delta, |x - b| lies within |c - b| +- delta, so the voxel value proves that x overlaps nothing
    (soft bound - delta >= 0.85 ra) or that x has a severe pair (hard bound + delta < 0.75 ra). 1e-3 A margin."""

    def __init__(self, fixed, rb, ra_max, v=1.5, margin=1e-3):
        S, N, _ = fixed.shape
        self.v, self.delta, self.margin = v, 0.5 * v * 3**0.5, margin
        reach = SOFT * (ra_max + float(rb.max())) + self.delta + margin
        k = int(reach / v) + 2
        r = torch.arange(-k, k + 1)
        off = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
        # an atom sits within delta of its voxel centre, so offsets beyond this ball reach no voxel in range
        off = off[off.float().norm(dim=-1) * v - self.delta < reach + SOFT * float(rb.max())]
        # k voxels of padding: every atom's offset ball lies inside the grid, so a neighbour is a fixed flat stride
        self.lo = fixed.min(1).values - (k + 1) * v
        self.dims = torch.floor((fixed.max(1).values - self.lo) / v).long().max(0).values + k + 2
        dx, dy, dz = (int(t) for t in self.dims)
        self.nvox = dx * dy * dz
        vox = torch.floor((fixed - self.lo[:, None]) / v)
        rel = self.lo[:, None] + (vox + 0.5) * v - fixed                    # atom -> its voxel centre, |rel| <= delta
        base = (torch.arange(S)[:, None] * self.nvox + (vox[..., 0] * dy + vox[..., 1]) * dz + vox[..., 2]).long()
        step = (off[:, 0] * dy + off[:, 1]) * dz + off[:, 2]
        flat = (base.reshape(-1, 1) + step).reshape(-1)
        # distance from each neighbouring voxel centre to the atom, |rel + o|^2 = |rel|^2 + 2 rel.o + |o|^2 from small
        # relative vectors (fp32 error ~1e-6 A, far inside the margin)
        o = off.float() * v
        rel = rel.reshape(-1, 3)
        dist = torch.addmm(rel.square().sum(-1, keepdim=True) + o.square().sum(-1), rel, 2 * o.T).clamp_min_(0).sqrt_()
        self.soft = torch.full((S * self.nvox,), float("inf")).scatter_reduce_(
            0, flat, dist.sub_(SOFT * rb.repeat(S)[:, None]).reshape(-1), "amin")
        # the atom attaining the soft bound has |c - b| - 0.75 rb = soft + 0.1 rb <= soft + 0.1 max(rb): an upper bound
        # on the hard field, which is all the severe proof needs
        self.lift = (SOFT - HARD) * float(rb.max())

    def classify(self, X, sample, ra):
        """X [n, 3] -> (far [n], severe [n]) bool."""
        c = torch.floor((X - self.lo[sample]) / self.v).long()
        inb = ((c >= 0) & (c < self.dims)).all(-1)
        dy, dz = int(self.dims[1]), int(self.dims[2])
        flat = torch.where(inb, sample * self.nvox + (c[:, 0] * dy + c[:, 1]) * dz + c[:, 2], 0)
        soft = torch.where(inb, self.soft[flat], float("inf"))
        return soft - self.delta >= SOFT * ra + self.margin, (soft + self.lift) + self.delta < HARD * ra - self.margin
