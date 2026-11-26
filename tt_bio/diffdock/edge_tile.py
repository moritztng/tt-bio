"""The fused edge-tile form of DiffDock's FC block + tensor product: constants and schedule.

:mod:`tt_bio.diffdock.device` runs the edge convolution as ttnn GEMMs over a materialised term
axis ``(k, u, w)``, about 6,500 columns wide at the conv signature. Every column goes through
DRAM several times, and that is where a step's 4.7 s goes. This module lays the same arithmetic
out for a kernel that keeps the per-edge weights in L1: for a tile of 32 edges it generates the
weights one 32x32 tile at a time and contracts each one before it would ever be written.

Per output irrep ``k`` (``W`` channels, ``K = 2l+1`` components) the paths into it are
concatenated along their input channel ``u`` into one axis ``c`` (``Ucat`` wide, padded to whole
tiles of 32 lanes). Then, for one edge,

    Y[c, m]  = sum_j b[j] * AG_j[c, m],      AG_j[c, m] = sum_i a[u(c), i] C_p(c)[i, j, m]
    Wt[w, c] = h' @ L2[:, (k, w, c)]         (bias folded in: h'[hidden] = 1)
    out[w, m] = sum_c Wt[w, c] * Y[c, m]

with each path's normalisation folded into its ``L2`` columns. On tiles that is

* ``AG`` tiles: ``a @ G``, one ``(k, m, block, j)`` tile per sh component ``j`` a block uses;
* ``Y`` tiles: ``sum_j AG ⊙ bcast_col(b_j)``, one per ``(k, m, block)``;
* ``Wt`` tiles: ``h' @ L2``, one per ``(k, w, block)``;
* the contraction: ``(Wt ⊙ Y) @ onehot(col)``, accumulated into the output tile, which sums
  over the 32 lanes and drops the result into the output column. ``onehot(c)`` is 1 in every
  row of column ``c``.

Output terms are split into *groups*, each owning whole output tiles, so cores of different
groups hold different weights in L1 and never write the same tile. The kernel's output columns
are therefore a fixed permutation of e3nn's (``perm``); it is 0/1 and applied after the scatter,
on node rows.

:func:`emulate` runs exactly this schedule in torch. In float64 it matches the host module to
roundoff, which proves the layout before a card is involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

TILE = 32


def pad32(n: int) -> int:
    return -(-n // TILE) * TILE


@dataclass
class Group:
    """One core group's constants (float64) and tile schedule."""
    terms: list                      # output term indices k
    out_tile0: int                   # first kernel-output tile this group writes
    n_out_tiles: int
    G: Tensor = None                 # [in_pad, n_ag * 32]
    L2: Tensor = None                # [hidden_pad, n_w * 32]
    ag_j: list = field(default_factory=list)      # per AG tile: sh component j
    ag_y: list = field(default_factory=list)      # per AG tile: the Y tile it accumulates into
    ag_ktiles: list = field(default_factory=list)  # per AG tile: input K-tiles with a nonzero G block
    n_y: int = 0
    # contraction: (w tile, y tile, output tile relative to out_tile0, output lane)
    mac: list = field(default_factory=list)
    n_w: int = 0


class EdgeTilePlan:
    """Constants and schedule of the fused form of ``tp(x1, x2, fc(edge_attr))``."""

    def __init__(self, tp, fc, groups: list[list[int]] | None = None):
        f64 = torch.float64
        w1, b1 = fc[0].weight.detach().to(f64), fc[0].bias.detach().to(f64)
        w2, b2 = fc[-1].weight.detach().to(f64), fc[-1].bias.detach().to(f64)
        self.hidden, self.n_feat = w1.shape
        self.hidden_pad = pad32(self.hidden + 1)          # one lane for the folded bias
        self.feat_pad = pad32(self.n_feat)
        self.in_dim, self.sh_dim, self.out_dim = tp.in_irreps.dim, tp.sh_irreps.dim, tp.out_irreps.dim
        self.in_pad, self.sh_pad = pad32(self.in_dim), pad32(self.sh_dim)
        self.n_sh = self.sh_dim

        # First FC layer, padded; the lane `hidden` comes out as relu(0 + 1) = 1.
        self.W1 = torch.zeros(self.feat_pad, self.hidden_pad, dtype=f64)
        self.W1[:self.n_feat, :self.hidden] = w1.T
        self.b1 = torch.zeros(self.hidden_pad, dtype=f64)
        self.b1[:self.hidden] = b1
        self.b1[self.hidden] = 1.0

        out_terms = tp.out_irreps.terms
        if groups is None:
            groups = [[k] for k in range(len(out_terms))]
        if sorted(k for g in groups for k in g) != list(range(len(out_terms))):
            raise ValueError(f"groups {groups} must partition the {len(out_terms)} output terms")

        wstart, s = [], 0
        for (u, v, w) in tp.weight_shapes:
            if v != 1:
                raise NotImplementedError("the fused form assumes sh multiplicity 1")
            wstart.append(s)
            s += u * w

        self.groups, self.perm_cols = [], {}
        tile0 = 0
        for terms in groups:
            g = self._group(tp, terms, wstart, w2, b2, tile0)
            self.groups.append(g)
            tile0 += g.n_out_tiles
        self.out_pad = tile0 * TILE
        # kernel column -> e3nn column, as a 0/1 matrix [out_pad, out_dim]
        self.perm = torch.zeros(self.out_pad, self.out_dim, dtype=f64)
        for kc, ec in self.perm_cols.items():
            self.perm[kc, ec] = 1.0

    def _group(self, tp, terms, wstart, w2, b2, tile0) -> Group:
        f64 = torch.float64
        out_sl = tp._out_slices
        # kernel columns of this group: its terms back to back, e3nn order inside each term
        width = sum(tp.out_irreps.terms[k].dim for k in terms)
        g = Group(terms=list(terms), out_tile0=tile0, n_out_tiles=pad32(width) // TILE)
        col0 = tile0 * TILE
        G_cols, L2_cols = [], []
        for k in terms:
            tk = tp.out_irreps.terms[k]
            K, Wm = 2 * tk.l + 1, tk.mul
            for t in range(K * Wm):
                self.perm_cols[col0 + t] = out_sl[k].start + t
            # the concatenated channel axis: paths into k, sorted by sh term so a block uses few j
            into = sorted((n for n, p in enumerate(tp.paths) if p[2] == k),
                          key=lambda n: (tp.paths[n][1], n))
            lanes = [(n, u) for n in into for u in range(tp.weight_shapes[n][0])]
            nblk = pad32(len(lanes)) // TILE
            # Y tiles (k, m, blk) and the AG tiles feeding them
            for m in range(K):
                for blk in range(nblk):
                    yi = g.n_y
                    g.n_y += 1
                    blanes = lanes[blk * TILE:(blk + 1) * TILE]
                    js = sorted({tp._sh_slices[tp.paths[n][1]].start + jj
                                 for n, _ in blanes
                                 for jj in range(2 * tp.sh_irreps.terms[tp.paths[n][1]].l + 1)})
                    for j in js:
                        col = torch.zeros(self.in_pad, TILE, dtype=f64)
                        for lane, (n, u) in enumerate(blanes):
                            i, jt, _ = tp.paths[n]
                            jj = j - tp._sh_slices[jt].start
                            if not 0 <= jj < 2 * tp.sh_irreps.terms[jt].l + 1:
                                continue
                            I = 2 * tp.in_irreps.terms[i].l + 1
                            C = tp._w3j64[n]
                            for ii in range(I):
                                col[tp._in_slices[i].start + u * I + ii, lane] = C[ii, jj, m]
                        g.ag_j.append(j)
                        g.ag_y.append(yi)
                        g.ag_ktiles.append([kt for kt in range(self.in_pad // TILE)
                                            if col[kt * TILE:(kt + 1) * TILE].abs().sum() > 0])
                        G_cols.append(col)
            ybase = g.n_y - K * nblk
            # W tiles (k, w, blk) and their contraction entries
            for w in range(Wm):
                for blk in range(nblk):
                    col = torch.zeros(self.hidden_pad, TILE, dtype=f64)
                    for lane, (n, u) in enumerate(lanes[blk * TILE:(blk + 1) * TILE]):
                        pw = tp.path_weights[n]
                        widx = wstart[n] + u * tp.weight_shapes[n][2] + w
                        col[:self.hidden, lane] = pw * w2[widx]
                        col[self.hidden, lane] = pw * b2[widx]
                    wi = g.n_w
                    g.n_w += 1
                    L2_cols.append(col)
                    for m in range(K):
                        oc = (col0 - tile0 * TILE) + w * K + m   # column within the group
                        g.mac.append((wi, ybase + m * nblk + blk, oc // TILE, oc % TILE))
            col0 += K * Wm
        g.G = torch.cat(G_cols, 1)
        g.L2 = torch.cat(L2_cols, 1)
        return g

    def sh_wide(self, x2: Tensor) -> Tensor:
        """``[E, sh]`` -> ``[E, 32 * sh]`` with component ``j`` in column ``32 j``: the
        column-0 operand of a broadcast multiply, one tile per component."""
        out = x2.new_zeros(x2.shape[0], TILE * self.n_sh)
        out[:, ::TILE] = x2[:, :self.n_sh]
        return out

    def hidden_act(self, edge_attr: Tensor) -> Tensor:
        ea = torch.nn.functional.pad(edge_attr, (0, self.feat_pad - edge_attr.shape[1]))
        return torch.relu(ea @ self.W1.to(ea.dtype) + self.b1.to(ea.dtype))


def emulate(plan: EdgeTilePlan, edge_attr: Tensor, x1: Tensor, x2: Tensor) -> Tensor:
    """The kernel's schedule in torch, tile for tile; returns e3nn's output layout."""
    dt = x1.dtype
    h = plan.hidden_act(edge_attr.to(dt))
    a = torch.nn.functional.pad(x1, (0, plan.in_pad - x1.shape[1]))
    b = x2[:, :plan.n_sh]
    E = x1.shape[0]
    out = x1.new_zeros(E, plan.out_pad)
    T = TILE
    for g in plan.groups:
        AG = a @ g.G.to(dt)
        Y = x1.new_zeros(E, g.n_y * T)
        for t, (j, yi) in enumerate(zip(g.ag_j, g.ag_y)):
            Y[:, yi * T:(yi + 1) * T] += AG[:, t * T:(t + 1) * T] * b[:, j:j + 1]
        Wt = h @ g.L2.to(dt)
        base = g.out_tile0 * T
        for wi, yi, ot, lane in g.mac:
            out[:, base + ot * T + lane] += (Wt[:, wi * T:(wi + 1) * T]
                                             * Y[:, yi * T:(yi + 1) * T]).sum(1)
    return out @ plan.perm.to(dt)
