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
LIST_BYTES = 1024  # compute.cpp schedule lists: TRISC0 has ~2 KB for them and its stack


def pad32(n: int) -> int:
    return -(-n // TILE) * TILE


@dataclass
class Group:
    """One core group's constants (float64) and tile schedule."""
    terms: list                      # output channel ranges [k, w0, w1], in column order
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

    def __init__(self, tp, fc, groups: list | None = None, l1_budget: int = 1000 * 1024):
        """``groups``: per core group a list of output terms ``k`` or channel ranges
        ``(k, w0, w1)``; default one group per term, split along ``w`` to fit ``l1_budget``."""
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
        muls = [t.mul for t in out_terms]
        wstart, s = [], 0
        for (u, v, w) in tp.weight_shapes:
            if v != 1:
                raise NotImplementedError("the fused form assumes sh multiplicity 1")
            wstart.append(s)
            s += u * w

        self.groups, self.perm_cols = [], {}
        tile0 = 0
        if groups is None:
            # one group per term, its output channels split until the group fits the L1 budget
            for k in range(len(out_terms)):
                for n in range(1, muls[k] + 1):
                    c = -(-muls[k] // n)
                    parts = [[(k, w0, min(w0 + c, muls[k]))] for w0 in range(0, muls[k], c)]
                    gs = [self._group(tp, pt, wstart, w2, b2, 0) for pt in parts]
                    if max(l1_bytes(self, g) for g in gs) <= l1_budget:
                        break
                for pt in parts:
                    g = self._group(tp, pt, wstart, w2, b2, tile0)
                    self.groups.append(g)
                    tile0 += g.n_out_tiles
        else:
            parts = [[(x, 0, muls[x]) if isinstance(x, int) else tuple(x) for x in g] for g in groups]
            cover = sorted(p for g in parts for p in g)
            want, ok = [], True
            for k in range(len(out_terms)):
                ks = [p for p in cover if p[0] == k]
                ok &= bool(ks) and ks[0][1] == 0 and ks[-1][2] == muls[k] and all(
                    a[2] == b[1] for a, b in zip(ks, ks[1:]))
            if not ok or len({p[0] for p in cover}) != len(out_terms):
                raise ValueError(f"groups {groups} must partition the output channels {muls}")
            for pt in parts:
                g = self._group(tp, pt, wstart, w2, b2, tile0)
                self.groups.append(g)
                tile0 += g.n_out_tiles
        self.out_pad = tile0 * TILE
        for g in self.groups:
            col0 = g.out_tile0 * TILE
            for k, w0, w1 in g.terms:
                K = 2 * tp.out_irreps.terms[k].l + 1
                for t in range(K * (w1 - w0)):
                    self.perm_cols[col0 + t] = tp._out_slices[k].start + w0 * K + t
                col0 += K * (w1 - w0)
        # kernel column -> e3nn column, as a 0/1 matrix [out_pad, out_dim]
        self.perm = torch.zeros(self.out_pad, self.out_dim, dtype=f64)
        for kc, ec in self.perm_cols.items():
            self.perm[kc, ec] = 1.0

    def _group(self, tp, parts, wstart, w2, b2, tile0) -> Group:
        f64 = torch.float64
        # kernel columns of this group: its channel ranges back to back, e3nn order inside each
        width = sum((2 * tp.out_irreps.terms[k].l + 1) * (w1 - w0) for k, w0, w1 in parts)
        g = Group(terms=[list(p) for p in parts], out_tile0=tile0, n_out_tiles=pad32(width) // TILE)
        col0 = tile0 * TILE
        G_cols, L2_cols = [], []
        for k, w0, w1 in parts:
            tk = tp.out_irreps.terms[k]
            K = 2 * tk.l + 1
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
            for w in range(w0, w1):
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
                        oc = (col0 - tile0 * TILE) + (w - w0) * K + m   # column within the group
                        g.mac.append((wi, ybase + m * nblk + blk, oc // TILE, oc % TILE))
            col0 += K * (w1 - w0)
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


# ---------------------------------------------------------------------------------------------
# The card: one generic_op, core groups side by side.

def _schedule(plan: EdgeTilePlan, g: Group):
    """Flatten a group's schedule into the kernel's define lists."""
    bj = sorted(set(g.ag_j))
    ab = sorted({(bj.index(j), kt) for j, kts in zip(g.ag_j, g.ag_ktiles) for kt in kts})
    ab_ix = {p: i for i, p in enumerate(ab)}
    y_entries = [[] for _ in range(g.n_y)]
    gblocks, nG = [], 0
    T = TILE
    for t, (j, yi, kts) in enumerate(zip(g.ag_j, g.ag_y, g.ag_ktiles)):
        for kt in kts:
            y_entries[yi].append((ab_ix[(bj.index(j), kt)], nG))
            gblocks.append(g.G[kt * T:(kt + 1) * T, t * T:(t + 1) * T])
            nG += 1
    y_list = []
    for ents in y_entries:
        y_list.append(len(ents))
        for a, gi in ents:
            y_list += [a, gi]
    by_w = [[] for _ in range(g.n_w)]
    for wi, yi, ot, lane in g.mac:
        by_w[wi].append((yi, ot, lane))
    w_list = []
    for ents in by_w:
        ents.sort(key=lambda e: e[1])
        if len(ents) > 4 or ents[-1][1] - ents[0][1] > 3:
            raise NotImplementedError("a weight tile feeds more than four DST slots")
        w_list.append(len(ents))
        for e in ents:
            w_list += list(e)
    # compute.cpp keeps these lists as uint8_t in TRISC local memory (4 KB with the firmware's
    # data and the stack): lists that outgrow it overflow the stack and hang the unpacker.
    lists = [x for p in ab for x in p] + y_list + w_list
    if max(lists) > 255 or len(lists) > LIST_BYTES:
        raise NotImplementedError(f"schedule lists of {len(lists)} entries (max {max(lists)}) "
                                  f"exceed the kernel's {LIST_BYTES}-byte uint8 budget")
    Gpack = torch.cat(gblocks, 1) if gblocks else torch.zeros(T, T, dtype=torch.float64)
    return dict(bj=bj, ab=[x for p in ab for x in p], y=y_list, w=w_list, Gpack=Gpack,
                nG=max(nG, 1), nab=len(ab), ops=len(ab) + sum(len(k) for k in g.ag_ktiles)
                + g.n_w * (plan.hidden_pad // T) + 2 * len(g.mac))


def cb_pages(plan: EdgeTilePlan, g: Group, s: dict) -> dict:
    """The kernel's circular buffers for one group: index -> (pages, fp32)."""
    KH, KA, nbj = plan.hidden_pad // TILE, plan.in_pad // TILE, len(s["bj"])
    return {0: (2 * KH, False), 1: (2 * KA, False), 2: (2 * nbj, False), 3: (g.n_w * KH, False),
            4: (s["nG"], False), 5: (33, False), 6: (s["nab"], False), 7: (g.n_y, False),
            8: (2, False), 9: (4, False), 16: (2 * g.n_out_tiles, True)}


def l1_bytes(plan: EdgeTilePlan, g: Group) -> int:
    return sum(n * (4096 if f32 else 2048) for n, f32 in cb_pages(plan, g, _schedule(plan, g)).values())


class DeviceEdgeTile:
    """``(edge_attr, x1, sh) -> tp(x1, sh, fc(edge_attr))`` as one fused kernel per call.

    Returns ``[E, plan.out_pad]`` float32 in the kernel's column order; ``plan.perm`` (0/1) takes
    it to e3nn's, and since it is a column permutation it commutes with the scatter.
    """

    KDIR = None

    def __init__(self, device, tp, fc, groups=None, *, math_fidelity=None, ablate: int = 0,
                 tap: int = 0, only=None):
        import ttnn
        from pathlib import Path
        self.ttnn, self.device = ttnn, device
        self.plan = plan = EdgeTilePlan(tp, fc, groups)
        self.kdir = Path(__file__).resolve().parents[1] / "kernels" / "dfd_edge_tp"
        self.fidelity = math_fidelity or ttnn.MathFidelity.HiFi4
        bf = ttnn.bfloat16

        def up(t, dt=bf):
            return ttnn.from_torch(t.to(torch.float32), dtype=dt, layout=ttnn.TILE_LAYOUT,
                                   device=device)

        self.w1 = up(plan.W1)
        self.b1 = up(plan.b1.reshape(1, -1))
        erep = torch.zeros(plan.sh_pad, TILE * plan.n_sh, dtype=torch.float64)
        for j in range(plan.n_sh):
            erep[j, j * TILE:(j + 1) * TILE] = 1.0
        self.erep = up(erep)
        oh = torch.zeros(TILE, 33 * TILE, dtype=torch.float64)
        for c in range(TILE):
            oh[:, c * TILE + c] = 1.0
        self.oh = up(oh)
        self.perm = up(plan.perm, ttnn.float32)
        self.sched = [_schedule(plan, g) for g in plan.groups]
        self.L2 = [up(g.L2) for g in plan.groups]
        self.G = [up(s["Gpack"]) for s in self.sched]
        self._pd = {}
        self.ablate, self.tap = ablate, tap
        # diagnostic: launch only these groups (their output columns; the rest stay unwritten)
        self.only = list(range(len(plan.groups))) if only is None else list(only)
        # diagnostic tap (see compute.cpp): each group writes its Y / AB / Wt tiles instead
        if tap:
            self.tap_n = [[g.n_y, s["nab"], g.n_w][tap - 1] for g, s in zip(plan.groups, self.sched)]
        else:
            self.tap_n = [g.n_out_tiles for g in plan.groups]
        self.tap0 = [sum(self.tap_n[:i]) for i in range(len(self.tap_n))]
        self.out_w = sum(self.tap_n) * TILE

    def _cores(self):
        grid = self.device.compute_with_storage_grid_size()
        cores = [(x, y) for y in range(grid.y) for x in range(grid.x)]
        ops = [s["ops"] for s in self.sched]
        n = [max(1, round(len(cores) * o / sum(ops))) for o in ops]
        while sum(n) > len(cores):
            n[n.index(max(n))] -= 1
        out, lo = [], 0
        for k in n:
            out.append(cores[lo:lo + k])
            lo += k
        return out

    def _build(self, ET, h, a, b, out):
        ttnn, plan = self.ttnn, self.plan
        TA = ttnn.TensorAccessorArgs
        src = ttnn.KernelDescriptor.SourceType.FILE_PATH
        KH, KA, KB = plan.hidden_pad // TILE, plan.in_pad // TILE, plan.n_sh
        OUTW = self.out_w // TILE
        kernels, cbs = [], []
        bf, f32 = ttnn.bfloat16, ttnn.float32
        tb, tf = 2048, 4096

        def cb(idx, n, cr, dt=bf, page=tb):
            fmt = ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)
            return ttnn.CBDescriptor(total_size=n * page, core_ranges=cr, format_descriptors=[fmt])

        for gi, (g, s, cl) in enumerate(zip(plan.groups, self.sched, self._cores())):
            if gi not in self.only:
                continue
            cr = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y))
                                    for x, y in cl])
            per = -(-ET // len(cl))
            rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for i, (x, y) in enumerate(cl):
                first = min(i * per, ET)
                n = min(per, ET - first)
                rrt[x][y] = [first, n]
                crt[x][y] = [n]
                wrt[x][y] = [first, n]
            nbj, nab, NW, NG = len(s["bj"]), s["nab"], g.n_w, s["nG"]
            NOUT = self.tap_n[gi]
            lst = lambda v: ",".join(str(int(x)) for x in v) if v else "0"
            rd = ttnn.KernelDescriptor(
                kernel_source=str(self.kdir / "reader.cpp"), source_type=src, core_ranges=cr,
                compile_time_args=[KH, nbj, KA, KB, NW, NG]
                + list(TA(h).get_compile_time_args()) + list(TA(a).get_compile_time_args())
                + list(TA(b).get_compile_time_args()) + list(TA(self.L2[gi]).get_compile_time_args())
                + list(TA(self.G[gi]).get_compile_time_args()) + list(TA(self.oh).get_compile_time_args()),
                defines=[("BJ_LIST", lst(s["bj"]))], runtime_args=rrt,
                common_runtime_args=[0] * 6, config=ttnn.ReaderConfigDescriptor())
            wr = ttnn.KernelDescriptor(
                kernel_source=str(self.kdir / "writer.cpp"), source_type=src, core_ranges=cr,
                compile_time_args=[NOUT, self.tap0[gi], OUTW] + list(TA(out).get_compile_time_args()),
                runtime_args=wrt, common_runtime_args=[0], config=ttnn.WriterConfigDescriptor())
            cfg = ttnn.ComputeConfigDescriptor(math_fidelity=self.fidelity, fp32_dest_acc_en=True)
            cp = ttnn.KernelDescriptor(
                kernel_source=str(self.kdir / "compute.cpp"), source_type=src, core_ranges=cr,
                compile_time_args=[KH, nbj, KA, nab, g.n_y, NW, NOUT, NW * KH, NG, self.ablate,
                                   self.tap],
                defines=[("AB_LIST", lst(s["ab"])), ("Y_LIST", lst(s["y"])),
                         ("W_LIST", lst(s["w"]))],
                runtime_args=crt, config=cfg)
            kernels += [rd, wr, cp]
            pages = cb_pages(plan, g, s)
            pages[16] = (2 * NOUT, True)                  # a tap writes NOUT tiles of its own
            cbs += [cb(i, n, cr, f32, tf) if wide else cb(i, n, cr) for i, (n, wide) in pages.items()]
        return kernels, cbs

    def __call__(self, edge_attr, x1, sh, h=None):
        """TILE bf16 or fp32 inputs ``[E, *]`` (E a multiple of 32) -> fp32 ``[E, out_pad]``.

        *h*, when given, is the FC block's hidden layer already built and *edge_attr* is not read.
        """
        ttnn = self.ttnn
        bf = ttnn.bfloat16
        E = int(x1.shape[0])
        cast = lambda t: t if t.dtype == bf else ttnn.typecast(t, bf)
        h = (ttnn.relu(ttnn.linear(cast(edge_attr), self.w1, bias=self.b1, dtype=bf))
             if h is None else cast(h))
        a = cast(x1)
        b = ttnn.matmul(cast(sh), self.erep, dtype=bf)
        out = ttnn.empty(ttnn.Shape([E, self.out_w]), ttnn.float32, ttnn.TILE_LAYOUT,
                         self.device, ttnn.DRAM_MEMORY_CONFIG)
        key = E // TILE
        if key not in self._pd:
            self._pd[key] = self._build(key, h, a, b, out)
        kernels, cbs = self._pd[key]
        for i, gi in zip(range(0, len(kernels), 3), self.only):
            kernels[i].common_runtime_args = [h.buffer_address(), a.buffer_address(),
                                              b.buffer_address(), self.oh.buffer_address(),
                                              self.L2[gi].buffer_address(),
                                              self.G[gi].buffer_address()]
            kernels[i + 1].common_runtime_args = [out.buffer_address()]
        pd = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        ttnn.generic_op([h, a, b, *self.L2, *self.G, self.oh, out], pd)
        return out
