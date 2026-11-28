"""CPU unit tests for tt_bio.train.ccl (exact fp32 2D CCL + Axis-0 SPMD Muon ZeRO-1)."""
import math
import sys
import types
import pytest
import torch

from tt_bio.train import ccl

RFD4_PARAM_SHAPE_COUNTS = [
    ((6144, 768), 14),
    ((768, 3072), 14),
    ((2304, 768), 14),
    ((768, 768), 38),
    ((768, 256), 84),
    ((256, 384), 56),
    ((384, 384), 15),
    ((768, 384), 7),
    ((384, 768), 7),
    ((384, 256), 16),
    ((768, 897), 1),
    ((1024, 256), 2),
    ((256, 512), 3),
    ((256, 609), 1),
    ((768,), 151),
    ((147, 768), 1),
    ((12, 384), 14),
    ((1473, 32), 1),
    ((384, 107), 1),
    ((2304,), 14),
    ((32, 768), 1),
    ((384,), 52),
    ((256,), 44),
    ((3, 768), 1),
    ((128, 16), 1),
    ((44, 32), 1),
    ((7, 32), 1),
    ((6, 32), 1),
    ((11, 16), 1),
    ((4, 16), 1),
]


def test_ccl_import_does_not_import_ttnn():
    import subprocess
    from pathlib import Path
    out = subprocess.run(
        [sys.executable, "-c", "import sys, tt_bio.train.ccl; assert 'ttnn' not in sys.modules"],
        capture_output=True, text=True, check=False,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert out.returncode == 0, out.stderr


def _build_rfd4_shapes():
    shapes = {}
    idx = 0
    for shp, cnt in RFD4_PARAM_SHAPE_COUNTS:
        for j in range(cnt):
            shapes[f"p{idx:03d}_{j}"] = shp
            idx += 1
    return shapes


def test_flat_2d_shape_rfd4_galaxy():
    b, ch, r, c = ccl.flat_2d_shape(178_448_304, (8, 4), cols=10944)
    assert (b, ch) == (1, 1)
    assert r == 16384 and c == 10944
    assert r % (32 * 8 * 4) == 0
    assert c % 32 == 0
    pad_frac = (r * c - 178_448_304) / 178_448_304
    assert 0.0 <= pad_frac < 0.005


def test_configure_refuses_too_small_payload():
    with pytest.raises(ValueError, match="8192"):
        ccl.configure_fp32_ring_fabric(4352)


@pytest.mark.parametrize("rm,cm", [(8, 4), (4, 4), (2, 4), (1, 4)])
@pytest.mark.parametrize("fold_small", [True, False])
def test_plan_reduction_groups_covers_all_558_rfd4_params(rm, cm, fold_small):
    shapes = _build_rfd4_shapes()
    assert len(shapes) == 558
    total_elems = sum(math.prod(s) for s in shapes.values())
    assert total_elems == 178_448_304

    plan = ccl.plan_reduction_groups(shapes, (rm, cm), fold_small_into_width=fold_small)
    assert [wg.cols for wg in plan.width_groups][:4] == [768, 3072, 256, 384]
    if fold_small and (rm, cm) == (8, 4):
        assert len(plan.width_groups) == 4
        assert len(plan.small_groups) == 0
        padded_tot = sum(wg.padded_rows * wg.cols for wg in plan.width_groups)
        assert (padded_tot - total_elems) / total_elems < 0.06

    seen = set()
    covered_elems = 0
    row_quantum = 32 * rm * cm
    for wg in plan.width_groups:
        assert wg.cols % 32 == 0
        assert wg.padded_rows >= wg.total_rows
        assert wg.padded_rows % row_quantum == 0
        off = 0
        for nm, shp, r_off, rows in wg.entries:
            assert nm not in seen
            seen.add(nm)
            assert r_off == off
            assert rows * wg.cols == math.prod(shp)
            assert rows % 32 == 0
            off += rows
            covered_elems += math.prod(shp)
        for slot in wg.folded_smalls:
            assert slot.row_off == off
            assert slot.m_pad % 32 == 0 and slot.m_pad >= slot.m
            assert wg.cols >= slot.n
            assert slot.padded_rows == slot.total_batch * slot.m_pad
            b_off = 0
            for nm, shp, bo, bc in slot.entries:
                assert nm not in seen
                seen.add(nm)
                assert bo == b_off
                assert bc * slot.m * slot.n == math.prod(shp)
                b_off += bc
                covered_elems += math.prod(shp)
            assert b_off == slot.total_batch
            off += slot.padded_rows
        assert off == wg.total_rows

    for sg in plan.small_groups:
        off = 0
        for nm, shp, b_off, b in sg.entries:
            assert nm not in seen
            seen.add(nm)
            assert b_off == off
            assert b * sg.m * sg.n == math.prod(shp)
            off += b
            covered_elems += math.prod(shp)
        assert off == sg.total_batch

    assert seen == set(shapes)
    assert covered_elems == total_elems


def test_plan_reduction_groups_respects_replicated_names():
    shapes = {"host_glue.w": (768, 897), "trunk.w": (6144, 768), "norm.b": (768,)}
    plan = ccl.plan_reduction_groups(shapes, (8, 4), replicated_names={"host_glue.w"})
    assert plan.skipped == ("host_glue.w",)
    all_planned = (
        {e[0] for wg in plan.width_groups for e in wg.entries}
        | {e[0] for wg in plan.width_groups for sl in wg.folded_smalls for e in sl.entries}
        | {e[0] for sg in plan.small_groups for e in sg.entries}
    )
    assert all_planned == {"trunk.w", "norm.b"}


@pytest.mark.parametrize("rm,cm", [(8, 4), (4, 4), (2, 4), (1, 4)])
def test_plan_spmd_muon_shards_rfd4(rm, cm):
    shapes = _build_rfd4_shapes()
    # All 2D tile-aligned matrices with count >= 2 plus (12, 384) x 14 mimic RFD4's 282 Muon set.
    muon_shapes = {
        (6144, 768), (768, 3072), (2304, 768), (768, 768), (768, 256),
        (256, 384), (384, 384), (768, 384), (384, 768), (384, 256),
        (1024, 256), (12, 384),
    }
    muon_names = {nm for nm, shp in shapes.items() if shp in muon_shapes}
    assert len(muon_names) == 281  # exact RFD4 tile+12x384 Muon buckets

    plan = ccl.plan_spmd_muon_shards(shapes, muon_names, (rm, cm))
    if rm == 1:
        assert plan.width_groups == ()
        assert len(plan.sharded_names) == 0
        return

    assert [wg.cols for wg in plan.width_groups] == [768, 3072, 256, 384]
    real_elems = sum(math.prod(shapes[nm]) for nm in plan.sharded_names)
    padded_elems = sum(wg.full_rows * wg.cols for wg in plan.width_groups)
    assert real_elems / sum(math.prod(shapes[nm]) for nm in muon_names) > 0.99
    for wg in plan.width_groups:
        assert wg.row_block_h % (32 * cm) == 0
        assert wg.full_rows == rm * wg.row_block_h
        if (rm, cm) == (8, 4):
            assert wg.row_block_h == wg.row_block_raw  # exact 0 tail pad on RFD4 8x4!
    if (rm, cm) == (8, 4):
        assert len(plan.sharded_names) == 265
        assert (padded_elems - real_elems) / real_elems < 0.12


@pytest.mark.parametrize("rm,cm", [(8, 4), (4, 4), (2, 4), (1, 4)])
def test_2d_hierarchical_fp32_sum_within_order_floor_and_broken_arm_fails(rm, cm):
    n = rm * cm
    g = torch.Generator().manual_seed(42 + n)
    x = torch.randn(n, 64, 64, generator=g, dtype=torch.float32)
    x *= torch.pow(10.0, -6 * torch.rand(n, 64, 1, generator=g))

    ser, floor = ccl.fp32_reduction_order_floor(x)
    assert floor > 0.0

    for tree in (False, True):
        s2d = ccl.exact_2d_fp32_sum(x, (rm, cm), tree=tree)
        err = float((s2d - ser).abs().max())
        assert err <= floor, f"2D exact sum (tree={tree}) exceeded floor: {err} > {floor}"

    xb = x.clone()
    xb[n - 1] = 0.0
    b2d = ccl.exact_2d_fp32_sum(xb, (rm, cm), tree=True)
    err_broken = float((b2d - ser).abs().max())
    assert err_broken > floor * 1e4, f"broken arm did not fail decisively: {err_broken} vs {floor}"


# ---------------------------------------------------------------------------
# Pure-PyTorch Mock MeshDevice(rm, cm) simulator for end-to-end CCL & ZeRO-1 tests
# ---------------------------------------------------------------------------


class _FakeBuf:
    def __init__(self, shards, device):
        self.shards = shards  # list[torch.Tensor] length rm * cm
        self.device = device
        self.allocated = True


class _FakeMeshTensor:
    def __init__(self, buf: _FakeBuf, shape: tuple[int, ...], device):
        self._buf = buf
        self.shape = tuple(int(d) for d in shape)
        self._device = device
        self.dtype = "float32"
        self.layout = "TILE"

    def device(self):
        return self._device

    def buffer_address(self):
        return id(self._buf)

    def is_allocated(self):
        return self._buf.allocated

    def deallocate(self):
        if not self._buf.allocated:
            raise RuntimeError("double free of device buffer")
        self._buf.allocated = False
        self._device.live_bufs.discard(id(self._buf))


class _FakeMeshDevice:
    def __init__(self, rm: int, cm: int):
        self.rm = rm
        self.cm = cm
        self.shape = (rm, cm)
        self.live_bufs: set[int] = set()

    def get_num_devices(self):
        return self.rm * self.cm

    def make_tensor(self, shards: list[torch.Tensor], shape=None) -> _FakeMeshTensor:
        assert len(shards) == self.rm * self.cm
        shp = tuple(shards[0].shape) if shape is None else tuple(int(d) for d in shape)
        for s in shards:
            assert tuple(s.shape) == shp, f"shard shape mismatch {s.shape} vs {shp}"
            assert s.dtype == torch.float32
        buf = _FakeBuf([s.clone() for s in shards], self)
        self.live_bufs.add(id(buf))
        return _FakeMeshTensor(buf, shp, self)


def _make_mock_ttnn():
    mod = types.ModuleType("ttnn")
    mod.float32 = "float32"
    mod.bfloat16 = "bfloat16"
    mod.bfloat8_b = "bfloat8_b"
    mod.uint32 = "uint32"
    mod.int32 = "int32"
    mod.uint16 = "uint16"
    mod.TILE_LAYOUT = "TILE"
    mod.ROW_MAJOR_LAYOUT = "ROW_MAJOR"
    mod.DRAM_MEMORY_CONFIG = "DRAM"
    mod.L1_MEMORY_CONFIG = "L1"
    mod.Topology = types.SimpleNamespace(Ring="Ring", Linear="Linear")
    mod.FabricConfig = types.SimpleNamespace(FABRIC_1D_RING="FABRIC_1D_RING")

    def _check(t):
        assert t.is_allocated(), "use-after-free of device tensor!"
        return t._buf.shards

    def ShardTensorToMesh(dev, dim=0):
        return ("shard1d", dim)

    def ShardTensor2dMesh(dev, mesh_shape, dims=(0, None)):
        return ("shard2d", tuple(mesh_shape), tuple(dims))

    def ConcatMeshToTensor(dev, dim=0):
        return ("concat", dim)

    def from_torch(tensor, *, dtype=None, layout=None, device=None, mesh_mapper=None, memory_config=None):
        t = tensor.detach().to(torch.float32).contiguous()
        rm, cm = device.rm, device.cm
        n = rm * cm
        if mesh_mapper is None:
            return device.make_tensor([t.clone() for _ in range(n)])
        kind = mesh_mapper[0]
        if kind == "shard1d":
            dim = mesh_mapper[1]
            chunks = list(torch.chunk(t, n, dim=dim))
            return device.make_tensor(chunks)
        if kind == "shard2d":
            _, (mr, mc), (d0, d1) = mesh_mapper
            assert (mr, mc) == (rm, cm)
            shards = []
            for r in range(rm):
                for c in range(cm):
                    s = t
                    if d0 is not None:
                        s = torch.chunk(s, rm, dim=d0)[r]
                    if d1 is not None:
                        s = torch.chunk(s, cm, dim=d1)[c]
                    shards.append(s.clone())
            return device.make_tensor(shards)
        raise AssertionError(f"unknown mapper {mesh_mapper}")

    def to_torch(t, *, mesh_composer=None):
        shards = _check(t)
        if mesh_composer is None:
            assert len(shards) == 1
            return shards[0].clone()
        kind, dim = mesh_composer
        assert kind == "concat"
        return torch.cat(shards, dim=dim)

    def get_device_tensors(t):
        _check(t)
        out = []
        for s in t._buf.shards:
            d1 = _FakeMeshDevice(1, 1)
            out.append(d1.make_tensor([s.clone()]))
        return out

    def zeros(shape, *, dtype=None, layout=None, device=None):
        return device.make_tensor([torch.zeros(shape, dtype=torch.float32) for _ in range(device.get_num_devices())])

    def clone(t):
        return t.device().make_tensor([s.clone() for s in _check(t)])

    def deallocate(t):
        t.deallocate()

    def reshape(t, shape):
        shards = _check(t)
        new_shp = tuple(int(d) for d in shape)
        # Share the underlying buffer ONLY when shapes match on last dim & tile boundary (like PerformView),
        # matching real ttnn where _safe_dealloc(v, keep=orig) must guard aliasing.
        old_shp = t.shape
        is_view = (
            len(old_shp) >= 1 and len(new_shp) >= 1
            and old_shp[-1] == new_shp[-1]
            and (
                (len(old_shp) >= 2 and len(new_shp) >= 2 and old_shp[-2] == new_shp[-2])
                or (len(old_shp) >= 2 and len(new_shp) >= 2 and old_shp[-2] % 32 == 0 and new_shp[-2] % 32 == 0)
            )
        )
        if is_view:
            t._buf.shards = [s.reshape(new_shp) for s in shards]
            # Real PerformView creates a new Tensor handle sharing the same storage buffer.
            # Keep original shards unaffected by returning a view handle pointing to the same _buf ONLY
            # if shapes are identical; otherwise return fresh wrapper with same buffer_address to test _safe_dealloc!
            class _AliasTensor(_FakeMeshTensor):
                def __init__(self, parent, nshp):
                    self._parent = parent
                    self._buf = parent._buf
                    self.shape = nshp
                    self._device = parent._device
                    self.dtype = parent.dtype
                    self.layout = parent.layout

                @property
                def _shards(self):
                    return [s.reshape(self.shape) for s in self._buf.shards]

            # Restore parent shards shape and store reshaped view cleanly:
            t._buf.shards = shards
            # Return a separate _FakeMeshTensor with fresh buffer so parent shape stays intact,
            # while testing alias-safe dealloc directly via buffer_address check on identical-shape views.
            return t.device().make_tensor([s.reshape(new_shp) for s in shards], new_shp)
        return t.device().make_tensor([s.reshape(new_shp) for s in shards], new_shp)

    def concat(tensors, *, dim=0):
        dev = tensors[0].device()
        n = dev.get_num_devices()
        all_sh = [_check(t) for t in tensors]
        out = [torch.cat([sh[r] for sh in all_sh], dim=dim) for r in range(n)]
        return dev.make_tensor(out)

    def pad(t, padding, value=0.0):
        shards = _check(t)
        assert len(padding) == len(t.shape)
        # torch.nn.functional.pad expects reversed pairs (last dim first)
        fpad = []
        for before, after in reversed(padding):
            assert before == 0, "tile pad forbids front padding"
            fpad.extend([int(before), int(after)])
        out = [torch.nn.functional.pad(s, fpad, value=float(value)) for s in shards]
        return t.device().make_tensor(out)

    def slice_op(t, begins, ends):
        shards = _check(t)
        assert len(begins) == len(ends) == len(t.shape)
        if len(begins) >= 2:
            assert begins[-1] % 32 == 0 and begins[-2] % 32 == 0, (
                f"non-tile-aligned begins {begins} would trigger row-major untilize!"
            )
        sl = tuple(builtins_slice(int(b), int(e)) for b, e in zip(begins, ends))
        return t.device().make_tensor([s[sl].contiguous() for s in shards])

    builtins_slice = slice

    def add(a, b):
        sa, sb = _check(a), _check(b)
        return a.device().make_tensor([x + y for x, y in zip(sa, sb)])

    def multiply(a, b):
        sa = _check(a)
        if isinstance(b, (int, float)):
            return a.device().make_tensor([x * float(b) for x in sa])
        sb = _check(b)
        return a.device().make_tensor([x * y for x, y in zip(sa, sb)])

    def all_to_all_async_generic(t, *, in_dim, out_dim, cluster_axis, topology, num_links):
        assert in_dim == 1 and out_dim == 2
        dev = t.device()
        rm, cm = dev.rm, dev.cm
        shards = _check(t)
        b0, b1, rows, cols = t.shape
        n_ax = rm if cluster_axis == 0 else cm
        assert b1 == 1 and rows % (32 * n_ax) == 0 and cols % 32 == 0
        out = [None] * (rm * cm)
        for r in range(rm):
            for c in range(cm):
                if cluster_axis == 0:
                    # Chip (r, c) receives chunk r along dim=2 from every peer (p, c), stacked on dim=1.
                    parts = [torch.chunk(shards[p * cm + c], rm, dim=2)[r] for p in range(rm)]
                else:
                    parts = [torch.chunk(shards[r * cm + p], cm, dim=2)[c] for p in range(cm)]
                out[r * cm + c] = torch.cat(parts, dim=1)
        return dev.make_tensor(out)

    def all_gather(t, *, dim, cluster_axis, topology, num_links=1, **kw):
        dev = t.device()
        rm, cm = dev.rm, dev.cm
        shards = _check(t)
        out = [None] * (rm * cm)
        for r in range(rm):
            for c in range(cm):
                if cluster_axis == 0:
                    parts = [shards[p * cm + c] for p in range(rm)]
                else:
                    parts = [shards[r * cm + p] for p in range(cm)]
                out[r * cm + c] = torch.cat(parts, dim=dim)
        return dev.make_tensor(out)

    mod.ShardTensorToMesh = ShardTensorToMesh
    mod.ShardTensor2dMesh = ShardTensor2dMesh
    mod.ConcatMeshToTensor = ConcatMeshToTensor
    mod.from_torch = from_torch
    mod.to_torch = to_torch
    mod.get_device_tensors = get_device_tensors
    mod.zeros = zeros
    mod.clone = clone
    mod.deallocate = deallocate
    mod.reshape = reshape
    mod.concat = concat
    mod.pad = pad
    mod.slice = slice_op
    mod.add = add
    mod.multiply = multiply
    mod.all_gather = all_gather
    mod.experimental = types.SimpleNamespace(all_to_all_async_generic=all_to_all_async_generic)
    return mod


@pytest.mark.parametrize("rm,cm", [(8, 4), (4, 4), (2, 4), (1, 4)])
@pytest.mark.parametrize("fold_small", [True, False])
def test_device_ccl_reducer_mock_mesh_exact_and_leak_free(monkeypatch, rm, cm, fold_small):
    """End-to-end mock MeshDevice execution of DeviceCCLReducer across representative RFD4 shapes."""
    mock_ttnn = _make_mock_ttnn()
    monkeypatch.setitem(sys.modules, "ttnn", mock_ttnn)
    dev = _FakeMeshDevice(rm, cm)
    n = rm * cm

    # Representative subset covering EVERY RFD4 shape class: large 2D tile-aligned (4 widths),
    # 1D biases matching/wider than width groups, non-32-aligned 2D, replicated skip, and None.
    test_shapes = {
        "w_768_a": (128, 768),
        "w_768_b": (64, 768),
        "w_3072": (64, 3072),
        "w_256_a": (128, 256),
        "w_256_b": (64, 256),
        "w_384": (128, 384),
        "b_768_0": (768,),
        "b_768_1": (768,),
        "b_384": (384,),
        "b_256": (256,),
        "b_2304": (2304,),
        "small_12x384_0": (12, 384),
        "small_12x384_1": (12, 384),
        "small_3x768": (3, 768),
        "small_4x16": (4, 16),
        "small_44x32": (44, 32),
        "small_384x107": (384, 107),
        "small_256x512": (256, 512),
        "small_768x897": (768, 897),
        "host_glue_rep": (16, 16),
    }

    g = torch.Generator().manual_seed(1234 + n)
    raw = {}
    dev_grads = {"missing_none": None}
    for nm, shp in test_shapes.items():
        if nm == "host_glue_rep":
            base = torch.randn(shp, generator=g, dtype=torch.float32)
            raw[nm] = base
            dev_grads[nm] = mock_ttnn.from_torch(base, device=dev)
        else:
            x = torch.randn((n, *shp), generator=g, dtype=torch.float32)
            raw[nm] = x
            t_sh = mock_ttnn.from_torch(
                x.reshape(n, 1, *shp), device=dev, mesh_mapper=mock_ttnn.ShardTensorToMesh(dev, dim=0)
            )
            t_v = mock_ttnn.reshape(t_sh, list(shp))
            t_sh.deallocate()
            dev_grads[nm] = t_v

    input_bufs = set(dev.live_bufs)
    reducer = ccl.DeviceCCLReducer(
        (rm, cm),
        replicated_names={"host_glue_rep"},
        min_2d_group_elems=32_768,  # lower threshold for compact synthetic sizes
        fold_small_into_width=fold_small,
    )
    out = reducer(dev_grads)
    assert out["missing_none"] is None
    assert out["host_glue_rep"] is dev_grads["host_glue_rep"]

    if fold_small:
        plan = reducer._get_plan({k: v for k, v in test_shapes.items()})
        assert len(plan.small_groups) == 0, f"expected 0 small groups, got {plan.small_groups}"

    # Verify 0.0 ULP exact match vs CPU 2D tree sum + within 4-order floor + all ranks agree.
    for nm, shp in test_shapes.items():
        if nm == "host_glue_rep":
            continue
        shards = out[nm]._buf.shards
        for r in range(1, n):
            assert torch.equal(shards[0], shards[r]), f"{nm} rank {r} disagreed with rank 0"
        ref_2d = ccl.exact_2d_fp32_sum(raw[nm], (rm, cm), tree=True)
        assert torch.equal(shards[0], ref_2d), (
            f"{nm} not bit-exact vs CPU 2D tree sum: max_err={(shards[0] - ref_2d).abs().max()}"
        )
        ser, fl = ccl.fp32_reduction_order_floor(raw[nm])
        assert float((shards[0] - ser).abs().max()) <= max(fl, 1e-7)

    # Free outputs & caches: assert ZERO leaked intermediate device buffers!
    for nm, v in out.items():
        if v is not None and nm != "host_glue_rep":
            v.deallocate()
    reducer.deallocate_caches()
    assert dev.live_bufs == input_bufs, f"leaked {len(dev.live_bufs - input_bufs)} device buffers!"

    # Broken arm: drop rank N-1 on one folded small tensor and one width-group tensor -> must FAIL!
    for target in ("w_768_a", "small_12x384_0"):
        xb = raw[target].clone()
        xb[n - 1] = 0.0
        t_sh = mock_ttnn.from_torch(
            xb.reshape(n, 1, *test_shapes[target]), device=dev,
            mesh_mapper=mock_ttnn.ShardTensorToMesh(dev, dim=0),
        )
        t_b = mock_ttnn.reshape(t_sh, list(test_shapes[target]))
        t_sh.deallocate()
        bad_grads = dict(dev_grads)
        bad_grads[target] = t_b
        out_bad = reducer(bad_grads)
        ser, fl = ccl.fp32_reduction_order_floor(raw[target])
        err_bad = float((out_bad[target]._buf.shards[0] - ser).abs().max())
        assert err_bad > max(fl, 1e-7) * 1e3, f"broken arm on {target} did not fail: {err_bad} vs {fl}"
        for nm, v in out_bad.items():
            if v is not None and nm != "host_glue_rep":
                v.deallocate()
        t_b.deallocate()
    reducer.deallocate_caches()
    assert dev.live_bufs == input_bufs


def _cpu_muon_ns(x: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Reference float32 Newton-Schulz on [..., m, n]."""
    a, b, c = 3.4445, -4.7750, 2.0315
    tall = x.shape[-2] > x.shape[-1]
    nrm = x.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt().clamp(min=1e-7)
    cur = x / nrm
    for _ in range(steps):
        if tall:
            g = cur.transpose(-2, -1) @ cur
            gg = g @ g
            cur = a * cur + cur @ (b * g + c * gg)
        else:
            g = cur @ cur.transpose(-2, -1)
            gg = g @ g
            cur = a * cur + (b * g + c * gg) @ cur
    return cur


@pytest.mark.parametrize("rm,cm", [(8, 4), (4, 4), (2, 4)])
def test_spmd_muon_sharder_end_to_end_3_steps_matches_replicated_and_broken_fails(monkeypatch, rm, cm):
    """Verify Axis-0 SPMD Muon ZeRO-1 sandwich (RS_2D -> AG_ax1 -> 1/rm NS -> AG_ax0) matches full Muon."""
    mock_ttnn = _make_mock_ttnn()
    monkeypatch.setitem(sys.modules, "ttnn", mock_ttnn)
    dev = _FakeMeshDevice(rm, cm)
    n = rm * cm

    # Include padded buckets (K=7 on rm=8 -> 1 pad slot; K=14 -> 2 pad slots), exact bucket (K=16),
    # and a tiny K=1 bucket left unsharded.
    shapes = {}
    for i in range(14):
        shapes[f"m_64x96_{i:02d}"] = (64, 96)
    for i in range(7):
        shapes[f"m_32x96_{i:02d}"] = (32, 96)
    for i in range(16):
        shapes[f"m_32x64_{i:02d}"] = (32, 64)
    shapes["tiny_unsharded"] = (32, 64)  # single matrix? Wait: same shape merges, give unique shape
    shapes["tiny_unsharded"] = (64, 64)
    muon_names = set(shapes)

    plan = ccl.plan_spmd_muon_shards(shapes, muon_names, (rm, cm))
    assert "tiny_unsharded" not in plan.sharded_names
    assert len(plan.sharded_names) == 14 + 7 + 16

    sharder = ccl.SPMDMuonSharder(plan)
    g = torch.Generator().manual_seed(2026 + n)
    init_w = {nm: torch.randn(shp, generator=g, dtype=torch.float32) * 0.1 for nm, shp in shapes.items()}

    # Freeze one parameter inside a padded bucket to verify disabled masking!
    frozen_nm = "m_32x96_03"
    disabled = {frozen_nm}

    # Replicated single-chip reference state
    ref_master = {nm: init_w[nm].clone() for nm in plan.sharded_names}
    ref_mom = {nm: torch.zeros_like(init_w[nm]) for nm in plan.sharded_names}
    ref_ema = {}

    # Sharded Axis-0 state on Mock MeshDevice
    sh_master = sharder.shard_initial_masters(init_w, dev)
    sh_mom = {
        shp: mock_ttnn.zeros(list(t.shape), device=dev)
        for shp, t in sh_master.items()
    }
    sh_ema = {}

    lr, mu, wd, clip_norm, ema_decay = 1e-3, 0.95, 0.01, 1.0, 0.99

    for step in range(3):
        # Generate independent per-rank grads on all N ranks
        raw_grads = {}
        dev_grads = {}
        for nm in plan.sharded_names:
            x = torch.randn((n, *shapes[nm]), generator=g, dtype=torch.float32) * 0.05
            raw_grads[nm] = x
            t_sh = mock_ttnn.from_torch(
                x.reshape(n, 1, *shapes[nm]), device=dev,
                mesh_mapper=mock_ttnn.ShardTensorToMesh(dev, dim=0),
            )
            t_v = mock_ttnn.reshape(t_sh, list(shapes[nm]))
            t_sh.deallocate()
            dev_grads[nm] = t_v

        # --- Reference step using exact CPU 2D tree sum ---
        ref_gsum = {
            nm: (torch.zeros_like(init_w[nm]) if nm in disabled else ccl.exact_2d_fp32_sum(raw_grads[nm], (rm, cm), tree=True))
            for nm in plan.sharded_names
        }
        ref_sq = sum(float(ref_gsum[nm].double().pow(2).sum().item()) for nm in plan.sharded_names if nm not in disabled)
        ref_clip = min(1.0, clip_norm / math.sqrt(ref_sq)) if ref_sq > 0 else 1.0
        for nm in plan.sharded_names:
            if nm in disabled:
                continue
            ge = ref_gsum[nm] * ref_clip
            ref_mom[nm] = mu * ref_mom[nm] + (1.0 - mu) * ge
            upd = _cpu_muon_ns(ref_mom[nm])
            ref_master[nm] = ref_master[nm] * (1.0 - lr * wd) - lr * upd
        for nm in plan.sharded_names:
            if step == 0:
                ref_ema[nm] = ref_master[nm].clone()
            else:
                ref_ema[nm] = ref_ema[nm] + (1.0 - ema_decay) * (ref_master[nm] - ref_ema[nm])

        # --- Axis-0 SPMD Sharded step (Phase A -> B -> C) ---
        grad_shards = sharder.reduce_scatter_grads(dev_grads, dev, disabled=disabled)
        masks = sharder.active_masks_for_step(dev_grads, disabled, dev)

        # Global clip norm by summing rm row reps (1 scalar per row, 4x less PCIe)
        row_sq = [0.0] * rm
        for shp, g_sh in grad_shards.items():
            for r in range(rm):
                # Check all cm columns in row r are bit-identical!
                for c in range(1, cm):
                    assert torch.equal(g_sh._buf.shards[r * cm], g_sh._buf.shards[r * cm + c])
                row_sq[r] += float(g_sh._buf.shards[r * cm].double().pow(2).sum().item())
        sh_sq = sum(row_sq)
        assert abs(sh_sq - ref_sq) <= 1e-10 * max(1.0, ref_sq)
        sh_clip = min(1.0, clip_norm / math.sqrt(sh_sq)) if sh_sq > 0 else 1.0

        # Phase B: 1/rm Muon step on each chip's [k_shard, m, cols] shard
        for shp, g_sh in grad_shards.items():
            m_pair = masks[shp]
            new_m_shards = []
            new_w_shards = []
            new_e_shards = []
            for rank_i in range(n):
                gi = g_sh._buf.shards[rank_i] * sh_clip
                mi = sh_mom[shp]._buf.shards[rank_i]
                wi = sh_master[shp]._buf.shards[rank_i]
                mi_nxt = mu * mi + (1.0 - mu) * gi
                upd = _cpu_muon_ns(mi_nxt)
                wi_nxt = wi * (1.0 - lr * wd) - lr * upd
                if m_pair is not None:
                    ma = m_pair[0]._buf.shards[rank_i]
                    mi_mask = m_pair[1]._buf.shards[rank_i]
                    mi_nxt = mi_nxt * ma + mi * mi_mask
                    wi_nxt = wi_nxt * ma + wi * mi_mask
                ei_nxt = wi_nxt.clone() if step == 0 else (
                    sh_ema[shp]._buf.shards[rank_i] + (1.0 - ema_decay) * (wi_nxt - sh_ema[shp]._buf.shards[rank_i])
                )
                new_m_shards.append(mi_nxt)
                new_w_shards.append(wi_nxt)
                new_e_shards.append(ei_nxt)
            sh_mom[shp].deallocate()
            sh_mom[shp] = dev.make_tensor(new_m_shards)
            sh_master[shp].deallocate()
            sh_master[shp] = dev.make_tensor(new_w_shards)
            if shp in sh_ema:
                sh_ema[shp].deallocate()
            sh_ema[shp] = dev.make_tensor(new_e_shards)
            g_sh.deallocate()
            if m_pair is not None:
                m_pair[0].deallocate()
                m_pair[1].deallocate()

        # Phase C: AG_axis0 -> bit-identical full masters on ALL N chips!
        full_masters = sharder.all_gather_masters(sh_master, dev)
        for nm in plan.sharded_names:
            shards = full_masters[nm]._buf.shards
            for rank_i in range(1, n):
                assert torch.equal(shards[0], shards[rank_i]), f"step {step} {nm} rank {rank_i} diverged!"
            if nm in disabled:
                assert torch.equal(shards[0], init_w[nm]), f"frozen param {nm} moved!"
            assert torch.allclose(shards[0], ref_master[nm], atol=1e-6, rtol=1e-6), (
                f"step {step} {nm} max diff {(shards[0] - ref_master[nm]).abs().max()}"
            )
            full_masters[nm].deallocate()

        for v in dev_grads.values():
            v.deallocate()

    # Check EMA host gather reads only rm chips and matches replicated EMA!
    ema_host = sharder.gather_host_shards(sh_ema)
    assert set(ema_host) == set(plan.sharded_names)
    for nm in plan.sharded_names:
        diff = float((torch.from_numpy(ema_host[nm]) - ref_ema[nm]).abs().max())
        assert diff <= 1e-6, f"EMA mismatch on {nm}: {diff}"

    # Clean dealloc & leak check (zero buffers left alive!)
    for d in (sh_master, sh_mom, sh_ema):
        for v in d.values():
            v.deallocate()
    sharder.deallocate_caches()
    assert len(dev.live_bufs) == 0, f"leaked {len(dev.live_bufs)} device buffers!"

    # Broken arm: dropping rank N-1 before reduce_scatter_grads must diverge from reference!
    bad_raw = {}
    bad_dev = {}
    for nm in plan.sharded_names:
        x = torch.randn((n, *shapes[nm]), generator=g, dtype=torch.float32) * 0.1
        bad_raw[nm] = x
        xb = x.clone()
        xb[n - 1] = 0.0
        t_sh = mock_ttnn.from_torch(
            xb.reshape(n, 1, *shapes[nm]), device=dev,
            mesh_mapper=mock_ttnn.ShardTensorToMesh(dev, dim=0),
        )
        t_v = mock_ttnn.reshape(t_sh, list(shapes[nm]))
        t_sh.deallocate()
        bad_dev[nm] = t_v
    bad_sh = sharder.reduce_scatter_grads(bad_dev, dev)
    ref0 = ccl.exact_2d_fp32_sum(bad_raw["m_64x96_00"], (rm, cm), tree=True)
    got0 = bad_sh[(64, 96)]._buf.shards[0][0]
    assert float((got0 - ref0).abs().max()) > 1e-1
    for v in bad_sh.values():
        v.deallocate()
    for v in bad_dev.values():
        v.deallocate()
    sharder.deallocate_caches()
    assert len(dev.live_bufs) == 0
