"""CPU controls. These doubles do not certify installed TTNN compatibility."""
import enum
import gc
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
import weakref

from perf.c10_generic_identity.observer import Capture, CaptureError, Encoder, sha


class CoreCoord:
    def __init__(self, x, y):
        self.x, self.y = x, y


class CoreRange:
    def __init__(self, start_coord, end_coord):
        self.start, self.end = start_coord, end_coord


class CoreRangeSet:
    def __init__(self, ranges):
        self._ranges = ranges

    def ranges(self):
        return self._ranges


class RuntimeArgsView:
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, x):
        class Column:
            def __getitem__(col, y):
                for c, values in self.pairs:
                    if (c.x, c.y) == (x, y):
                        return values
                raise IndexError('no entry')
        return Column()


class SourceType(enum.Enum):
    FILE_PATH = 0
    SOURCE_CODE = 1


class ReaderConfigDescriptor:
    pass


class ComputeConfigDescriptor:
    def __init__(self):
        self.math_fidelity = 4
        self.math_approx_mode = False
        self.fp32_dest_acc_en = False
        self.dst_full_sync_en = False
        self.unpack_to_dest_mode = []
        self.bfp8_pack_precise = False


class KernelDescriptor:
    SourceType = SourceType

    def __init__(self, source):
        self.kernel_source = source
        self.source_type = SourceType.FILE_PATH
        self.core_ranges = CoreRangeSet([CoreRange(CoreCoord(0, 0), CoreCoord(1, 1))])
        self.compile_time_args = [2, 3, 5]
        self.named_compile_time_args = [('tiles', 3)]
        self.defines = [('ACTIVE', '1')]
        self.common_runtime_args = [80]
        self.runtime_args = RuntimeArgsView([(CoreCoord(0, 0), [16, 32]), (CoreCoord(1, 1), [])])
        self.config = ComputeConfigDescriptor()
        # opt_level intentionally absent, like audited nanobind getter.


class ProgramDescriptor:
    def __init__(self, kernels):
        self.kernels = kernels
        self.semaphores = []
        self.cbs = []
        # custom_program_hash intentionally absent.


class Tensor:
    def __init__(self, shape=(3, 5)):
        self.shape = shape
        self.padded_shape = (32, 32)
        self.dtype = 'bf16'
        self.layout = 'tile'

    def __eq__(self, other):
        raise AssertionError('tensor equality must never run')

    def __repr__(self):
        raise AssertionError('tensor repr must never run')

    def cpu(self):
        raise AssertionError('readback must never run')

    def to_torch(self):
        raise AssertionError('readback must never run')

    def device(self):
        raise AssertionError('device lookup must never run')

    def buffer_address(self):
        raise AssertionError('address lookup must never run')


def binding(fn):
    module = types.SimpleNamespace(generic_op=fn, __version__='CPU-double-only')
    for cls in (CoreCoord, CoreRange, CoreRangeSet, RuntimeArgsView, ReaderConfigDescriptor,
                ComputeConfigDescriptor, KernelDescriptor, ProgramDescriptor, Tensor):
        setattr(module, cls.__name__, cls)
    # A deliberate invariant for paired runtime controls, not the C++ hash algorithm.
    module.compute_program_descriptor_hash = lambda pd: 12345
    return module


def load_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class ObserverControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'kernel.cpp'
        self.source.write_bytes(b'void kernel_main() {}\n')
        self.kernel = KernelDescriptor(str(self.source))
        self.pd = ProgramDescriptor([self.kernel])
        self.io = [Tensor(), Tensor()]
        self.output = self.root / 'capture.jsonl'

    def tearDown(self):
        self.temp.cleanup()

    def capture(self, fn=None):
        self.module = binding(fn if fn is not None else lambda *a, **kw: self.io[-1])
        return Capture(self.module, self.output)

    def calls(self, path=None):
        return [r for r in load_records(path or self.output) if r['kind'] == 'call']

    def fields(self, call=0):
        return self.calls()[call]['descriptor']['fields']['kernels'][0]['fields']

    def test_paired_arguments_call_count_and_return_identity(self):
        calls = []
        result = object()
        extra = object()
        def original(*args, **kwargs):
            calls.append((args, kwargs))
            return result
        cap = self.capture(original)
        self.assertIs(original(self.io, self.pd, extra=extra), result)
        with cap:
            self.assertIs(self.module.generic_op(self.io, self.pd, extra=extra), result)
        self.assertEqual(len(calls), 2)
        for args, kw in calls:
            self.assertIs(args[0], self.io)
            self.assertIs(args[1], self.pd)
            self.assertIs(kw['extra'], extra)
        self.assertIs(self.module.generic_op, original)
        self.assertEqual(self.fields()['compile_time_args'], [2, 3, 5])
        self.assertEqual(self.fields()['defines'], [['ACTIVE', '1']])

    def test_keyword_call_is_not_rewritten(self):
        seen = []
        def fn(*args, **kw):
            self.assertFalse(args)
            seen.append(kw)
            return self.io[-1]
        cap = self.capture(fn)
        with cap:
            self.module.generic_op(io_tensors=self.io, program_descriptor=self.pd)
        self.assertIs(seen[0]['program_descriptor'], self.pd)
        self.assertIs(seen[0]['io_tensors'], self.io)

    def test_original_exception_object_preserved(self):
        error = ValueError('original')
        count = []
        def fn(*args, **kw):
            count.append(1)
            raise error
        cap = self.capture(fn)
        for callable_ in (fn,):
            with self.assertRaises(ValueError) as raised:
                callable_(self.io, self.pd)
            self.assertIs(raised.exception, error)
        with self.assertRaises(ValueError) as raised:
            with cap:
                self.module.generic_op(self.io, self.pd)
        self.assertIs(raised.exception, error)
        self.assertEqual(len(count), 2)
        self.assertIs(self.module.generic_op, fn)

    def test_reused_descriptor_changed_runtime_and_common_args(self):
        cap = self.capture()
        with cap:
            self.module.generic_op(self.io, self.pd)
            self.kernel.runtime_args.pairs[0][1][0] = 4096
            self.kernel.common_runtime_args[0] = 8192
            self.module.generic_op(self.io, self.pd)
        a, b = self.calls()
        self.assertEqual(a['binding_cache_hash'], b['binding_cache_hash'])
        self.assertEqual(a['descriptor_object_id'], b['descriptor_object_id'])
        self.assertNotEqual(a['execution_sha256'], b['execution_sha256'])
        self.assertEqual(self.fields(0)['runtime_args']['entries'][0]['values'], [16, 32])
        self.assertEqual(self.fields(1)['runtime_args']['entries'][0]['values'], [4096, 32])
        self.assertEqual(self.fields(0)['common_runtime_args'], [80])
        self.assertEqual(self.fields(1)['common_runtime_args'], [8192])

    def test_view_sparse_empty_and_unenumerable_entries(self):
        encoder = Encoder(binding(lambda: None))
        rt = encoder.runtime(self.kernel)
        self.assertEqual(rt['binding_entry_count'], 2)
        self.assertEqual([e['values'] for e in rt['entries']], [[16, 32], []])
        self.kernel.runtime_args.pairs.append((CoreCoord(9, 9), [99]))
        self.assertIn('unavailable', encoder.runtime(self.kernel))
        self.kernel.runtime_args.pairs[-1] = (CoreCoord(0, 0), [123])
        self.assertIn('unavailable', encoder.runtime(self.kernel))

    def test_native_only_view_and_vector_exports(self):
        module = binding(lambda: None)
        del module.RuntimeArgsView
        class VectorUInt32(list):
            pass
        module._ttnn = types.SimpleNamespace(program_descriptor=types.SimpleNamespace(
            RuntimeArgsView=RuntimeArgsView, VectorUInt32=VectorUInt32))
        self.kernel.runtime_args.pairs[0] = (CoreCoord(0, 0), VectorUInt32([16, 32]))
        encoder = Encoder(module)
        out = encoder.runtime(self.kernel)
        self.assertNotIn('unavailable', out)
        self.assertEqual(out['entries'][0]['values'], [16, 32])
        self.assertEqual(encoder.value(VectorUInt32([4, 5])), [4, 5])

    def test_list_runtime_preserves_storage_order(self):
        self.kernel.runtime_args = [(CoreCoord(1, 1), [5]), (CoreCoord(0, 0), [7])]
        out = Encoder(binding(lambda: None)).runtime(self.kernel)
        self.assertEqual([e['core'] for e in out['entries']], [[1, 1], [0, 0]])

    def test_changed_source_same_path_size_and_mtime(self):
        cap = self.capture()
        old = self.source.stat()
        first = self.source.read_bytes()
        with cap:
            self.module.generic_op(self.io, self.pd)
            self.source.write_bytes(first.replace(b'void', b'VOID'))
            os.utime(self.source, ns=(old.st_atime_ns, old.st_mtime_ns))
            self.module.generic_op(self.io, self.pd)
        a, b = self.fields(0), self.fields(1)
        self.assertNotEqual(a['source_identity']['sha256'], b['source_identity']['sha256'])
        self.assertEqual(a['source_identity']['sha256'], sha(first))
        sources = [r for r in load_records(self.output) if r['kind'] == 'source']
        self.assertEqual(len({r['sha256'] for r in sources}), len(sources))

    def test_inline_source_identity(self):
        self.kernel.source_type = SourceType.SOURCE_CODE
        self.kernel.kernel_source = 'void kernel_main() { /* inline */ }'
        with self.capture():
            self.module.generic_op(self.io, self.pd)
        self.assertEqual(self.fields()['source_identity']['sha256'], sha(self.kernel.kernel_source.encode()))

    def test_unknown_kernel_does_not_infer_roles_from_rank(self):
        self.io = [Tensor((1, 2, 32, 64)), Tensor((1, 2, 32, 64)), Tensor((32, 64))]
        with self.capture():
            self.module.generic_op(self.io, self.pd)
        self.assertEqual([o['role'] for o in self.calls()[0]['operands']], ['unknown'] * 3)
        self.assertIn('unavailable', self.calls()[0]['arithmetic'])

    def test_missing_metadata_source_and_binding_fields_explicit(self):
        del self.io[0].shape
        self.source.unlink()
        del self.kernel.defines
        with self.capture():
            self.module.generic_op(self.io, self.pd)
        self.assertIn('unavailable', self.fields()['source_identity'])
        self.assertIn('unavailable', self.fields()['defines'])
        self.assertIn('unavailable', self.fields()['opt_level'])
        self.assertIn('unavailable', self.calls()[0]['operands'][0]['shape'])

    def test_no_arbitrary_repr_or_getter(self):
        class Opaque:
            def __repr__(self):
                raise AssertionError('repr')
            @property
            def shape(self):
                raise AssertionError('shape')
        self.kernel.config = Opaque()
        self.io[0] = Opaque()
        with self.capture():
            self.module.generic_op(self.io, self.pd)
        self.assertIn('unavailable', self.fields()['config'])
        self.assertIn('unavailable', self.calls()[0]['operands'][0])

    def test_observer_failure_visible_after_call_and_restoration(self):
        count = []
        result = object()
        cap = self.capture(lambda *a, **kw: count.append(1) or result)
        original = self.module.generic_op
        def fail(*a):
            raise RuntimeError('observer')
        cap.snapshot = fail
        with self.assertRaises(CaptureError):
            with cap:
                self.assertIs(self.module.generic_op(self.io, self.pd), result)
        self.assertEqual(count, [1])
        self.assertIs(self.module.generic_op, original)
        footer = load_records(self.output)[-1]
        self.assertFalse(footer['observation_ok'])

    def test_observer_failure_does_not_replace_original_exception(self):
        error = KeyError('original')
        def fn(*a, **kw):
            raise error
        cap = self.capture(fn)
        cap.snapshot = lambda *a: 1 / 0
        with self.assertRaises(KeyError) as raised:
            with cap:
                self.module.generic_op(self.io, self.pd)
        self.assertIs(raised.exception, error)
        self.assertTrue(cap.errors)
        self.assertIs(self.module.generic_op, fn)

    def test_sink_failure_restores_and_preserves_dispatch(self):
        cap = self.capture()
        original = self.module.generic_op
        with self.assertRaises(CaptureError):
            with cap:
                cap.stream.close()
                self.assertIs(self.module.generic_op(self.io, self.pd), self.io[-1])
        self.assertIs(self.module.generic_op, original)
        self.assertTrue(cap.errors)

    def test_nested_contexts_call_original_once_and_restore_on_error(self):
        count = []
        cap = self.capture(lambda *a, **kw: count.append(1) or self.io[-1])
        original = self.module.generic_op
        inner_path = self.root / 'inner.jsonl'
        with cap:
            outer = self.module.generic_op
            with self.assertRaisesRegex(ValueError, 'body'):
                with Capture(self.module, inner_path):
                    self.module.generic_op(self.io, self.pd)
                    raise ValueError('body')
            self.assertIs(self.module.generic_op, outer)
            self.module.generic_op(self.io, self.pd)
        self.assertEqual(len(count), 2)
        self.assertEqual(len(self.calls()), 2)
        self.assertEqual(len(self.calls(inner_path)), 1)
        self.assertIs(self.module.generic_op, original)

    def test_no_tensor_references_retained(self):
        cap = self.capture(lambda *a, **kw: None)
        tensor = Tensor()
        ref = weakref.ref(tensor)
        with cap:
            self.module.generic_op([tensor, tensor], self.pd)
            del tensor
            gc.collect()
            self.assertIsNone(ref())
        self.assertEqual(self.calls()[0]['operands'][1]['same_object_as'], 0)

    def test_generator_not_consumed(self):
        read = []
        def sequence():
            read.append(1)
            yield self.io[0]
        gen = sequence()
        seen = []
        cap = self.capture(lambda *a, **kw: seen.append(a[0]))
        with cap:
            self.module.generic_op(gen, self.pd)
        self.assertFalse(read)
        self.assertIs(seen[0], gen)

    def test_unsupported_descriptor_passthrough(self):
        pd = object()
        with self.capture():
            self.module.generic_op(self.io, pd)
        self.assertIn('unavailable', self.calls()[0])

    def test_audited_loaded_matmul_wrapper_roles(self):
        file = Path(__file__).resolve().parents[2] / 'tt_bio/mm_generic.py'
        code = compile(file.read_bytes(), str(file), 'exec', dont_inherit=True)
        wrapper_code = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == 'generic_minimal_matmul')
        # Execute only the already-audited wrapper, with its helpers replaced by CPU controls.
        for i, tensor in enumerate(self.io):
            tensor.buffer_address = lambda i=i: i + 16
        out = Tensor()
        out.buffer_address = lambda: 64
        entry = {'pd': self.pd, 'addrs': (16, 17, (64,))}
        cap = self.capture()
        wrapper = types.FunctionType(wrapper_code, {'ttnn': self.module, '_key': lambda *a: 0, '_CACHE': {0: entry}})
        with cap:
            result = wrapper(None, self.io[0], self.io[1], [out], None, None, (), None, None, None, None)
        self.assertIs(result, out)
        self.assertEqual([o['role'] for o in self.calls()[0]['operands']], ['left', 'right', 'output'])
        self.assertIn('role_basis', self.calls()[0]['caller'])

    def test_audited_sdpa_roles_plain_fused_and_gated(self):
        file = Path(__file__).resolve().parents[2] / 'tt_bio/sdpa_generic.py'
        code = compile(file.read_bytes(), str(file), 'exec', dont_inherit=True)
        wrapper_code = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == 'sdpa')
        for mode in ('plain', 'fused', 'gated'):
            with self.subTest(mode=mode):
                ts = [Tensor((1, 2, 32, 64)) for _ in range(8)]
                for i, tensor in enumerate(ts):
                    tensor.buffer_address = lambda i=i: i + 16
                q, k, v, mask, out, x, w, g = ts
                entry = {'pd': self.pd, 'addrs': tuple(range(16, 21)),
                         'fuse': (0, 0, 21, 22), 'gate': 23}
                class Cache:
                    def get(self, key):
                        return entry
                cap = self.capture()
                cap.output = self.root / (mode + '.jsonl')
                wrapper = types.FunctionType(wrapper_code, {'ttnn': self.module, 'os': os, '_CACHE': Cache()})
                kw = {'fuse_qkv': (x, w, 1)} if mode == 'fused' else {'gate': (g, 1)} if mode == 'gated' else {}
                with cap:
                    self.assertIs(wrapper(None, q, k, v, mask, out, 32, 32, (1, 1), (), 1.0, **kw), out)
                expected = ['projection_input', 'projection_weight', 'mask', 'output'] if mode == 'fused' else ['query', 'key', 'value', 'mask', 'output']
                if mode == 'gated':
                    expected += ['gate']
                self.assertEqual([o['role'] for o in self.calls(cap.output)[0]['operands']], expected)

    def test_audited_reblock_roles_forward_back_and_gated(self):
        file = Path(__file__).resolve().parents[2] / 'tt_bio/reblock_permute.py'
        code = compile(file.read_bytes(), str(file), 'exec', dont_inherit=True)
        for name in ('reblock_permute', 'reblock_permute_back', 'reblock_permute_gated'):
            with self.subTest(name=name):
                x, out = Tensor((1, 32, 32, 32)), Tensor((1, 32, 32, 32))
                x.buffer_address = lambda: 16
                out.buffer_address = lambda: 32
                self.pd.kernels = [self.kernel, KernelDescriptor(str(self.source))]
                entry = {'pd': self.pd}
                cap = self.capture()
                cap.output = self.root / (name + '.jsonl')
                self.module.Shape = lambda shape: shape
                self.module.TILE_LAYOUT = 'tile'
                self.module.allocate_tensor_on_device = lambda *a: out
                ns = {'ttnn': self.module, '_DTYPE': 'bf16', 'ADDR_WRITE_MODE': 'in_place',
                      '_prepare': lambda *a: entry, '_prepare_back': lambda *a: entry,
                      '_prepare_gated': lambda *a: entry, 'STATS': [0], 'STATS_BACK': [0],
                      'STATS_GATED': [0], 'TILE_H': 32, 'TILE_W': 32, 'GATE_FIDELITY': 4,
                      'GATE_FP32_ACC': False}
                wrapper_code = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == name)
                wrapper = types.FunctionType(wrapper_code, ns)
                with cap:
                    if name.endswith('gated'):
                        wrapper(x, 0, 32, 32, None, object(), out, 0)
                    else:
                        wrapper(x, object(), object())
                expected = ['projection_and_gate' if name.endswith('gated') else 'input', 'output']
                self.assertEqual([o['role'] for o in self.calls(cap.output)[0]['operands']], expected)

    def test_loaded_code_mismatch_keeps_roles_unknown(self):
        file = Path(__file__).resolve().parents[2] / 'tt_bio/mm_generic.py'
        code = compile(file.read_bytes(), str(file), 'exec', dont_inherit=True)
        original = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == 'generic_minimal_matmul')
        changed = original.replace(co_consts=original.co_consts + ('unapproved loaded constant',))
        for i, tensor in enumerate(self.io):
            tensor.buffer_address = lambda i=i: i + 16
        cap = self.capture()
        entry = {'pd': self.pd, 'addrs': (16, 17, (17,))}
        wrapper = types.FunctionType(changed, {'ttnn': self.module, '_key': lambda *a: 0, '_CACHE': {0: entry}})
        with cap:
            wrapper(None, self.io[0], self.io[1], [self.io[1]], None, None, (), None, None, None, None)
        self.assertEqual([o['role'] for o in self.calls()[0]['operands']], ['unknown'] * 3)
        self.assertIn('unavailable', self.calls()[0]['caller'])

    def test_cross_thread_dispatch_is_visible_but_not_suppressed(self):
        import threading
        seen = []
        cap = self.capture(lambda *a: seen.append(a[0]))
        with self.assertRaises(CaptureError):
            with cap:
                thread = threading.Thread(target=self.module.generic_op, args=(self.io, self.pd))
                thread.start()
                thread.join()
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0], self.io)
        self.assertTrue(cap.errors)

    def test_special_source_file_is_not_opened(self):
        fifo = self.root / 'fifo.cpp'
        os.mkfifo(fifo)
        self.kernel.kernel_source = str(fifo)
        with self.capture():
            self.module.generic_op(self.io, self.pd)
        self.assertEqual(self.fields()['source_identity']['unavailable'], 'not a regular source file')

    def test_new_source_with_same_wrapper_name_not_trusted(self):
        cap = self.capture()
        source = self.root / 'unapproved.py'
        source.write_text('def generic_minimal_matmul(io, pd):\n    return ttnn.generic_op(io, pd)\n')
        ns = {'ttnn': self.module}
        exec(compile(source.read_bytes(), str(source), 'exec'), ns)
        with cap:
            ns['generic_minimal_matmul'](self.io, self.pd)
        self.assertEqual([o['role'] for o in self.calls()[0]['operands']], ['unknown'] * 2)

    def test_install_failure_never_patches(self):
        cap = self.capture()
        original = self.module.generic_op
        cap.provenance = lambda: 1 / 0
        with self.assertRaises(ZeroDivisionError):
            cap.__enter__()
        self.assertIs(self.module.generic_op, original)
        self.assertTrue(cap.stream.closed)

    def test_file_collision_never_overwrites_or_patches(self):
        cap = self.capture()
        original = self.module.generic_op
        self.output.write_text('keep')
        with self.assertRaises(FileExistsError):
            cap.__enter__()
        self.assertEqual(self.output.read_text(), 'keep')
        self.assertIs(self.module.generic_op, original)


if __name__ == '__main__':
    unittest.main()
