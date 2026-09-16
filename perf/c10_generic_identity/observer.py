"""Opt-in, host metadata only. Importing this module does not import ttnn."""
from __future__ import annotations

import base64
import enum
import hashlib
import json
import marshal
import os
from pathlib import Path
import stat
import sys
import threading
import types
import uuid

SCHEMA = 1
_FIELDS = {
    'ProgramDescriptor': ('kernels', 'semaphores', 'cbs', 'custom_program_hash'),
    'KernelDescriptor': ('kernel_source', 'source_type', 'core_ranges', 'compile_time_args',
                         'named_compile_time_args', 'defines', 'common_runtime_args',
                         'config', 'opt_level'),
    'ComputeConfigDescriptor': ('math_fidelity', 'math_approx_mode', 'fp32_dest_acc_en',
                                'dst_full_sync_en', 'unpack_to_dest_mode', 'bfp8_pack_precise'),
    'DataMovementConfigDescriptor': ('processor', 'noc', 'noc_mode'),
    'ReaderConfigDescriptor': (), 'WriterConfigDescriptor': (),
    'CBDescriptor': ('total_size', 'core_ranges', 'format_descriptors',
                     'remote_format_descriptors', 'address_offset'),
    'CBFormatDescriptor': ('buffer_index', 'data_format_as_uint8', 'page_size', 'tile'),
    'TileDescriptor': ('height', 'width', 'transpose'),
    'SemaphoreDescriptor': ('id', 'core_type', 'core_ranges', 'initial_value'),
    'CoreCoord': ('x', 'y'), 'CoreRange': ('start_coord', 'end_coord'),
}
_VECTORS = ('VectorUInt32', 'VectorUnpackToDestMode')
_GAPS = ['transitive includes and JIT compiler flags are not captured automatically',
         'observed files are not proof of the binary executed on a cache hit',
         'machine instructions, tensor contents, physical traffic and cycles are unmeasured',
         'trace replay and pre-bound aliases bypassing ttnn.generic_op are not observed',
         'device/graph/profiler call IDs are unavailable; sequence is host observation order']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def typename(obj):
    return type(obj).__module__ + '.' + type(obj).__qualname__


def missing(reason, obj=None):
    out = {'unavailable': reason}
    if obj is not None:
        out['type'] = typename(obj)
    return out


def unavailable_paths(value, path=''):
    if isinstance(value, dict):
        if 'unavailable' in value:
            yield path or '/'
        for key, child in value.items():
            yield from unavailable_paths(child, path + '/' + key)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from unavailable_paths(child, path + '/' + str(i))


def code_hash(code):
    """Ignore deployment path; retain bytecode, constants and line information."""
    consts = tuple(normalize_code(c) if isinstance(c, types.CodeType) else c
                   for c in code.co_consts)
    return sha(marshal.dumps(code.replace(co_filename='', co_consts=consts)))


def normalize_code(code):
    return code.replace(co_filename='', co_consts=tuple(
        normalize_code(c) if isinstance(c, types.CodeType) else c for c in code.co_consts))


class CaptureError(RuntimeError):
    pass


def binding_class(binding, name):
    cls = getattr(binding, name, None)
    if cls is None:
        native = getattr(binding, '_ttnn', None)
        descriptors = getattr(native, 'program_descriptor', None)
        cls = getattr(descriptors, name, None)
    return cls


class Encoder:
    """Allowlisted binding getters only. Never repr, tensor conversion or device queries."""
    def __init__(self, binding):
        self.binding = binding
        self.classes = {binding_class(binding, n): n for n in _FIELDS if binding_class(binding, n) is not None}
        self.vectors = tuple(binding_class(binding, n) for n in _VECTORS if binding_class(binding, n) is not None)

    def field(self, obj, name):
        try:
            return self.value(getattr(obj, name))
        except Exception as exc:
            return missing('getter failed: ' + type(exc).__name__)

    def value(self, obj, depth=0):
        if depth > 30:
            return missing('nesting limit')
        if obj is None or type(obj) in (bool, int, str):
            return obj
        if type(obj) is float:
            import math
            return obj if math.isfinite(obj) else missing('non-finite float')
        if isinstance(obj, enum.Enum):
            return {'enum': typename(obj), 'name': obj.name, 'value': self.value(obj.value)}
        if type(obj) in (list, tuple) or (self.vectors and isinstance(obj, self.vectors)):
            return [self.value(x, depth + 1) for x in obj]
        if type(obj) is dict:
            if not all(type(k) is str for k in obj):
                return missing('non-string dictionary keys')
            return {k: self.value(v, depth + 1) for k, v in obj.items()}
        if type(obj) in self.classes:
            name = self.classes[type(obj)]
            fields = {n: self.field(obj, n) for n in _FIELDS[name]}
            if name == 'CBDescriptor':
                for n in ('has_buffer', 'has_global_circular_buffer'):
                    try:
                        fields[n] = self.value(getattr(obj, n)())
                    except Exception as exc:
                        fields[n] = missing('getter failed: ' + type(exc).__name__)
                fields['buffer_address'] = missing('pointer/address access deliberately omitted')
            return {'type': name, 'fields': fields}
        core_set = getattr(self.binding, 'CoreRangeSet', None)
        if core_set is not None and type(obj) is core_set:
            return {'type': 'CoreRangeSet', 'ranges': self.value(list(obj.ranges()))}
        return missing('opaque object; no serializer', obj)

    def runtime(self, kernel):
        try:
            rt = kernel.runtime_args
            if type(rt) in (list, tuple):
                entries = [(int(c.x), int(c.y), self.value(v)) for c, v in rt]
                expected = len(rt)
                method = 'ordered pairs'
            else:
                view = binding_class(self.binding, 'RuntimeArgsView')
                if view is None or type(rt) is not view:
                    return missing('unsupported runtime argument container', rt)
                expected = len(rt)
                coords = set()
                for r in kernel.core_ranges.ranges():
                    sx, sy, ex, ey = int(r.start_coord.x), int(r.start_coord.y), int(r.end_coord.x), int(r.end_coord.y)
                    if not (0 <= sx <= ex < 1024 and 0 <= sy <= ey < 1024):
                        return missing('core range outside observation bound')
                    for x in range(sx, ex + 1):
                        for y in range(sy, ey + 1):
                            coords.add((x, y))
                entries = []
                for x, y in sorted(coords):
                    try:
                        values = rt[x][y]
                    except IndexError:
                        continue  # C++ out_of_range: no entry, distinct from an empty entry.
                    entries.append((x, y, self.value(values)))
                method = 'coordinate view; storage order unavailable'
            result = {'schema': 'core(x,y) -> positional uint32 values; meanings unknown',
                      'method': method, 'binding_entry_count': expected,
                      'entries': [{'core': [x, y], 'values': v, 'length': len(v) if isinstance(v, list) else None}
                                  for x, y, v in entries]}
            if len(entries) != expected:
                result['unavailable'] = 'entry count mismatch; duplicates or out-of-range cores cannot be enumerated'
            return result
        except Exception as exc:
            return missing('runtime getter failed: ' + type(exc).__name__)


class Sources:
    """Re-read and hash file bytes on each call; deduplicate payloads by content only."""
    def __init__(self, emit):
        self.emit = emit
        self.seen = set()

    def bytes(self, data):
        h = sha(data)
        if h not in self.seen:
            self.emit({'kind': 'source', 'sha256': h, 'encoding': 'base64',
                       'bytes': len(data), 'content': base64.b64encode(data).decode('ascii')})
            self.seen.add(h)
        return h

    def file(self, path, embed=True):
        try:
            p = Path(path).resolve(strict=True)
            # Reject devices/FIFOs before reading, and recheck the opened descriptor.
            if not stat.S_ISREG(p.stat().st_mode):
                return missing('not a regular source file')
            fd = os.open(p, os.O_RDONLY | os.O_NONBLOCK)
            with os.fdopen(fd, 'rb') as f:
                before = os.fstat(f.fileno())
                if not stat.S_ISREG(before.st_mode):
                    return missing('not a regular source file')
                if embed and before.st_size > 16 * 1024 * 1024:
                    return missing('source exceeds 16 MiB capture limit')
                if embed:
                    data = f.read(16 * 1024 * 1024 + 1)
                    if len(data) > 16 * 1024 * 1024:
                        return missing('source exceeds 16 MiB capture limit')
                    h = self.bytes(data)
                else:
                    hasher = hashlib.sha256()
                    for chunk in iter(lambda: f.read(1024 * 1024), b''):
                        hasher.update(chunk)
                    h = hasher.hexdigest()
                after = os.fstat(f.fileno())
            out = {'path': str(p), 'sha256': h, 'size': after.st_size,
                   'basis': 'file bytes at observation, not compiled binary identity'}
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                out['unavailable'] = 'source changed during read'
            return out
        except Exception as exc:
            return missing('file read failed: ' + type(exc).__name__)


class Capture:
    """Context manager patching only the supplied module's generic_op attribute.

    Use a single dispatch thread. Nested contexts in that thread each observe calls once.
    Observation errors leave calls untouched, then raise CaptureError on clean context exit.
    """
    def __init__(self, binding, output, *, provenance_files=(), config=None):
        self.binding = binding
        self.output = Path(output)
        self.provenance_files = tuple(provenance_files)
        self.config = config
        self.encoder = Encoder(binding)
        self.errors = []
        self.sequence = 0
        self.incomplete_calls = 0
        self.session = str(uuid.uuid4())
        self.active = False
        self.sources = Sources(self.emit)
        self._compiled = {}
        self.role_manifest = json.loads(Path(__file__).with_name('roles.json').read_text())

    def emit(self, record):
        self.stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + '\n')
        self.stream.flush()

    def error(self, stage, exc):
        # Do not retain exceptions/tracebacks: those keep dispatch tensors alive.
        item = {'stage': stage, 'exception_type': typename(exc)}
        self.errors.append(item)
        try:
            os.write(2, ('C10 identity observer error: ' + stage + ': ' + typename(exc) + '\n').encode())
        except OSError:
            pass

    def provenance(self):
        paths = list(self.provenance_files)
        for name in ('ttnn', 'ttnn._ttnn'):
            module = sys.modules.get(name)
            if module is not None and isinstance(getattr(module, '__file__', None), str):
                paths.append(module.__file__)
        return {'python': sys.version, 'executable': self.sources.file(sys.executable, embed=False),
                'binding_version': self.encoder.field(self.binding, '__version__'),
                'files': [self.sources.file(p, embed=False) for p in dict.fromkeys(paths)],
                'explicit_config': self.encoder.value(self.config),
                'environment': {k: v for k, v in os.environ.items()
                                if k.startswith('TT_BIO_') or k in (
                                    'TT_METAL_HOME', 'TT_METAL_RUNTIME_ROOT', 'TTNN_CONFIG_PATH',
                                    'TTNN_CONFIG_OVERRIDES', 'TT_METAL_CACHE_DIR', 'TT_METAL_DEVICE_PROFILER')},
                'unavailable': 'loaded ELF dependencies, effective JIT flags and source-to-build linkage not established'}

    def __enter__(self):
        if self.active:
            raise CaptureError('a Capture instance cannot be entered twice')
        self.stream = self.output.open('x', encoding='utf8')
        try:
            self.emit({'kind': 'header', 'schema': SCHEMA, 'session': self.session,
                       'provenance': self.provenance(), 'limits': _GAPS,
                       'performance_measurement': False})
            self.original = self.binding.generic_op
            self.owner_thread = threading.get_ident()
            def observed(*args, **kwargs):
                seq = self.sequence
                self.sequence += 1
                caller = sys._getframe(1)
                # Nested observers pass through the original application caller frame.
                while caller.f_code.co_name == 'observed' and caller.f_globals is globals():
                    caller = caller.f_back
                try:
                    if threading.get_ident() != self.owner_thread:
                        raise CaptureError('dispatch from another thread is unsupported')
                    record = self.snapshot(args, kwargs, caller)
                    record['unavailable_fields'] = list(unavailable_paths(record))
                    if record['unavailable_fields']:
                        self.incomplete_calls += 1
                    self.emit({'kind': 'call', 'sequence': seq, **record})
                except Exception as exc:
                    self.error('observe', exc)
                finally:
                    del caller
                try:
                    result = self.original(*args, **kwargs)
                except BaseException as exc:
                    self.finish_call(seq, 'raised', typename(exc))
                    raise
                self.finish_call(seq, 'returned', typename(result))
                return result
            self.wrapper = observed
            self.binding.generic_op = observed
            self.active = True
            return self
        except BaseException:
            self.stream.close()
            raise

    def finish_call(self, seq, outcome, value_type):
        try:
            self.emit({'kind': 'outcome', 'sequence': seq, 'outcome': outcome, 'type': value_type})
        except Exception as exc:
            self.error('outcome', exc)

    def __exit__(self, exc_type, exc, tb):
        # Restore before file I/O or reporting can fail.
        if self.binding.generic_op is not self.wrapper:
            self.error('patch ownership changed', CaptureError())
        self.binding.generic_op = self.original
        self.active = False
        try:
            self.emit({'kind': 'footer', 'calls': self.sequence, 'errors': self.errors,
                       'observation_ok': not self.errors, 'identity_complete': False,
                       'body_raised': exc_type is not None, 'incomplete_calls': self.incomplete_calls, 'limits': _GAPS})
        except Exception as err:
            self.error('footer', err)
        finally:
            try:
                self.stream.close()
            except Exception as err:
                self.error('close', err)
        if self.errors and exc_type is None:
            raise CaptureError('identity capture failed; inspect errors and stderr')
        return False

    def roles(self, frame, tensors):
        info = {'file': self.sources.file(frame.f_code.co_filename),
                'function': frame.f_code.co_name, 'loaded_code_sha256': code_hash(frame.f_code),
                'line': frame.f_lineno}
        unknown = ['unknown'] * len(tensors)
        row = self.role_manifest.get(info['file'].get('sha256'), {}).get(frame.f_code.co_name)
        if row is None:
            return unknown, info
        key = (info['file']['sha256'], sys.flags.optimize)
        if key not in self._compiled:
            # The approved bytes must match the currently loaded code as well as the file.
            data = Path(frame.f_code.co_filename).read_bytes()
            if sha(data) != key[0]:
                return unknown, info
            mod = compile(data, frame.f_code.co_filename, 'exec', dont_inherit=True)
            self._compiled[key] = {c.co_name: code_hash(c) for c in mod.co_consts
                                   if isinstance(c, types.CodeType)}
        if self._compiled[key].get(frame.f_code.co_name) != info['loaded_code_sha256']:
            info['unavailable'] = 'loaded code differs from audited file'
            return unknown, info
        # Identity checks read locals only; no tensor equality, shapes or data reads.
        local = frame.f_locals
        names = row
        if row == 'matmul':
            outs = local.get('outs')
            if type(outs) not in (list, tuple):
                return unknown, info
            pairs = [('left', local.get('in0')), ('right', local.get('in1'))]
            pairs += [('output', t) for t in outs]
        elif row == 'sdpa':
            fuse = local.get('fuse')
            if fuse is None:
                names = [('query', 'q'), ('key', 'k'), ('value', 'v'), ('mask', 'mask'), ('output', 'out')]
                pairs = [(role, local.get(name)) for role, name in names]
            elif type(fuse) in (tuple, list) and len(fuse) >= 2:
                pairs = [('projection_input', fuse[0]), ('projection_weight', fuse[1]),
                         ('mask', local.get('mask')), ('output', local.get('out'))]
            else:
                return unknown, info
            gate = local.get('gate')
            if gate is not None:
                if type(gate) not in (tuple, list) or not gate:
                    return unknown, info
                pairs.append(('gate', gate[0]))
        else:
            pairs = [(role, local.get(name)) for role, name in names]
        if len(pairs) == len(tensors) and all(t is other for t, (_, other) in zip(tensors, pairs)):
            info['role_basis'] = 'audited loaded wrapper code and argument object identity; declared roles only'
            return [role for role, _ in pairs], info
        info['unavailable'] = 'argument identity differs from audited wrapper'
        return unknown, info

    def snapshot(self, args, kwargs, frame):
        tensors = args[0] if args else kwargs.get('io_tensors')
        pd = args[1] if len(args) > 1 else kwargs.get('program_descriptor')
        if type(tensors) not in (list, tuple):
            return {'unavailable': 'io_tensors is not a list/tuple; not consumed'}
        roles, caller = self.roles(frame, tensors)
        tensor_type = getattr(self.binding, 'Tensor', None)
        operands = []
        for i, tensor in enumerate(tensors):
            metadata = {'index': i, 'role': roles[i],
                        'same_object_as': next((j for j in range(i) if tensors[j] is tensor), None)}
            if tensor_type is not None and type(tensor) is tensor_type:
                for field in ('shape', 'padded_shape'):
                    try:
                        metadata[field] = [int(d) for d in getattr(tensor, field)]
                    except Exception as exc:
                        metadata[field] = missing('getter failed: ' + type(exc).__name__)
                for field in ('dtype', 'layout'):
                    metadata[field] = self.encoder.field(tensor, field)
            else:
                metadata['unavailable'] = 'unknown tensor type; metadata not accessed'
            metadata['storage_alias'] = missing('storage pointers are not queried')
            operands.append(metadata)
        pd_type = getattr(self.binding, 'ProgramDescriptor', None)
        if pd_type is None or type(pd) is not pd_type:
            return {'operands': operands, 'caller': caller,
                    'unavailable': 'unsupported descriptor; MeshProgramDescriptor is outside single-program scope'}
        descriptor = self.encoder.value(pd)
        for encoded, kernel in zip(descriptor['fields']['kernels'], pd.kernels):
            if 'fields' not in encoded:
                continue
            fields = encoded['fields']
            fields['runtime_args'] = self.encoder.runtime(kernel)
            source_type = kernel.source_type
            source_enum = self.binding.KernelDescriptor.SourceType
            if source_type == source_enum.FILE_PATH:
                fields['source_identity'] = self.sources.file(kernel.kernel_source) if Path(kernel.kernel_source).is_absolute() else missing('relative source resolution not established')
            elif source_type == source_enum.SOURCE_CODE:
                fields['source_identity'] = {'sha256': self.sources.bytes(kernel.kernel_source.encode('utf8')),
                                             'basis': 'inline UTF-8 source bytes'}
            else:
                fields['source_identity'] = missing('unknown source type')
        cache_hash = missing('binding hash function unavailable')
        fn = getattr(self.binding, 'compute_program_descriptor_hash', None)
        if fn is not None:
            try:
                cache_hash = int(fn(pd))
            except Exception as exc:
                cache_hash = missing('binding hash failed: ' + type(exc).__name__)
        record = {'operands': operands, 'caller': caller, 'descriptor': descriptor,
                  'descriptor_object_id': id(pd), 'descriptor_object_id_scope': 'process address; can be reused after destruction',
                  'binding_cache_hash': cache_hash,
                  'cache_hash_basis': 'binding structural hash; not a source content hash or execution identity',
                  'environment_at_call': {k: v for k, v in os.environ.items() if k.startswith('TT_BIO_') or k in ('TT_METAL_RUNTIME_ROOT', 'TT_METAL_HOME')},
                  'argument_form': {'positional_count': len(args), 'keyword_names': sorted(kwargs)},
                  'arithmetic': missing('source identities do not establish an arithmetic schedule')}
        record['execution_sha256'] = digest({'descriptor': descriptor, 'operands': operands,
                                              'environment': record['environment_at_call']})
        return record
