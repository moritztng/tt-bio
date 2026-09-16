"""Synthetic controls; remapped IDs here are not device measurements."""
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from stream import DEFAULTS, FIELDS, append_chunk, reduce_chunk, verify_chunk

HEADER = 'ARCH: blackhole, CHIP_FREQ[MHz]: 1350, Max Compute Cores: 120\n'


def row(call=1024, kind='ZONE_START', tick=2**54+17, zone='BRISC-KERNEL', risc='BRISC'):
    return ['0','2','3',risc,'19',str(tick),'0',str(call),'','',zone,kind,'77','synthetic.cc','']


def fixture():
    # Interleaved operation IDs, and END appears before START in the second call.
    return [row(), row(2048, 'ZONE_END', 2**54+117),
            row(kind='ZONE_END', tick=2**54+117), row(2048),
            row(kind='ZONE_TOTAL', zone='DM-CB-WAIT-FRONT')]


class StreamTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_chunk(self, rows=None, *, content=None, limits=None, name='one', expected=None):
        if content is None:
            f = io.StringIO(newline=''); f.write(HEADER)
            w = csv.writer(f, lineterminator='\n'); w.writerow(FIELDS)
            w.writerows(rows if rows is not None else fixture())
            content = f.getvalue().encode()
        p = self.root / 'raw.csv'; p.write_bytes(content)
        return reduce_chunk(p, self.root/'out', chunk_id=name, counter_scope='synthetic/process1/boot1',
                            expected_bytes=len(content), expected_sha256=expected or hashlib.sha256(content).hexdigest(),
                            limits={**DEFAULTS, **(limits or {})})

    def records(self, receipt):
        verify_chunk(receipt['path'])
        with gzip.open(receipt['path'], 'rt') as f:
            return [json.loads(l) for l in f]

    def test_interleaved_integer_endpoints(self):
        r = self.records(self.run_chunk())
        ops = [x for x in r if x['kind']=='program']
        self.assertEqual([x['global_call_id'] for x in ops], [1024,2048])
        self.assertEqual(ops[0]['summary']['start_cycle'], 2**54+17)
        self.assertEqual(ops[0]['summary']['span_cycles'],100)
        self.assertEqual(r[-1]['noncontiguous_returns'],3)
        self.assertEqual(sum(m['rows'] for m in r[-1]['marker_coverage']),5)

    def test_device_trace_replay_are_separate(self):
        rows = []
        for device, trace, replay in [('0','',''),('1','',''),('0','4','1'),('0','4','2')]:
            for kind,tick in [('ZONE_START',100),('ZONE_END',200)]:
                r = row(kind=kind,tick=tick); r[0]=device;r[8]=trace;r[9]=replay;rows.append(r)
        ops = [r for r in self.records(self.run_chunk(rows)) if r['kind']=='program']
        self.assertEqual(len(ops),4)
        self.assertEqual(len({(r['global_call_id'],r['device'],r['trace'],r['replay']) for r in ops}),4)

    def test_repeated_ids_across_chunks_are_observations(self):
        a=self.records(self.run_chunk(name='a'));b=self.records(self.run_chunk(name='b'))
        self.assertEqual(a[1:-1],b[1:-1])
        self.assertEqual(len(list((self.root/'out').glob('*.gz'))),2)
        with self.assertRaises(FileExistsError):self.run_chunk(name='a')

    def test_missing_and_reversed_endpoints(self):
        for rows in ([row()], [row(),row(kind='ZONE_END',tick=5)],
                     [row(),row(kind='ZONE_END')]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError,'endpoints'):
                self.run_chunk(rows)
        self.assertFalse(list((self.root/'out').glob('*.gz')))

    def test_split_endpoint_pair_cannot_certify_either_chunk(self):
        for name,rows in [('first',[row()]),('second',[row(kind='ZONE_END',tick=2**54+117)])]:
            with self.assertRaisesRegex(ValueError,'endpoints'):
                self.run_chunk(rows,name=name)
        self.assertFalse(list((self.root/'out').glob('*.gz')))

    def test_publication_failure_retains_raw(self):
        with patch('stream.os.link',side_effect=OSError('synthetic publication failure')):
            with self.assertRaisesRegex(OSError,'publication'):
                self.run_chunk()
        self.assertTrue((self.root/'raw.csv').exists())
        self.assertFalse(list((self.root/'out').glob('*.gz')))

    def test_header_is_a_compatibility_check_not_clock_evidence(self):
        content=(HEADER.replace('1350','800')+','.join(FIELDS)+'\n').encode()
        with self.assertRaisesRegex(ValueError,'header'):self.run_chunk(content=content)
        records=self.records(self.run_chunk())
        self.assertIsNone(records[0]['clock_telemetry'])

    def test_duplicate_kernel_and_sum(self):
        for rows in (fixture()+[row()], fixture()+[fixture()[-1]]):
            with self.assertRaisesRegex(ValueError,'Duplicate'):self.run_chunk(rows)

    def test_sum_without_risc_endpoints_is_preserved_as_unplaced(self):
        rows=fixture(); rows[-1][3]='NCRISC';rows[-1][6]='7'
        records=self.records(self.run_chunk(rows))
        op=next(r for r in records if r.get('global_call_id')==1024)
        self.assertEqual(op['unplaced_zone_sums'],[dict(core=[2,3],risc='NCRISC',zone='DM-CB-WAIT-FRONT',cycles=7)])
        self.assertNotIn('NCRISC',op['summary']['threads'])

    def test_sum_without_core_endpoints_is_rejected(self):
        rows=fixture();rows[-1][1]='99'
        with self.assertRaisesRegex(ValueError,'own RISC'):self.run_chunk(rows)

    def test_per_core_oversum_cannot_hide_in_aggregate(self):
        rows=[row(tick=10),row(kind='ZONE_END',tick=20)]
        total=row(kind='ZONE_TOTAL',zone='DM-CB-WAIT-FRONT',tick=10);total[6]='11';rows.append(total)
        for kind,tick in [('ZONE_START',10),('ZONE_END',100)]:
            r=row(kind=kind,tick=tick);r[1]='4';rows.append(r)
        with self.assertRaisesRegex(ValueError,'own core'):self.run_chunk(rows)

    def test_unknown_markers_preserve_payload_and_duplicate_count(self):
        unknown=row(call=3000,risc='ERISC',zone='UNKNOWN',kind='CUSTOM',tick=20);unknown[14]='opaque'
        rows=fixture()+[unknown,unknown]
        records=self.records(self.run_chunk(rows))
        self.assertEqual([r['values'] for r in records if r['kind']=='unreduced_marker'],[unknown,unknown])
        op=next(r for r in records if r.get('global_call_id')==3000)
        self.assertEqual(op['status'],'unsupported_no_kernel_endpoints');self.assertIsNone(op['summary'])

    def test_unknown_zone_sum_preserved_without_attribution(self):
        rows=fixture();rows[-1][10]='UNRECOGNIZED';rows[-1][6]='7'
        records=self.records(self.run_chunk(rows))
        op=next(r for r in records if r.get('global_call_id')==1024)
        self.assertEqual(op['cores'][0]['zone_cycles'],{'UNRECOGNIZED':7})
        marker=next(r for r in records[-1]['marker_coverage'] if r['zone']=='UNRECOGNIZED')
        self.assertFalse(marker['known_zone'])

    def test_malformed_and_truncated_csv(self):
        for content in (HEADER.encode(), (HEADER+','.join(FIELDS)+'\n0,1\n').encode(),
                        (HEADER+','.join(FIELDS)+'\n'+','.join(row())).encode(),
                        (HEADER+','.join(FIELDS)+'\n"unfinished\n').encode()):
            with self.subTest(content=content), self.assertRaises((ValueError,StopIteration,csv.Error)):
                self.run_chunk(content=content)

    def test_bad_integer_and_incomplete_trace(self):
        for index,value in [(5,'1.0'),(5,'-1'),(8,'2')]:
            rows=fixture();rows[0][index]=value
            with self.assertRaises(ValueError):self.run_chunk(rows)

    def test_resource_caps_fail_without_publication(self):
        for key,value in [('max_raw_bytes',100),('max_rows',2),('max_operations',1),
                          ('max_core_riscs',1),('max_markers',1),('max_line_bytes',20),
                          ('max_output_bytes',100),('max_archive_bytes',1)]:
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_chunk(limits={key:value})
        self.assertFalse(list((self.root/'out').glob('*.gz')))

    def test_wrong_raw_receipt(self):
        with self.assertRaisesRegex(ValueError,'receipt'):self.run_chunk(expected='0'*64)

    def test_truncated_gzip_input(self):
        p=self.root/'bad.gz';p.write_bytes(gzip.compress(HEADER.encode())[:-5])
        with self.assertRaises((EOFError,StopIteration)):
            reduce_chunk(p,self.root/'out',chunk_id='bad',counter_scope='test',expected_bytes=len(HEADER),
                         expected_sha256=hashlib.sha256(HEADER.encode()).hexdigest(),limits=DEFAULTS)
        self.assertFalse(list((self.root/'out').glob('*.gz')))

    def test_footer_and_gzip_integrity(self):
        r=self.run_chunk();p=Path(r['path']);data=gzip.decompress(p.read_bytes())
        p.write_bytes(gzip.compress(data.replace(b'"span_cycles":100',b'"span_cycles":101')))
        with self.assertRaisesRegex(ValueError,'digest'):verify_chunk(p)
        p.write_bytes(gzip.compress(b'\n'.join(data.splitlines()[:-1])+b'\n'))
        with self.assertRaisesRegex(ValueError,'footer'):verify_chunk(p)
        p.write_bytes(gzip.compress(data)[:-5])
        with self.assertRaises(EOFError):verify_chunk(p)

    def test_synchronous_api_and_memory_cap(self):
        self.run_chunk();p=self.root/'raw.csv';b=p.read_bytes()
        kwargs=dict(chunk_id='api',counter_scope='test',expected_bytes=len(b),expected_sha256=hashlib.sha256(b).hexdigest())
        r=append_chunk(p,self.root/'api',**kwargs);self.assertTrue(Path(r['path']).exists())
        with self.assertRaises(subprocess.CalledProcessError):
            append_chunk(p,self.root/'low-memory',max_memory_mib=1,**kwargs)
        self.assertFalse(list((self.root/'low-memory').glob('*.gz')))

if __name__=='__main__':unittest.main()
