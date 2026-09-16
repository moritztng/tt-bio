"""CPU archive controls and process resource measurements; never a device run."""
from __future__ import annotations
import argparse
import csv
import collections
import gzip
import hashlib
import json
from pathlib import Path
import time

from stream import ROOT, append_chunk, read_raw, span, verify_chunk, file_sha256
from analyze import reduce_op
from reduce_census import union_cycles

CENSUS = ROOT / 'perf/c10_burst_census/runs/census1'
CALIBRATION = ROOT / 'perf/c10_dm_control'


def receipt(path):
    h=hashlib.sha256();size=0
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):
            size+=len(block);h.update(block)
    return size,h.hexdigest()


def run(path, out, name, scope):
    size,sha=receipt(path)
    start=time.monotonic()
    result=append_chunk(path,out,chunk_id=name,counter_scope=scope,expected_bytes=size,expected_sha256=sha)
    result['cpu_replay_wall_s']=time.monotonic()-start
    result['output_sha256']=file_sha256(result['path'])
    footer=verify_chunk(result['path'])
    result['noncontiguous_returns']=footer['noncontiguous_returns']
    result['raw_sha256']=sha
    return result,footer


def compare(path,result,published_ops=None,calibration_rows=None):
    header,raw,zones=read_raw(path)
    seen=set();unreduced=unplaced=0
    with gzip.open(result['path'],'rt') as f:
        for line in f:
            r=json.loads(line)
            if r['kind']=='unreduced_marker':unreduced+=1
            if r['kind']!='program':continue
            i=r['global_call_id'];assert i not in seen;seen.add(i)
            unplaced+=len(r['unplaced_zone_sums'])
            present={risc for c,risc in raw[i]['kernels']}
            assert r['unplaced_zone_sums']==[dict(core=list(c),risc=risc,zone=z,cycles=v)
                for (c,risc,z),v in raw[i]['sums'].items() if risc not in present]
            assert r['summary']==span(raw[i]),(path,i,'calibrated span')
            assert r['device']==0 and r['trace'] is None and r['replay'] is None
            cores={(tuple(c['core']),c['risc']):c for c in r['cores']}
            assert cores.keys()==raw[i]['kernels'].keys()
            sums=collections.defaultdict(dict)
            for (c,risc,z),v in raw[i]['sums'].items():sums[c,risc][z]=v
            for key,ends in raw[i]['kernels'].items():
                assert cores[key]['start_cycle']==ends['ZONE_START']
                assert cores[key]['end_cycle']==ends['ZONE_END']
                assert cores[key]['zone_cycles']==sums.get(key,{})
            if published_ops and i in published_ops:
                accepted=published_ops[i]
                assert all(r['summary'][k]==accepted[k] for k in r['summary']), (path,i,'published census')
            if calibration_rows:
                accepted=reduce_op(calibration_rows[i],raw[i])
                assert all(r['summary'][k]==accepted[k] for k in ('start_cycle','end_cycle','span_cycles'))
                for risc,t in accepted['threads'].items():
                    observed=r['summary']['threads'][risc]
                    assert observed['resident_core_cycles']==t['resident_core_cycles_sum']
                    assert observed['unclassified_core_cycles']==t['unclassified_core_cycles_sum']
                    assert all(observed['zone_core_cycles'].get(z,0)==v for z,v in t['zone_core_cycles_sum'].items())
    assert seen==raw.keys()
    footer=verify_chunk(result['path'])
    actual={(m['risc'],m['zone']):m['rows'] for m in footer['marker_coverage'] if m['type']=='ZONE_TOTAL'}
    assert actual==zones
    assert sum(m['rows'] for m in footer['marker_coverage'] if m['arithmetic']=='unreduced')==unreduced
    return dict(programs=len(seen),unreduced_markers_preserved=unreduced,unplaced_zone_sums_preserved=unplaced,
                accepted_span_count_zone_core_endpoint_match=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',required=True,type=Path)
    ap.add_argument('--report',required=True,type=Path)
    ap.add_argument('--repeat-largest',type=int,default=2)
    args=ap.parse_args()
    if args.repeat_largest<0:ap.error('repeat-largest must be nonnegative')
    accepted=json.loads((CENSUS/'analysis.json').read_text())
    cap=json.loads((CENSUS/'out/capture.json').read_text())
    drains={d['label']:d for d in cap['drains'] if d['retained']}
    report=dict(scope='CPU replay of archives at sampled 1350 MHz; no new hardware measurement',
                predicted_model_cycles_saved=0,measured_model_cycles_saved=0,
                census=[],calibration=None,repeated_archive_resource_exercise=[],
                largest_discarded_chunk_bytes=377218386,
                limits='Retained inputs and repeats do not establish the discarded largest chunk marker mix, peak memory, completeness or capture overhead. No full-fold calibration or ceiling.')
    for w in accepted['windows']:
        name=w['case'];p=CENSUS/'out/windows'/(name+'.csv.gz')
        result,footer=run(p,args.output_dir,name,'archived-census1')
        assert footer['raw_sha256']==drains[name]['sha256']
        assert footer['raw_bytes']==drains[name]['bytes']
        check=compare(p,result,{o['global_call_id']:o for o in w['ops']})
        expected_ids=set(range(next(x for x in cap['windows'] if x['case']==name)['device_counter_before']-3,
                               next(x for x in cap['windows'] if x['case']==name)['device_counter_after']+3))
        ids=[];spans=[]
        with gzip.open(result['path'],'rt') as f:
            for line in f:
                r=json.loads(line)
                if r['kind']=='program':
                    ids.append(r['global_call_id'])
                    if r['global_call_id'] in {o['global_call_id'] for o in w['ops']}:spans.append(r['summary'])
        assert set(ids)=={i<<10 for i in expected_ids}
        assert len(spans)==w['invocations']
        assert sum(s['span_cycles'] for s in spans)==w['program_span_cycles_sum']
        assert union_cycles(spans)==w['program_union_cycles']
        result.update(check,accepted_inner_programs=w['invocations'],accepted_inner_span_sum=w['program_span_cycles_sum'])
        report['census'].append(result)
        print(json.dumps(dict(case=name,**result)),flush=True)
    p=CALIBRATION/'raw/profile_log_device.csv.gz'
    result,footer=run(p,args.output_dir,'calibration','archived-calibration')
    with (CALIBRATION/'raw/ops.csv').open() as f:
        rows={int(r['GLOBAL CALL COUNT']):r for r in csv.DictReader(f)}
    result.update(compare(p,result,calibration_rows=rows))
    cal=json.loads((CALIBRATION/'analysis.json').read_text())
    _,raw,_=read_raw(p)
    for interval in cal['intervals']:
        for op in interval['ops']:
            assert reduce_op(rows[op['call_id']],raw[op['call_id']])==op
    result['published_intervals']=len(cal['intervals'])
    result['published_interval_ops']=sum(len(i['ops']) for i in cal['intervals'])
    report['calibration']=result
    del raw
    largest=max(report['census'],key=lambda r:r['raw_bytes'])
    largest_name=Path(largest['path']).name.removesuffix('.jsonl.gz')
    for i in range(args.repeat_largest):
        result,footer=run(CENSUS/'out/windows'/(largest_name+'.csv.gz'),args.output_dir,
                          f'resource-repeat-{i}','archived-census1')
        assert result['raw_sha256']==largest['raw_sha256']
        result['synthetic_scope']='Same archived bytes and IDs, additional reduction only; zero new device observations'
        report['repeated_archive_resource_exercise'].append(result)
        print(json.dumps(result),flush=True)
    report['verdict']='GO_CPU_PREREQUISITE_ONLY'
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(args.report,flush=True)

if __name__=='__main__':main()
