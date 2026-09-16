"""Known-answer controls for the measurement, not a model-reference test."""
import tempfile, unittest
from pathlib import Path
import numpy as np
from reduce import coverage, holder_coverage, kabsch_rmsd, score, validate_atoms, validate_sequence

class Controls(unittest.TestCase):
    def test_monotonic_units(self):
        start=912345678912345;end=start+14250000000
        self.assertEqual((end-start)/1e9,14.25)
        self.assertEqual((end-start)/1e9*1350,19237.5)
    def samples(self):return [dict(read_start_ns=i*1000000,read_end_ns=i*1000000+100,MHz=1350) for i in range(1,20)]
    def test_clock_accept(self):
        r=coverage(self.samples(),dict(start_monotonic_ns=0,end_monotonic_ns=20000000))
        self.assertTrue(r['pass']);self.assertEqual(r['samples'],19)
    def test_clock_reject_low_and_error(self):
        for patch in [dict(MHz=800),dict(error='read failed')]:
            s=self.samples();s[8].pop('MHz');s[8].update(patch)
            self.assertFalse(coverage(s,dict(start_monotonic_ns=0,end_monotonic_ns=20000000))['pass'])
    def test_clock_reject_gap_and_prefold(self):
        self.assertFalse(coverage(self.samples()[:3],dict(start_monotonic_ns=0,end_monotonic_ns=20000000))['pass'])
        self.assertFalse(coverage(self.samples(),dict(start_monotonic_ns=30000000,end_monotonic_ns=60000000))['pass'])
    def test_holder_reject_foreign_or_gap(self):
        interval=dict(start_monotonic_ns=0,end_monotonic_ns=1000000000)
        s=[dict(monotonic_ns=500000000,owner_nodes=['/dev/tenstorrent/0'],holders=[dict(pid=7)])]
        self.assertTrue(holder_coverage(s,interval,7)['passed'])
        s[0]['holders'].append(dict(pid=8));self.assertFalse(holder_coverage(s,interval,7)['passed'])
        self.assertFalse(holder_coverage([],interval,7)['passed'])
    def test_rigid_and_reflection(self):
        x=np.array([[0.,0,0],[1,0,0],[0,2,0],[0,0,3],[2,3,1]])
        rot=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        self.assertLess(kabsch_rmsd(x,x@rot+[3,-2,6]),1e-12)
        self.assertGreater(kabsch_rmsd(x,x*[-1,1,1]),.1)
    def test_invalid_coordinates_and_duplicate_identity(self):
        keys=[('A',str(i),'CA','ALA') for i in range(1,299)];xyz=np.zeros((298,3),dtype=np.float64)
        validate_atoms(keys,xyz,298)
        xyz[0,0]=np.nan
        with self.assertRaises(ValueError):validate_atoms(keys,xyz,298)
        xyz[0,0]=0;keys[0]=keys[1]
        with self.assertRaises(ValueError):validate_atoms(keys,xyz,298)
    def test_wrong_residue_sequence(self):
        keys=[('A',str(i),'CA','ALA') for i in range(1,299)]
        with self.assertRaises(ValueError):validate_sequence(keys,298)
    def test_atom_mapping_reordered_rows(self):
        cols=['label_asym_id','label_seq_id','label_atom_id','label_comp_id','Cartn_x','Cartn_y','Cartn_z']
        rows=[f'A {i} CA ALA {i%7} {i%13} {i%19}' for i in range(1,299)]
        head='data_test\nloop_\n'+'\n'.join('_atom_site.'+c for c in cols)+'\n'
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.cif',Path(d)/'b.cif';a.write_text(head+'\n'.join(rows)+'\n#\n');b.write_text(head+'\n'.join(rows[::-1])+'\n#\n')
            self.assertLess(score(a,b,298)['max_domain_all_atom_A'],1e-12)

if __name__=='__main__':unittest.main()

class RouteCounter(unittest.TestCase):
    """The accepted folds record above_cap_sdpa_counts == [0, 0]. That is only evidence the fused
    above-cap route stayed unreached if the counter can be made to move, so move it here."""
    def setUp(self):
        import tt_bio.tenstorrent as T
        self.T=T;self.sdpa=T._triatt_sdpa.sdpa;self.pairs=T._triatt_sdpa.fused_pairs
        T.SDPA_FUSED_LARGE_S_STATS[:]=[0,0]
    def tearDown(self):
        self.T._triatt_sdpa.sdpa=self.sdpa;self.T._triatt_sdpa.fused_pairs=self.pairs
        self.T.SDPA_FUSED_LARGE_S_STATS[:]=[0,0]
    def tensor(self,seq):return type("S",(),{"shape":(1,8,seq,32),"dtype":"bfloat16"})()
    def call(self,seq):
        q=self.tensor(seq)
        return self.T._tri_att_sdpa_at(q,q,q,q,0.176)
    def test_counter_moves_above_cap(self):
        self.T._triatt_sdpa.fused_pairs=lambda *a,**k:((256,2048),)
        self.T._triatt_sdpa.sdpa=lambda *a,**k:"fused"
        self.assertEqual(self.call(2048),"fused")
        self.assertEqual(list(self.T.SDPA_FUSED_LARGE_S_STATS),[1,0])
        self.T._triatt_sdpa.sdpa=lambda *a,**k:None
        with self.assertRaises(Exception):self.call(2048)
        self.assertEqual(list(self.T.SDPA_FUSED_LARGE_S_STATS),[1,1])
    def test_counter_stays_zero_at_baseline_sizes(self):
        reached=[]
        self.T._triatt_sdpa.fused_pairs=lambda *a,**k:reached.append(a) or ()
        for seq in [320,544]:
            with self.assertRaises(Exception):self.call(seq)
        self.assertEqual(reached,[])
        self.assertEqual(list(self.T.SDPA_FUSED_LARGE_S_STATS),[0,0])

class Ambient(unittest.TestCase):
    """A known CPU burner must show up in the host witness at the CPU it actually burned."""
    def test_sees_a_one_core_burner(self):
        import json, subprocess, sys, time
        with tempfile.TemporaryDirectory() as d:
            log=Path(d)/"ambient.jsonl"
            witness=subprocess.Popen([sys.executable,"perf/c10_bare_baseline/ambient.py",str(log)],stdin=subprocess.PIPE)
            burner=subprocess.Popen([sys.executable,"-c","import time\nt=time.time()\nwhile time.time()-t<1.5:pass"])
            try:
                burner.wait(timeout=20)
            finally:
                witness.communicate(b"stop\n",timeout=15)
                if burner.poll() is None:burner.kill();burner.wait(timeout=5)
            rows=[json.loads(x) for x in log.read_text().splitlines() if x.strip()]
            self.assertGreater(len(rows),3)
            seen=[b for r in rows for b in r["busy"] if b["pid"]==burner.pid]
            self.assertTrue(seen,"burner never observed")
            self.assertGreater(max(b["cpu_pct"] for b in seen),80)
            self.assertLess(max(b["cpu_pct"] for b in seen),130)
            self.assertTrue(all(b["own"] for b in seen),"same session must read as own")
    def test_separates_a_foreign_session(self):
        import json, subprocess, sys
        with tempfile.TemporaryDirectory() as d:
            log=Path(d)/"ambient.jsonl"
            witness=subprocess.Popen([sys.executable,"perf/c10_bare_baseline/ambient.py",str(log)],stdin=subprocess.PIPE)
            burner=subprocess.Popen([sys.executable,"-c","import time\nt=time.time()\nwhile time.time()-t<1.5:pass"],start_new_session=True)
            try:
                burner.wait(timeout=20)
            finally:
                witness.communicate(b"stop\n",timeout=15)
                if burner.poll() is None:burner.kill();burner.wait(timeout=5)
            rows=[json.loads(x) for x in log.read_text().splitlines() if x.strip()]
            foreign=[b for r in rows for b in r["busy"] if not b["own"] and b["cpu_pct"]>=20]
            self.assertTrue(any(b["pid"]==burner.pid for b in foreign),"foreign burner not flagged")
