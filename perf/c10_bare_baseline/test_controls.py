"""Known-answer controls for the measurement, not a model-reference test."""
import tempfile, unittest
from pathlib import Path
import numpy as np
from reduce import coverage, holder_coverage, kabsch_rmsd, score, validate_atoms

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
    def test_atom_mapping_reordered_rows(self):
        cols=['label_asym_id','label_seq_id','label_atom_id','label_comp_id','Cartn_x','Cartn_y','Cartn_z']
        rows=[f'A {i} CA ALA {i%7} {i%13} {i%19}' for i in range(1,299)]
        head='data_test\nloop_\n'+'\n'.join('_atom_site.'+c for c in cols)+'\n'
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.cif',Path(d)/'b.cif';a.write_text(head+'\n'.join(rows)+'\n#\n');b.write_text(head+'\n'.join(rows[::-1])+'\n#\n')
            self.assertLess(score(a,b,298)['max_domain_all_atom_A'],1e-12)

if __name__=='__main__':unittest.main()
