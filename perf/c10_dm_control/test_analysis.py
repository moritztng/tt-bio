"""Independent checks for raw ticks, per-operation denominators and clock gaps."""
import copy, unittest
from fractions import Fraction
from analyze import RISCS, ZONES, aggregate, reduce_op, integer
from control import coverage
import dm_report

def fixture():
    base=2**54+17
    raw={"kernels":{((c,0),r):{"ZONE_START":base,"ZONE_END":base+1350}
                    for c in range(2) for r in RISCS},"sums":{}}
    for c in range(2):
        raw["sums"][((c,0),"NCRISC","DM-NOC-READ-BARRIER")]=135
        raw["sums"][((c,0),"NCRISC","DM-CB-RESERVE-BACK")]=270
        raw["sums"][((c,0),"TRISC_0","CB-COMPUTE-WAIT-FRONT")]=810
        raw["sums"][((c,0),"TRISC_2","CB-COMPUTE-RESERVE-BACK")]=675
    row={"GLOBAL CALL COUNT":"1","CORE COUNT":"2","OP CODE":"MatmulDeviceOperation",
         dm_report.DUR:"1000",**{v:"0" for v in dm_report.COLS.values()}}
    row.update({k:"fixture" for k in ("ATTRIBUTES","MATH FIDELITY","COMPUTE KERNEL SOURCE",
      "COMPUTE KERNEL HASH","DATA MOVEMENT KERNEL SOURCE","DATA MOVEMENT KERNEL HASH","PROGRAM HASH","PROGRAM CACHE HIT")})
    for name,value in (("nc_noc_read",200),("nc_reserve_back",400),
                       ("trisc0_wait_front",1200),("trisc2_reserve_back",1000)):
        row[dm_report.COLS[name]]=str(value)
    return row,raw

class AnalysisTests(unittest.TestCase):
    def test_integer_ticks_and_risc_ownership(self):
        row,raw=fixture(); op=reduce_op(row,raw)
        self.assertEqual(op["start_cycle"],2**54+17)
        self.assertEqual(op["span_cycles"],1350)
        self.assertEqual(op["threads"]["NCRISC"]["zone_fraction_of_op_span"]["DM-NOC-READ-BARRIER"],0.1)
        self.assertEqual(op["threads"]["TRISC_0"]["unclassified_core_cycles_sum"],1080)
        self.assertEqual(op["threads"]["TRISC_2"]["unclassified_core_cycles_sum"],1350)
        self.assertEqual(op["threads"]["TRISC_1"]["unclassified_core_cycles_sum"],2700)
    def test_each_operations_own_grid(self):
        ops=[{"cores":2,"span_cycles":100,"threads":{"NCRISC":{"zone_core_cycles_sum":{"read":100}}}},
             {"cores":4,"span_cycles":200,"threads":{"NCRISC":{"zone_core_cycles_sum":{"read":200}}}}]
        self.assertEqual(aggregate(ops,"NCRISC","read"),float(Fraction(1,3)))
    def test_missing_core_and_reversed_time_rejected(self):
        row,raw=fixture(); row["CORE COUNT"]="3"
        with self.assertRaisesRegex(ValueError,"Core count"):reduce_op(row,raw)
        row,raw=fixture()
        raw["kernels"][((0,0),"BRISC")]["ZONE_END"]=1
        with self.assertRaisesRegex(ValueError,"span"):reduce_op(row,raw)
    def test_accumulators_do_not_define_elapsed_time(self):
        row,raw=fixture()
        raw["sums"][((0,0),"NCRISC","DM-CB-RESERVE-BACK")]=99999
        with self.assertRaisesRegex(ValueError,"residency"):reduce_op(row,raw)
    def test_clock_coverage_rejects_gap_and_sag(self):
        interval={"start_monotonic_ns":0,"end_monotonic_ns":40_000_000}
        samples=[{"read_start_ns":n,"read_end_ns":n+2,"MHz":1350} for n in (1,20_000_000,39_000_000)]
        self.assertFalse(coverage(samples,interval)["pass"])
        samples=[{"read_start_ns":n,"read_end_ns":n+2,"MHz":1350} for n in range(1,40_000_000,1_000_000)]
        self.assertTrue(coverage(samples,interval)["pass"])
        samples[20]["MHz"]=800
        self.assertFalse(coverage(samples,interval)["pass"])
    def test_float_timestamp_rejected(self):
        with self.assertRaises(ValueError):integer("18014398509481984.0")

if __name__=="__main__":unittest.main()
