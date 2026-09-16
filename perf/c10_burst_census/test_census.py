import copy,gzip,json,unittest
from pathlib import Path
from reduce_census import graph_programs,host_ops,reduce_window,span,union_cycles,operand_work
P=Path(__file__).parent/"runs/census1"
class CensusTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cap=json.loads((P/"out/capture.json").read_text());cls.meta,cls.bad=host_ops(P/"raw/tracy_ops_data.csv.gz")
  cls.w=next(w for w in cls.cap["windows"] if w["case"]=="stream_copy_0")
 def test_live_legacy_clone_join(self):
  self.assertFalse(self.bad);r=reduce_window(P,self.w,self.meta)
  self.assertEqual(r["invocations"],4);self.assertEqual({o["class"] for o in r["ops"]},{"CloneOperation"})
  self.assertEqual(r["program_span_cycles_sum"],3643374)
 def test_counter_namespace_and_multiplicity(self):
  w=copy.deepcopy(self.w);w["device_counter_before"]+=1
  with self.assertRaisesRegex(ValueError,"counter range"):reduce_window(P,w,self.meta)
 def test_opcode_join_rejects_stale_identity(self):
  m=dict(self.meta);i=self.w["device_counter_before"]<<10;m[i]=dict(m[i],op_code="MatmulDeviceOperation")
  with self.assertRaisesRegex(ValueError,"opcode mismatch|count mismatch"):reduce_window(P,self.w,m)
 def test_native_end_is_required(self):
  g=json.load(gzip.open(P/"out/windows/stream_copy_0_graph.json.gz","rt"))
  n=next(n for n in g if n["node_type"]=="function_start" and (n.get("params") or {}).get("name")=="CloneOperation");n["connections"]=[]
  with self.assertRaisesRegex(ValueError,"end is ambiguous"):graph_programs(g,{"CloneOperation"})
 def test_missing_endpoints_rejected(self):
  r={"kernels":{((0,0),"BRISC"):{"ZONE_START":5}},"sums":{}}
  with self.assertRaisesRegex(ValueError,"endpoints"):span(r)
 def test_overlap_is_not_double_counted(self):
  self.assertEqual(union_cycles([{"start_cycle":0,"end_cycle":10},{"start_cycle":5,"end_cycle":12}]),12)
 def test_live_uncounted_work_stays_unknown(self):
  r=json.loads((P/"analysis.json").read_text())
  self.assertEqual(r["model_invocations_measured"],1478)
  c=next(c for c in r["classes"] if c["class"]=="NlpCreateHeadsDeviceOperation")
  self.assertEqual(c["invocations"],31);self.assertIsNone(c["modeled_dram_boundary_bytes"])
  self.assertIsNone(c["matrix_shape_flops"]);self.assertIsNone(c["cycles_above_roof"])
  self.assertEqual(sum(w["invocations"] for w in r["windows"]),1514)
  self.assertEqual(sum(o["class"]=="SDPAOperation" for w in r["windows"] for o in w["ops"]),30)
if __name__=="__main__":unittest.main()
