"""Negative controls for the archived live execution join and identity gate."""
import json, shutil, tempfile, unittest
from pathlib import Path
from reduce import reduce, fenced
P=Path(__file__).parent
RAW=P/"runs/smoke1"

class JoinControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="test-",dir=P/"runs")
        self.p=Path(self.tmp.name)
        (self.p/"raw").symlink_to((RAW/"raw").resolve())
        for name in ("criterion.json","raw_manifest.json"):
            shutil.copyfile(RAW/name,self.p/name)
        shutil.copytree(RAW/"out",self.p/"out")
    def tearDown(self): self.tmp.cleanup()
    def test_live_gaps_refuse_continuation(self):
        r=reduce(self.p)
        self.assertEqual(r["verdict"],"STOP")
        self.assertFalse(r["identity_smoke_pass"])
        self.assertEqual(len(r["joins"]),16)
        self.assertEqual(r["model_invocations_measured"],0)
    def test_missing_graph_dispatch(self):
        f=self.p/"out/matmul_on_graph.json";g=json.loads(f.read_text())
        n=next(n for n in g if (n.get("params") or {}).get("name")=="GenericOpDeviceOperation")
        g.remove(n);f.write_text(json.dumps(g))
        with self.assertRaisesRegex(ValueError,"Graph multiplicity"): reduce(self.p)
    def test_wrong_role(self):
        f=self.p/"out/matmul_on.jsonl";rows=[json.loads(l) for l in f.read_text().splitlines()]
        next(r for r in rows if r["kind"]=="call")["operands"][0]["role"]="output"
        f.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
        with self.assertRaisesRegex(ValueError,"operand roles"): reduce(self.p)
    def test_missing_fence(self):
        rows=[{"OP CODE":"UnaryDeviceOperation"} for _ in range(23)]
        with self.assertRaisesRegex(ValueError,"Fence count"): fenced(rows,4)
    def test_descriptor_cache_mismatch(self):
        f=self.p/"out/matmul_on.jsonl";rows=[json.loads(l) for l in f.read_text().splitlines()]
        next(r for r in rows if r["kind"]=="call")["binding_cache_hash"]+=1
        f.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
        with self.assertRaisesRegex(ValueError,"cache identity changed"): reduce(self.p)
if __name__=="__main__": unittest.main()
