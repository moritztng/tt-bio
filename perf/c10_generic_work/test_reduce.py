import copy
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from perf.c10_flop_contract.audit import matrix_work
from perf.c10_generic_work.controls import examples
from perf.c10_generic_work.reduce import HERE, ROOT, digest, inspect_observer, reduce_contract, sha


def resign(item):
    item["descriptor_sha256"] = digest(item["descriptor"])
    return item


class Contracts(unittest.TestCase):
    def setUp(self):
        self.examples = examples()

    def result(self, name):
        r = reduce_contract(self.examples[name])
        self.assertEqual(r["status"], "SOURCE_CONDITIONAL", r)
        self.assertFalse(r["exact_total"])
        for field in ("issued_arithmetic", "sfpu_flops", "physical_bytes"):
            self.assertIsNone(r[field])
        return r

    def refuse(self, name, change, phrase):
        item = copy.deepcopy(self.examples[name])
        change(item["descriptor"])
        result = reduce_contract(resign(item))
        self.assertEqual(result["status"], "REFUSED", result)
        self.assertIn(phrase, result["reason"])
        self.assertIsNone(result["logical_matrix_flops"])
        self.assertIsNone(result["compulsory_logical_bytes"])

    def test_all_examples_remain_partial(self):
        for name in self.examples:
            with self.subTest(name=name):
                self.result(name)

    def test_dense_rectangular_flatten_counts(self):
        for name, m, k, n, bytes_ in (
            ("matmul_dense", 32, 32, 32, 6144),
            ("matmul_rectangular", 64, 32, 96, 22528),
            ("matmul_flattened", 128, 32, 96, 38912)):
            r = self.result(name)
            # Independent scalar triplet enumeration, no shared counter in the oracle.
            expected = sum(2 for _ in itertools.product(range(m), range(n), range(k)))
            self.assertEqual(r["logical_matrix_flops"], expected)
            self.assertEqual(r["compulsory_logical_bytes"]["total"], bytes_)

    def test_matmul_padding_is_separate(self):
        r = self.result("matmul_dense")
        self.assertEqual(r["logical_matrix_flops"], 65536)
        self.assertEqual(r["source_schedule"]["matrix_padded_extents"], [64, 64, 64])
        self.assertEqual(r["source_schedule"]["matmul_block_calls"], 8)
        # Independently enumerate the full grid and each core's two K steps.
        count = sum(1 for _ in itertools.product(range(2), range(2), range(2)))
        self.assertEqual(count, r["source_schedule"]["matmul_block_calls"])
        transposed_grid = self.examples["matmul_flattened"]
        transposed_grid["descriptor"]["compile"]["grid"] = [3, 2]
        r = reduce_contract(resign(transposed_grid))
        self.assertEqual(r["status"], "SOURCE_CONDITIONAL")
        self.assertEqual(r["source_schedule"]["matrix_padded_extents"], [192, 64, 128])
        self.assertEqual(r["source_schedule"]["matmul_block_calls"], 48)

    def test_float64_matrix_reference_and_shared_transpose_broadcast(self):
        # Dyadic operands: double dot product has an independent closed-form answer.
        m, k, n = 3, 5, 7
        for i, j in itertools.product(range(m), range(n)):
            got = math.fsum((i + t / 2.0) * (t / 4.0 - j) for t in range(k))
            sum_t = k * (k - 1) / 2
            sum_t2 = k * (k - 1) * (2 * k - 1) / 6
            expected = (i / 4 - j / 2) * sum_t - k * i * j + sum_t2 / 8
            self.assertEqual(got, expected)
        self.assertEqual(matrix_work((5, 3), (5, 7), transpose_a=True)["shape_flops"], 210)
        self.assertEqual(matrix_work((1, 3, 5), (4, 7, 5), transpose_b=True)["shape_flops"], 840)
        # These shape operations do NOT imply that mm_generic supports transpose/batched B.
        self.refuse("matmul_dense", lambda d: d["compile"].update(transpose_a=True), "transpose_a")
        self.refuse("matmul_dense", lambda d: d["compile"].update(transpose_b=True), "transpose_b")
        self.refuse("matmul_dense", lambda d: d["operands"][1].update(
            shape=[2, 32, 32], padded_shape=[2, 32, 32]), "batched right")

    def test_reblock_permutation_reference_and_bytes(self):
        # Distinct exactly representable doubles prove the two index maps are inverses.
        n, c = 3, 4
        original = {(i, j, ch): float((i * n + j) * c + ch)
                    for i, j, ch in itertools.product(range(n), range(n), range(c))}
        moved = {(ch, i, j): value for (i, j, ch), value in original.items()}
        inverse = {(i, j, ch): moved[ch, i, j] for i, j, ch in original}
        self.assertEqual(original, inverse)
        for name, elements, tile_calls in (("reblock_dense", 32768, 32),
                                           ("reblock_ragged", 69696, 256),
                                           ("reblock_reverse", 65536, 64)):
            r = self.result(name)
            self.assertEqual(r["logical_matrix_flops"], 0)
            self.assertEqual(r["conventional"], {})
            self.assertEqual(r["compulsory_logical_bytes"]["total"], 4 * elements)
            self.assertEqual(r["source_schedule"]["transpose_wh_tile_calls"], tile_calls)

    def test_gated_slices_rows_and_tail(self):
        for name, elements, tile_calls, rows in (
            ("reblock_gated", 32768, 32, [0, 32]),
            ("reblock_gated_row", 65536, 64, [32, 64]),
            ("reblock_gated_tail", 1056, 64, [32, 33])):
            r = self.result(name)
            self.assertEqual(r["conventional"], dict(multiply=elements, sigmoid_evaluations=elements))
            self.assertEqual(r["compulsory_logical_bytes"]["total"], 6 * elements)
            self.assertEqual(r["source_schedule"]["sigmoid_tile_calls"], tile_calls)
            self.assertEqual(r["output_region"]["rows"], rows)
        # Two independent double expressions of the reference gate. No bf16/device claim.
        for p, g in itertools.product((-2.0, 0.0, 0.5, 4.0), (-3.0, 0.0, 2.0)):
            self.assertAlmostEqual(p / (1.0 + math.exp(-g)), p * (1 + math.tanh(g / 2)) / 2, places=14)

    def test_sdpa_rectangular_and_broadcast(self):
        r = self.result("sdpa_rectangular")
        self.assertEqual(r["matrix_components"], {"QK_transpose": 262144, "PV": 262144})
        self.assertEqual(r["conventional"]["sum_add"], 4032)
        self.assertEqual(r["conventional"]["exp_evaluations"], 4096)
        self.assertEqual(r["conventional"]["mask_add"], 4096)
        self.assertEqual(r["source_schedule"]["qk_chunk_pairs"], 4)
        b = self.result("sdpa_broadcast_mask")
        self.assertEqual(b["logical_matrix_flops"], 1048576)
        self.assertEqual(b["conventional"]["mask_add"], 8192)
        self.assertEqual(b["compulsory_logical_bytes"]["read_by_role"]["mask"], 4096)
        self.assertEqual(self.result("sdpa_persistent_mask")["compulsory_logical_bytes"],
                         b["compulsory_logical_bytes"])

    def test_sdpa_fusions_do_not_charge_virtual_qkv(self):
        r = self.result("sdpa_fused_qkv")
        self.assertEqual(r["matrix_components"], dict(QK_transpose=262144, PV=262144, QKV_projection=1572864))
        self.assertEqual(r["compulsory_logical_bytes"]["read_by_role"],
                         dict(projection_input=8192, projection_weight=24576, mask=4096))
        self.assertEqual(r["compulsory_logical_bytes"]["write"], 8192)
        g = self.result("sdpa_gated")
        self.assertEqual(g["conventional"]["gate_multiply"], 2048)
        self.assertEqual(g["conventional"]["sigmoid_evaluations"], 2048)

    def test_float64_attention_reference(self):
        # Uniform zero scores give an exact analytic mean, rectangular Q/K lengths.
        values = [[0.25, -0.5], [0.75, 0.5], [1.25, 1.5], [1.75, 2.5]]
        logits = [0.0] * 4
        exps = [math.exp(x - max(logits)) for x in logits]
        probabilities = [x / math.fsum(exps) for x in exps]
        output = [math.fsum(p * v[j] for p, v in zip(probabilities, values)) for j in range(2)]
        self.assertEqual(output, [1.0, 1.0])
        self.assertEqual(math.fsum(probabilities), 1.0)

    def test_source_roles_flags_refuse_even_with_new_hash(self):
        self.refuse("matmul_dense", lambda d: d["sources"].update({next(iter(d["sources"])): "0" * 64}), "source hash")
        self.refuse("matmul_dense", lambda d: d.update(function="some_other_wrapper"), "function")
        self.refuse("matmul_dense", lambda d: d["operands"][0].update(role="right"), "roles/order")
        self.refuse("matmul_dense", lambda d: d.update(extra_defines={"FUSE_BIAS": "1"}), "defines")
        self.refuse("matmul_dense", lambda d: d["compile"].update(bias=True), "bias")
        self.refuse("sdpa_dense", lambda d: d["compile"].update(causal=True), "causal")
        self.refuse("sdpa_dense", lambda d: d["compile"].pop("phases"), "missing or unsupported")
        self.refuse("sdpa_dense", lambda d: d["compile"].update(streaming=True), "streaming")
        self.refuse("sdpa_fused_qkv", lambda d: d["compile"].update(gate_epilogue=True), "gate_epilogue")
        self.refuse("sdpa_fused_qkv", lambda d: d["compile"].update(heads_per_worker=2), "worker/buffer")
        self.refuse("sdpa_persistent_mask", lambda d: d["compile"].update(heads_per_worker=2), "one head")
        self.refuse("reblock_gated", lambda d: d["compile"].update(skip_sigmoid=True), "skip_sigmoid")

    def test_hash_tampering(self):
        item = self.examples["matmul_dense"]
        item["descriptor"]["compile"]["grid"] = [3, 3]
        self.assertIn("hash mismatch", reduce_contract(item)["reason"])

    def test_unsupported_shapes_dtypes_and_aliases(self):
        self.refuse("matmul_dense", lambda d: d["operands"][0].update(dtype="float32"), "bf16 TILE")
        self.refuse("matmul_dense", lambda d: d["operands"][0].update(shape=[31, 32]), "tile-aligned")
        self.refuse("matmul_dense", lambda d: d["operands"][2].update(storage="left"), "storage aliases")
        self.refuse("reblock_gated", lambda d: d["compile"].update(p_slice=128), "outside")
        self.refuse("reblock_gated", lambda d: d["compile"].update(p_slice=0), "overlapping")
        self.refuse("reblock_gated", lambda d: d["compile"].update(row_off=32), "row block")
        self.refuse("reblock_gated", lambda d: d["compile"].update(granularity=3), "granularity")
        self.refuse("reblock_reverse", lambda d: d["operands"][0].update(shape=[1, 64, 31, 31]), "ragged")
        self.refuse("sdpa_dense", lambda d: d["compile"].update(k_chunk=64), "padded chunk")
        self.refuse("sdpa_dense", lambda d: d["operands"][3].update(shape=[2, 1, 32, 32], padded_shape=[2, 1, 32, 32]), "mask broadcast")

    def test_ownership_and_unsupported_family(self):
        self.refuse("sdpa_dense", lambda d: d["ownership"].update(scope="inclusive_parent"), "ownership")
        self.refuse("sdpa_dense", lambda d: d["ownership"].update(cover="partial"), "ownership")
        self.refuse("sdpa_dense", lambda d: d.update(family="opaque_generic"), "unsupported family")

    def test_archived_refusal_hash_and_identity(self):
        provenance = json.loads((HERE / "evidence/provenance.json").read_text())
        for name, p in provenance.items():
            path = HERE / "evidence" / name
            self.assertEqual(sha(path.read_bytes()), p["sha256"])
            result = inspect_observer(path, 1)
            self.assertEqual(result["status"], "REFUSED")
            self.assertIn("per-core runtime arguments/ownership unavailable", result["reason"])
            self.assertIsNone(result["logical_matrix_flops"])
            self.assertEqual(len(result["execution_sha256"]), 64)
            self.assertIn("missing/duplicate", inspect_observer(path, 99)["reason"])

    def test_corrupt_archive_fails(self):
        records = [json.loads(x) for x in (HERE / "evidence/matmul_on.jsonl").read_text().splitlines()]
        next(r for r in records if r["kind"] == "call" and r["sequence"] == 1)["execution_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records))
            self.assertIn("execution hash mismatch", inspect_observer(path, 1)["reason"])
            path.write_text("\n".join(json.dumps(r) for r in records[:-1]))
            self.assertIn("incomplete", inspect_observer(path, 1)["reason"])
            records = [json.loads(x) for x in (HERE / "evidence/reblock_on.jsonl").read_text().splitlines()]
            next(r for r in records if r["kind"] == "source")["content"] = "YQ=="
            path.write_text("\n".join(json.dumps(r) for r in records))
            self.assertIn("embedded source hash mismatch", inspect_observer(path, 1)["reason"])

    def test_exact_total_cli_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            src, out = Path(directory) / "contract.json", Path(directory) / "report.json"
            src.write_text(json.dumps(self.examples["reblock_dense"]))
            command = [sys.executable, str(HERE / "reduce.py"), "--contract", str(src), "--out", str(out)]
            self.assertEqual(subprocess.run(command, cwd=ROOT, capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run(command + ["--require-exact-total"], cwd=ROOT, capture_output=True).returncode, 2)
            self.assertFalse(json.loads(out.read_text())["exact_total"])


if __name__ == "__main__":
    unittest.main()
