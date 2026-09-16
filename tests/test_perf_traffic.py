"""CPU controls for capture-byte accounting; imports no ttnn or model code."""
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LEGACY = load_module("traffic_legacy", "perf/b2x_difflayer/real_traffic.py")
CORRECTED = load_module("traffic_corrected", "perf/roof_arb/corrected_traffic.py")
COUNTERS = [LEGACY.counts, CORRECTED.counts]
TENSOR_BYTES = 8192 * 8192 * 2


def matmul_graph():
    return json.loads((ROOT / "perf/roof_budget/control_matmul_8192.json").read_text())


def node(counter, kind, connections=(), **params):
    return {"counter": counter, "node_type": kind, "connections": list(connections),
            "params": params, "input_tensors": []}


def alias_then_clone(clone=True, output_kind="DRAM"):
    nodes = matmul_graph()[:-1]  # replace capture_end with the extra operations
    nodes[14]["connections"] = [16]  # output tensor -> metadata view
    nodes[8]["connections"].append(17)  # both tensor aliases share ONE buffer
    nodes.extend([
        node(16, "function_start", name="ttnn.reshape"),
        node(17, "tensor", [19] if clone else []),
        node(18, "function_end", [17], name="ttnn.reshape"),
    ])
    if clone:
        nodes.extend([
            node(19, "function_start", name="ttnn.clone"),
            node(20, "buffer", [22], size=TENSOR_BYTES, type=output_kind),
            node(21, "buffer_allocate", [20]),
            node(22, "tensor"),
            node(23, "function_end", [22], name="ttnn.clone"),
        ])
    nodes.append(node(24, "capture_end"))
    return nodes


@pytest.mark.parametrize("counter", COUNTERS)
def test_dense_matmul_has_two_reads_and_one_write(counter):
    result = counter({"nodes": matmul_graph()})
    assert result["real_MB"] * 1e6 == 402653184
    assert result["real_r_MB"] * 1e6 == 2 * TENSOR_BYTES
    assert result["real_w_MB"] * 1e6 == TENSOR_BYTES
    assert result["floor_MB"] == result["real_MB"]
    assert result["terminal_read_removed_MB"] * 1e6 == TENSOR_BYTES
    assert result["assumed_read_MB"] == 0
    assert result["traffic_model_version"] == 2


@pytest.mark.parametrize("counter", COUNTERS)
@pytest.mark.parametrize("clone", [False, True])
def test_alias_does_not_read_but_its_actual_consumer_does(counter, clone):
    result = counter({"nodes": alias_then_clone(clone)})
    expected = (5 if clone else 3) * TENSOR_BYTES
    assert result["real_MB"] * 1e6 == expected
    assert result["assumed_read_MB"] == 0
    assert all(row[1] != "ttnn.reshape" for row in result["per_op"])
    # Only the final output lacks a read. The intermediate must retain its read.
    assert result["terminal_read_removed_MB"] * 1e6 == TENSOR_BYTES


@pytest.mark.parametrize("counter", COUNTERS)
@pytest.mark.parametrize("scratch", ["opaque", "internal_tensor", "freed_tensor"])
def test_internal_scratch_is_preserved_and_reported_as_assumed(counter, scratch):
    nodes = matmul_graph()
    extra = [node(17, "buffer", [19] if scratch != "opaque" else [],
                  size=4096, type="DRAM"),
             node(18, "buffer_allocate", [17])]
    if scratch != "opaque":
        extra.append(node(19, "tensor", [2] if scratch == "internal_tensor" else []))
    if scratch == "freed_tensor":
        extra.append(node(20, "buffer_deallocate", [17]))
    nodes[12:12] = extra  # allocated/used inside the matmul's device operation
    result = counter({"nodes": nodes})
    assert result["real_MB"] * 1e6 == pytest.approx(3 * TENSOR_BYTES + 8192)
    assert result["assumed_read_MB"] * 1e6 == 4096
    assert result["opaque_buffer_assumed_read_MB"] * 1e6 == (4096 if scratch == "opaque" else 0)
    assert result["terminal_read_removed_MB"] * 1e6 == TENSOR_BYTES


def test_corrected_counter_retains_dram_read_for_l1_output():
    result = CORRECTED.counts({"nodes": alias_then_clone(output_kind="L1")})
    assert result["real_MB"] * 1e6 == 4 * TENSOR_BYTES
    assert result["terminal_read_removed_MB"] == 0
    assert result["assumed_read_MB"] == 0
    assert sum(result["by_op"]) + result["unattributed_B"] == 4 * TENSOR_BYTES


def test_counters_share_semantics_when_optional_corrections_are_off():
    call = {"nodes": alias_then_clone()}
    old = LEGACY.counts(call)
    new = CORRECTED.counts(call, l1=False, pre=False, gate=False)
    for key in ("real_MB", "real_r_MB", "real_w_MB", "floor_MB", "once_MB",
                "terminal_read_removed_MB", "assumed_read_MB", "per_op"):
        assert old[key] == new[key]
