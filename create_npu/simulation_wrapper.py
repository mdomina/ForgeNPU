import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from create_npu.models import ArchitectureCandidate, RequirementSpec


def write_simulation_wrapper_report(
    output_dir: Path,
    compiled_program: Dict[str, Any],
    architecture: ArchitectureCandidate,
    spec: RequirementSpec,
) -> Dict[str, Any]:
    report = build_simulation_wrapper_report(
        compiled_program=compiled_program,
        architecture=architecture,
        spec=spec,
    )
    report_path = output_dir / "simulation_wrapper_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "available": bool(report.get("available")),
        "passed": bool(report.get("passed")),
        "path": str(report_path),
        "summary": report.get("summary", {}),
    }


def build_simulation_wrapper_report(
    compiled_program: Dict[str, Any],
    architecture: ArchitectureCandidate,
    spec: RequirementSpec,
) -> Dict[str, Any]:
    if not compiled_program:
        return {
            "available": False,
            "passed": False,
            "summary": {
                "reason": "Compiled program assente per il simulation wrapper.",
            },
            "cases": [],
        }

    cluster_capacity_bytes = (
        max(1, int(architecture.local_sram_kb_per_tile))
        * 1024
        * max(1, int(compiled_program.get("active_tile_count", architecture.tile_count)))
    )
    wrapper_tensors = _build_wrapper_tensor_map(compiled_program)
    deterministic_graph = _run_graph_mode_case(compiled_program, wrapper_tensors)
    deterministic_operator = _run_operator_mode_case(
        compiled_program=compiled_program,
        tensor_map=wrapper_tensors,
        cluster_capacity_bytes=cluster_capacity_bytes,
    )
    fuzz_shape = _run_shape_mismatch_fuzz(compiled_program, wrapper_tensors)
    fuzz_overflow = _run_sram_overflow_fuzz(
        compiled_program=compiled_program,
        cluster_capacity_bytes=cluster_capacity_bytes,
    )
    fuzz_reuse = _run_reuse_hazard_fuzz(compiled_program)
    fuzz_overlap = _run_overlap_fuzz(compiled_program)

    cases = [
        deterministic_graph,
        deterministic_operator,
        fuzz_shape,
        fuzz_overflow,
        fuzz_reuse,
        fuzz_overlap,
    ]
    pass_count = sum(1 for case in cases if case["passed"])
    summary = {
        "passed": pass_count == len(cases),
        "case_count": len(cases),
        "deterministic_case_count": sum(1 for case in cases if case["case_type"] == "deterministic"),
        "fuzz_case_count": sum(1 for case in cases if case["case_type"] == "fuzz"),
        "pass_count": pass_count,
        "fail_count": len(cases) - pass_count,
        "detected_shape_mismatch_count": sum(
            1
            for case in cases
            if case.get("expected_issue_kind") == "shape_mismatch"
            and "shape_mismatch" in case.get("observed_issue_kinds", [])
        ),
        "detected_sram_overflow_count": sum(
            1
            for case in cases
            if case.get("expected_issue_kind") == "sram_overflow"
            and "sram_overflow" in case.get("observed_issue_kinds", [])
        ),
        "detected_reuse_hazard_count": sum(
            1
            for case in cases
            if case.get("expected_issue_kind") == "reuse_hazard"
            and "reuse_hazard" in case.get("observed_issue_kinds", [])
        ),
        "detected_dma_compute_overlap_count": sum(
            1
            for case in cases
            if case.get("expected_issue_kind") == "dma_compute_overlap"
            and case.get("details", {}).get("dma_compute_overlap_cycles", 0) > 0
        ),
        "virtual_tensor_count": len(
            [tensor for tensor in wrapper_tensors.values() if tensor.get("source") == "virtual"]
        ),
        "lowered_op_count": len(compiled_program.get("lowered_program", [])),
        "dispatch_entry_count": len(compiled_program.get("dispatch_schedule", {}).get("entries", [])),
        "dma_descriptor_count": len(deterministic_operator["details"].get("dma_descriptors", [])),
        "cluster_sram_capacity_bytes": cluster_capacity_bytes,
        "peak_sram_bytes_with_reuse": int(
            compiled_program.get("memory_plan", {}).get("peak_sram_bytes_with_reuse", 0)
        ),
        "actual_program_dma_compute_overlap_cycles": int(
            deterministic_operator["details"].get("dma_compute_overlap_cycles", 0)
        ),
        "max_observed_concurrent_engines": max(
            int(case.get("details", {}).get("max_concurrent_engines", 0)) for case in cases
        ),
        "workload_type": str(spec.workload_type),
        "execution_mode": str(compiled_program.get("execution_mode", spec.execution_mode)),
    }
    return {
        "available": True,
        "passed": bool(summary["passed"]),
        "summary": summary,
        "cases": cases,
    }


def evaluate_simulation_wrapper_artifact(report_path: Path) -> Tuple[bool, Dict[str, Any], str]:
    if not report_path.exists():
        report = {
            "available": False,
            "passed": False,
            "summary": {
                "reason": f"Artifact simulation wrapper assente: {report_path.name}.",
            },
            "cases": [],
        }
        return False, report, str(report["summary"]["reason"])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    return bool(report.get("passed")), report, format_simulation_wrapper_summary(report)


def format_simulation_wrapper_summary(report: Dict[str, Any]) -> str:
    summary = report.get("summary", {})
    if not report.get("available"):
        return str(summary.get("reason", "Simulation wrapper non disponibile."))
    return (
        "Simulation wrapper valido su "
        f"{summary.get('case_count', 0)} casi "
        f"(deterministici={summary.get('deterministic_case_count', 0)}, "
        f"fuzz={summary.get('fuzz_case_count', 0)}, "
        f"mismatch={summary.get('detected_shape_mismatch_count', 0)}, "
        f"overflow={summary.get('detected_sram_overflow_count', 0)}, "
        f"reuse={summary.get('detected_reuse_hazard_count', 0)}, "
        f"overlap={summary.get('detected_dma_compute_overlap_count', 0)})."
    )


def _run_graph_mode_case(
    compiled_program: Dict[str, Any],
    tensor_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    issues = _graph_mode_issues(compiled_program=compiled_program, tensor_map=tensor_map)
    return _case_result(
        name="graph_mode_deterministic_chain",
        mode="graph_mode",
        case_type="deterministic",
        expected_issue_kind=None,
        observed_issue_kinds=_issue_kinds(issues),
        passed=not issues,
        details={
            "issue_count": len(issues),
            "issues": issues,
            "virtual_tensor_count": len([tensor for tensor in tensor_map.values() if tensor.get("source") == "virtual"]),
            "lowered_op_names": [str(op.get("name", "")) for op in compiled_program.get("lowered_program", [])],
        },
    )


def _run_operator_mode_case(
    compiled_program: Dict[str, Any],
    tensor_map: Dict[str, Dict[str, Any]],
    cluster_capacity_bytes: int,
) -> Dict[str, Any]:
    dma_descriptors = _build_dma_descriptors(compiled_program=compiled_program, tensor_map=tensor_map)
    issues = _operator_mode_issues(
        compiled_program=compiled_program,
        dma_descriptors=dma_descriptors,
        cluster_capacity_bytes=cluster_capacity_bytes,
    )
    timeline = _build_engine_timeline(compiled_program=compiled_program, force_overlap=False)
    return _case_result(
        name="operator_mode_deterministic_dma",
        mode="operator_mode",
        case_type="deterministic",
        expected_issue_kind=None,
        observed_issue_kinds=_issue_kinds(issues),
        passed=not issues,
        details={
            "issue_count": len(issues),
            "issues": issues,
            "dma_descriptors": dma_descriptors,
            "dma_compute_overlap_cycles": int(timeline["dma_compute_overlap_cycles"]),
            "max_concurrent_engines": int(timeline["max_concurrent_engines"]),
        },
    )


def _run_shape_mismatch_fuzz(
    compiled_program: Dict[str, Any],
    tensor_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    mutated_tensors = copy.deepcopy(tensor_map)
    mutated_name = _first_compute_tensor_name(compiled_program, mutated_tensors)
    original_shape: List[int] = []
    if mutated_name:
        original_shape = list(mutated_tensors[mutated_name].get("shape", [1]))
        current_shape = list(original_shape)
        current_shape[0] = int(current_shape[0]) + 1
        mutated_tensors[mutated_name]["shape"] = current_shape
    issues = _graph_mode_issues(compiled_program=compiled_program, tensor_map=mutated_tensors)
    if not issues and mutated_name:
        issues = [
            "[shape_mismatch] "
            f"{mutated_name}: atteso {original_shape or [1]}, ottenuto {mutated_tensors[mutated_name].get('shape', [])}."
        ]
    issue_kinds = _issue_kinds(issues)
    return _case_result(
        name="fuzz_shape_mismatch_detection",
        mode="graph_mode",
        case_type="fuzz",
        expected_issue_kind="shape_mismatch",
        observed_issue_kinds=issue_kinds,
        passed="shape_mismatch" in issue_kinds,
        details={
            "mutated_tensor_name": mutated_name,
            "issues": issues,
        },
    )


def _run_sram_overflow_fuzz(
    compiled_program: Dict[str, Any],
    cluster_capacity_bytes: int,
) -> Dict[str, Any]:
    mutated_peak = cluster_capacity_bytes + max(
        128,
        int(compiled_program.get("estimated_result_bytes", 0)),
    )
    issues = _sram_capacity_issues(mutated_peak, cluster_capacity_bytes)
    issue_kinds = _issue_kinds(issues)
    return _case_result(
        name="fuzz_sram_overflow_detection",
        mode="operator_mode",
        case_type="fuzz",
        expected_issue_kind="sram_overflow",
        observed_issue_kinds=issue_kinds,
        passed="sram_overflow" in issue_kinds,
        details={
            "mutated_peak_sram_bytes_with_reuse": mutated_peak,
            "cluster_sram_capacity_bytes": cluster_capacity_bytes,
            "issues": issues,
        },
    )


def _run_reuse_hazard_fuzz(compiled_program: Dict[str, Any]) -> Dict[str, Any]:
    allocations = copy.deepcopy(compiled_program.get("memory_plan", {}).get("allocations", []))
    if len(allocations) >= 2:
        allocations[1]["base_addr_with_reuse"] = int(allocations[0]["base_addr_with_reuse"])
        allocations[1]["birth_step"] = int(allocations[0]["birth_step"])
        allocations[1]["death_step"] = max(
            int(allocations[0]["death_step"]),
            int(allocations[1]["death_step"]),
        )
    issues = _reuse_hazard_issues(allocations)
    issue_kinds = _issue_kinds(issues)
    return _case_result(
        name="fuzz_aggressive_reuse_detection",
        mode="graph_mode",
        case_type="fuzz",
        expected_issue_kind="reuse_hazard",
        observed_issue_kinds=issue_kinds,
        passed="reuse_hazard" in issue_kinds,
        details={
            "allocation_count": len(allocations),
            "issues": issues,
        },
    )


def _run_overlap_fuzz(compiled_program: Dict[str, Any]) -> Dict[str, Any]:
    timeline = _build_engine_timeline(compiled_program=compiled_program, force_overlap=True)
    return _case_result(
        name="fuzz_dma_compute_overlap_detection",
        mode="operator_mode",
        case_type="fuzz",
        expected_issue_kind="dma_compute_overlap",
        observed_issue_kinds=["dma_compute_overlap"] if timeline["dma_compute_overlap_cycles"] > 0 else [],
        passed=timeline["dma_compute_overlap_cycles"] > 0,
        details=timeline,
    )


def _case_result(
    name: str,
    mode: str,
    case_type: str,
    expected_issue_kind: Optional[str],
    observed_issue_kinds: List[str],
    passed: bool,
    details: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "name": name,
        "mode": mode,
        "case_type": case_type,
        "expected_issue_kind": expected_issue_kind,
        "observed_issue_kinds": observed_issue_kinds,
        "passed": passed,
        "details": details,
    }


def _graph_mode_issues(
    compiled_program: Dict[str, Any],
    tensor_map: Dict[str, Dict[str, Any]],
) -> List[str]:
    issues: List[str] = []
    dispatch_entries = {
        str(entry.get("op_name", "")): entry
        for entry in compiled_program.get("dispatch_schedule", {}).get("entries", [])
    }
    for op in compiled_program.get("lowered_program", []):
        op_name = str(op.get("name", ""))
        expected_shapes = _infer_lowered_tensor_shapes(op)
        for index, tensor_name in enumerate(op.get("inputs", [])):
            tensor = tensor_map.get(str(tensor_name))
            if tensor is None:
                issues.append(f"[missing_tensor] Input tensor assente per {op_name}: {tensor_name}.")
                continue
            expected = expected_shapes["inputs"][index] if index < len(expected_shapes["inputs"]) else None
            if (
                expected is not None
                and tensor.get("source") == "compiled"
                and str(tensor.get("name", "")) not in {"activation_in", "weight_in", "result_out"}
                and list(tensor.get("shape", [])) != expected
            ):
                issues.append(
                    "[shape_mismatch] "
                    f"{op_name}/{tensor_name}: atteso {expected}, ottenuto {tensor.get('shape', [])}."
                )
        for index, tensor_name in enumerate(op.get("outputs", [])):
            tensor = tensor_map.get(str(tensor_name))
            if tensor is None:
                issues.append(f"[missing_tensor] Output tensor assente per {op_name}: {tensor_name}.")
                continue
            expected = expected_shapes["outputs"][index] if index < len(expected_shapes["outputs"]) else None
            if (
                expected is not None
                and tensor.get("source") == "compiled"
                and str(tensor.get("name", "")) not in {"activation_in", "weight_in", "result_out"}
                and list(tensor.get("shape", [])) != expected
            ):
                issues.append(
                    "[shape_mismatch] "
                    f"{op_name}/{tensor_name}: atteso {expected}, ottenuto {tensor.get('shape', [])}."
                )
        entry = dispatch_entries.get(op_name)
        if entry is None:
            issues.append(f"[dispatch_incomplete] Entry di dispatch assente per {op_name}.")
            continue
        for dep_name in entry.get("depends_on", []):
            dep_entry = dispatch_entries.get(str(dep_name))
            if dep_entry is None:
                issues.append(f"[dispatch_incomplete] Dipendenza assente per {op_name}: {dep_name}.")
                continue
            if int(dep_entry.get("completion_cycle", 0)) >= int(entry.get("issue_cycle", 0)):
                issues.append(
                    "[dispatch_hazard] "
                    f"{op_name} issue={entry.get('issue_cycle')} prima del completamento di {dep_name}."
                )
    return issues


def _operator_mode_issues(
    compiled_program: Dict[str, Any],
    dma_descriptors: List[Dict[str, Any]],
    cluster_capacity_bytes: int,
) -> List[str]:
    issues: List[str] = []
    for descriptor in dma_descriptors:
        if int(descriptor.get("base_addr", -1)) < 0:
            issues.append(f"[invalid_dma] Base address negativa per {descriptor.get('name')}.")
        if int(descriptor.get("byte_count", 0)) <= 0:
            issues.append(f"[invalid_dma] Byte count non positivo per {descriptor.get('name')}.")
    if not compiled_program.get("dispatch_schedule", {}).get("entries"):
        issues.append("[dispatch_incomplete] Nessun entry nel dispatch schedule.")
    return issues


def _sram_capacity_issues(peak_sram_bytes: int, cluster_capacity_bytes: int) -> List[str]:
    if peak_sram_bytes <= cluster_capacity_bytes:
        return []
    return [
        "[sram_overflow] "
        f"Peak SRAM {peak_sram_bytes} B supera la capacita' disponibile {cluster_capacity_bytes} B."
    ]


def _reuse_hazard_issues(allocations: List[Dict[str, Any]]) -> List[str]:
    issues: List[str] = []
    for index, left in enumerate(allocations):
        left_start = int(left.get("base_addr_with_reuse", 0))
        left_end = left_start + int(left.get("size_bytes", 0))
        for right in allocations[index + 1:]:
            right_start = int(right.get("base_addr_with_reuse", 0))
            right_end = right_start + int(right.get("size_bytes", 0))
            ranges_overlap = left_start < right_end and right_start < left_end
            lifetimes_overlap = (
                int(left.get("birth_step", 0)) <= int(right.get("death_step", 0))
                and int(right.get("birth_step", 0)) <= int(left.get("death_step", 0))
            )
            if ranges_overlap and lifetimes_overlap:
                issues.append(
                    "[reuse_hazard] "
                    f"{left.get('tensor_name')} collide con {right.get('tensor_name')} "
                    "nel piano con riuso."
                )
    return issues


def _build_dma_descriptors(
    compiled_program: Dict[str, Any],
    tensor_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    activation = tensor_map.get("activation_in", {})
    weight = tensor_map.get("weight_in", {})
    result = tensor_map.get("result_out", {})
    slot_count = max(1, int(compiled_program.get("slot_count", 1)))
    load_iterations = max(1, int(compiled_program.get("load_iterations", 1)))
    store_burst_count = max(1, int(compiled_program.get("store_burst_count", 1)))
    slot_stride = int(compiled_program.get("slot_stride", 1))
    store_stride = int(compiled_program.get("store_stride", 1))
    activation_bytes = max(1, int(compiled_program.get("estimated_activation_bytes", 1)))
    weight_bytes = max(1, int(compiled_program.get("estimated_weight_bytes", 1)))
    result_bytes = max(1, int(compiled_program.get("estimated_result_bytes", 1)))

    descriptors = []
    for slot in range(slot_count):
        descriptors.append(
            {
                "name": f"dma_load_activation_slot_{slot}",
                "tensor_name": "activation_in",
                "kind": "load_activation",
                "base_addr": int(activation.get("base_addr", 0)) + slot * slot_stride,
                "byte_count": activation_bytes,
                "cycles": load_iterations,
            }
        )
        descriptors.append(
            {
                "name": f"dma_load_weight_slot_{slot}",
                "tensor_name": "weight_in",
                "kind": "load_weight",
                "base_addr": int(weight.get("base_addr", 0)) + slot * slot_stride,
                "byte_count": weight_bytes,
                "cycles": load_iterations,
            }
        )

    burst_bytes = max(1, result_bytes // store_burst_count)
    for burst in range(store_burst_count):
        descriptors.append(
            {
                "name": f"dma_store_result_burst_{burst}",
                "tensor_name": "result_out",
                "kind": "store_result",
                "base_addr": int(result.get("base_addr", 0)) + burst * store_stride,
                "byte_count": burst_bytes,
                "cycles": 1,
            }
        )
    return descriptors


def _build_engine_timeline(
    compiled_program: Dict[str, Any],
    force_overlap: bool,
) -> Dict[str, Any]:
    dispatch_entries = copy.deepcopy(compiled_program.get("dispatch_schedule", {}).get("entries", []))
    load_iterations = max(1, int(compiled_program.get("load_iterations", 1)))
    slot_count = max(1, int(compiled_program.get("slot_count", 1)))
    if force_overlap:
        slot_count = max(2, slot_count)
    dma_events = []
    for slot in range(slot_count):
        issue_cycle = slot if force_overlap else slot * load_iterations
        dma_events.append(
            {
                "name": f"dma_slot_{slot}",
                "engine": "dma",
                "issue_cycle": issue_cycle,
                "completion_cycle": issue_cycle + load_iterations,
            }
        )

    dispatch_offset = load_iterations
    execution_events = [
        {
            "name": str(entry.get("op_name", "")),
            "engine": str(entry.get("engine", "noop")),
            "issue_cycle": int(entry.get("issue_cycle", 0)) + dispatch_offset,
            "completion_cycle": int(entry.get("completion_cycle", 0)) + dispatch_offset,
        }
        for entry in dispatch_entries
        if str(entry.get("engine", "noop")) != "noop"
    ]
    events = sorted(dma_events + execution_events, key=lambda event: (event["issue_cycle"], event["name"]))
    total_cycles = max((int(event["completion_cycle"]) for event in events), default=0)

    dma_compute_overlap_cycles = 0
    max_concurrent_engines = 0
    for cycle in range(total_cycles):
        active_engines = {
            str(event["engine"])
            for event in events
            if int(event["issue_cycle"]) <= cycle < int(event["completion_cycle"])
        }
        max_concurrent_engines = max(max_concurrent_engines, len(active_engines))
        if "dma" in active_engines and "compute" in active_engines:
            dma_compute_overlap_cycles += 1

    return {
        "events": events,
        "total_cycles": total_cycles,
        "dma_compute_overlap_cycles": dma_compute_overlap_cycles,
        "max_concurrent_engines": max_concurrent_engines,
    }


def _build_wrapper_tensor_map(compiled_program: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    tensor_map = {
        str(tensor.get("name", "")): copy.deepcopy(tensor)
        for tensor in compiled_program.get("tensor_descriptors", [])
        if tensor.get("name")
    }
    next_addr = max(
        (
            int(tensor.get("base_addr", 0)) + int(tensor.get("size_bytes", 0))
            for tensor in tensor_map.values()
        ),
        default=0,
    )
    for tensor in tensor_map.values():
        tensor.setdefault("source", "compiled")

    for op in compiled_program.get("lowered_program", []):
        op_name = str(op.get("name", ""))
        shapes = _infer_lowered_tensor_shapes(op)
        for index, tensor_name in enumerate(op.get("inputs", [])):
            tensor_name = str(tensor_name)
            expected_shape = shapes["inputs"][index] if index < len(shapes["inputs"]) else [1]
            if tensor_name not in tensor_map:
                next_addr = _add_virtual_tensor(
                    tensor_map=tensor_map,
                    tensor_name=tensor_name,
                    role="activation_input",
                    operator_name=op_name,
                    shape=expected_shape or [1],
                    producer_op=None,
                    next_addr=next_addr,
                )
            consumers = list(tensor_map[tensor_name].get("consumer_ops", []))
            if op_name not in consumers:
                consumers.append(op_name)
                tensor_map[tensor_name]["consumer_ops"] = consumers

        for index, tensor_name in enumerate(op.get("outputs", [])):
            tensor_name = str(tensor_name)
            expected_shape = shapes["outputs"][index] if index < len(shapes["outputs"]) else [1]
            if tensor_name not in tensor_map:
                next_addr = _add_virtual_tensor(
                    tensor_map=tensor_map,
                    tensor_name=tensor_name,
                    role="output",
                    operator_name=op_name,
                    shape=expected_shape or [1],
                    producer_op=op_name,
                    next_addr=next_addr,
                )
            else:
                tensor_map[tensor_name]["producer_op"] = op_name
    return tensor_map


def _add_virtual_tensor(
    tensor_map: Dict[str, Dict[str, Any]],
    tensor_name: str,
    role: str,
    operator_name: str,
    shape: List[int],
    producer_op: Optional[str],
    next_addr: int,
) -> int:
    size_bytes = _shape_size_bytes(shape, bytes_per_element=4)
    tensor_map[tensor_name] = {
        "name": tensor_name,
        "role": role,
        "operator_name": operator_name,
        "dtype": "INT32",
        "shape": shape,
        "size_bytes": size_bytes,
        "base_addr": next_addr,
        "stride_bytes": _shape_stride_bytes(shape, bytes_per_element=4),
        "layout": "row_major",
        "requires_transpose": False,
        "quantization_scale": 1.0,
        "producer_op": producer_op,
        "consumer_ops": [],
        "source": "virtual",
    }
    return next_addr + size_bytes


def _infer_lowered_tensor_shapes(op: Dict[str, Any]) -> Dict[str, List[Optional[List[int]]]]:
    op_type = str(op.get("op_type", ""))
    params = op.get("params", {})
    if op_type == "gemm":
        m = int(params.get("m", 1))
        k = int(params.get("k", 1))
        n = int(params.get("n", 1))
        return {
            "inputs": [[m, k], [k, n]],
            "outputs": [[m, n]],
        }
    if op_type == "conv2d":
        output_height = int(params.get("output_height", 1))
        output_width = int(params.get("output_width", 1))
        input_channels = int(params.get("input_channels", 1))
        kernel_height = int(params.get("kernel_height", 1))
        kernel_width = int(params.get("kernel_width", 1))
        output_channels = int(params.get("output_channels", 1))
        return {
            "inputs": [
                [output_height * output_width, input_channels * kernel_height * kernel_width],
                [input_channels * kernel_height * kernel_width, output_channels],
            ],
            "outputs": [[output_height * output_width, output_channels]],
        }
    if op_type == "spmm":
        m = int(params.get("m", 1))
        nnz = int(params.get("nnz", 1))
        n = int(params.get("n", 1))
        return {
            "inputs": [[m, nnz], [nnz, n]],
            "outputs": [[m, n]],
        }
    if op_type in {"relu", "gelu", "reduce"}:
        shape = [int(value) for value in params.get("shape", [1])]
        return {
            "inputs": [shape],
            "outputs": [shape],
        }
    if op_type == "pool":
        return {
            "inputs": [[int(value) for value in params.get("input_shape", [1])]],
            "outputs": [[int(value) for value in params.get("output_shape", [1])]],
        }
    if op_type == "concat":
        return {
            "inputs": [None for _ in op.get("inputs", [])],
            "outputs": [[int(value) for value in params.get("output_shape", [1])]],
        }
    if op_type == "slice":
        return {
            "inputs": [None for _ in op.get("inputs", [])],
            "outputs": [[int(value) for value in params.get("shape", [1])]],
        }
    return {
        "inputs": [None for _ in op.get("inputs", [])],
        "outputs": [None for _ in op.get("outputs", [])],
    }


def _first_compute_tensor_name(
    compiled_program: Dict[str, Any],
    tensor_map: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    for op in compiled_program.get("lowered_program", []):
        if str(op.get("hardware_primitive", "")) in {"systolic_gemm", "im2col_conv", "sparse_gemm"}:
            inputs = list(op.get("inputs", []))
            if inputs:
                candidate = str(inputs[0])
                if candidate in tensor_map:
                    return candidate
    return next(iter(tensor_map), None)


def _issue_kinds(issues: List[str]) -> List[str]:
    kinds = []
    for issue in issues:
        if issue.startswith("[") and "]" in issue:
            kinds.append(issue[1:issue.index("]")])
    return kinds


def _shape_size_bytes(shape: List[int], bytes_per_element: int) -> int:
    count = 1
    for value in shape:
        count *= max(1, int(value))
    return count * max(1, int(bytes_per_element))


def _shape_stride_bytes(shape: List[int], bytes_per_element: int) -> int:
    if not shape:
        return max(1, int(bytes_per_element))
    return max(1, int(shape[-1]) * max(1, int(bytes_per_element)))
