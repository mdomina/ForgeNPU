import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from create_npu.models import PipelineResult, ToolResult
from create_npu.pipeline import CreateNPUPipeline


def run_regression_benchmark(
    output_dir: Path,
    require_full_toolchain: bool = False,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = CreateNPUPipeline(base_output_dir=output_dir)
    cases = _benchmark_cases(llm_model=llm_model)
    case_results: List[Dict[str, Any]] = []

    for case in cases:
        case_dir = output_dir / str(case["case_id"])
        result = pipeline.run(
            requirement_text=str(case["requirement"]),
            output_dir=case_dir,
            num_candidates=int(case["num_candidates"]),
            generator_backend=str(case["generator_backend"]),
            llm_model=case.get("llm_model"),
        )
        failures = _validate_case(
            case=case,
            result=result,
            require_full_toolchain=require_full_toolchain,
        )
        case_results.append(
            {
                "case_id": case["case_id"],
                "description": case["description"],
                "passed": not failures,
                "failures": failures,
                "output_dir": result.output_dir,
                "selected_candidate": result.architecture.candidate_id,
                "score": result.score,
            }
        )

    passed = all(case["passed"] for case in case_results)
    summary = {
        "passed": passed,
        "require_full_toolchain": require_full_toolchain,
        "cases": case_results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["summary_path"] = str(summary_path)
    return summary


def _benchmark_cases(llm_model: Optional[str]) -> List[Dict[str, Any]]:
    selected_llm_model = llm_model or "gpt-test"
    return [
        {
            "case_id": "toolchain_transformer_smoke",
            "description": "End-to-end regression del flow transformer con toolchain completa.",
            "requirement": "Voglio una NPU INT8 da 50 TOPS con supporto transformer e batch 1-4.",
            "num_candidates": 2,
            "generator_backend": "heuristic",
            "llm_model": None,
            "expected_candidate_id": "balanced",
            "expected_generator_backend": "heuristic",
            "expected_requested_backend": "heuristic",
            "expected_summary_values": [
                (("top_level_case_count",), 4),
                (("total_cycles",), 45),
                (("busy_cycles",), 37),
                (("done_cycles",), 4),
                (("idle_cycles",), 4),
                (("memory_path", "dma_cycles"), 11),
                (("memory_path", "load_cycles"), 8),
                (("memory_path", "store_cycles"), 7),
                (("memory_path", "working_set_utilization"), 0.5),
                (("memory_path", "total_dma_bits_transferred"), 176),
                (("memory_path", "total_store_bits_transferred"), 576),
                (("memory_path", "peak_external_bandwidth_gb_per_s"), 16.0),
                (("compute_path", "compute_cycles"), 5),
                (("compute_path", "flush_cycles"), 4),
                (("compute_path", "clear_cycles"), 1),
                (("compute_path", "estimated_mac_operations"), 24),
                (("top_npu_throughput", "estimated_effective_tops"), 5.688889),
                (("top_npu_throughput", "theoretical_peak_tops"), 51.2),
            ],
            "required_supporting_files": [],
        },
        {
            "case_id": "convolution_weight_stationary_mapping",
            "description": "Regressione workload-specifica per convolution con mapping weight-stationary.",
            "requirement": (
                "Voglio una NPU INT8 da 8 TOPS per CNN conv2d kernel 3x3 "
                "a bassa latenza con HBM e weight-stationary dataflow, batch 1-2."
            ),
            "num_candidates": 1,
            "generator_backend": "heuristic",
            "llm_model": None,
            "expected_candidate_id": "balanced",
            "expected_generator_backend": "heuristic",
            "expected_requested_backend": "heuristic",
            "expected_summary_values": [
                (("compiled_program", "tiling_strategy"), "weight_stationary_window"),
                (("compiled_program", "problem_shape", "kernel_height"), 3),
                (("compiled_program", "problem_shape", "output_channels"), 64),
                (("compiled_program", "mapping_plan", "dataflow"), "weight_stationary"),
                (("compiled_program", "operator_descriptors", 0, "op_type"), "conv2d"),
                (("workload_profile", "workload_type"), "convolution"),
                (("top_npu_throughput", "estimated_effective_tops"), 0.98304),
            ],
            "required_supporting_files": ["compiled_program.json"],
        },
        {
            "case_id": "sparse_spmm_mapping",
            "description": "Regressione workload-specifica per sparse matmul con mapping stream-compacted.",
            "requirement": "Voglio una NPU INT8 da 4 TOPS per sparse matmul SpMM con batch 1-2 e sparsity 2:4.",
            "num_candidates": 1,
            "generator_backend": "heuristic",
            "llm_model": None,
            "expected_candidate_id": "balanced",
            "expected_generator_backend": "heuristic",
            "expected_requested_backend": "heuristic",
            "expected_summary_values": [
                (("compiled_program", "tiling_strategy"), "sparse_stream_compaction"),
                (("compiled_program", "problem_shape", "density_percent"), 50),
                (("compiled_program", "mapping_plan", "dataflow"), "sparse_streaming"),
                (("compiled_program", "operator_descriptors", 0, "op_type"), "spmm"),
                (("compiled_program", "operator_descriptors", 0, "block_structure"), "2:4"),
                (("workload_profile", "workload_type"), "sparse_linear_algebra"),
                (("top_npu_throughput", "estimated_effective_tops"), 0.4096),
            ],
            "required_supporting_files": ["compiled_program.json"],
        },
        {
            "case_id": "tensor_descriptor_gemm",
            "description": (
                "Verifica che compiled_program esponga tensor_descriptors con shape, dtype, "
                "size_bytes e base_addr per un candidato dense GEMM."
            ),
            "requirement": "Voglio una NPU INT8 da 4 TOPS per dense GEMM con batch 1.",
            "num_candidates": 1,
            "generator_backend": "heuristic",
            "llm_model": None,
            "expected_candidate_id": "balanced",
            "expected_generator_backend": "heuristic",
            "expected_requested_backend": "heuristic",
            "expected_summary_values": [
                (("compiled_program", "tensor_descriptors", 0, "role"), "activation_input"),
                (("compiled_program", "tensor_descriptors", 0, "dtype"), "INT8"),
                (("compiled_program", "tensor_descriptors", 1, "role"), "weight_input"),
                (("compiled_program", "tensor_descriptors", 2, "role"), "output"),
                (("compiled_program", "tensor_descriptors", 2, "dtype"), "INT32"),
            ],
            "required_supporting_files": ["compiled_program.json"],
        },
        {
            "case_id": "gemmini_reference_delta_gemm",
            "description": (
                "Verifica che il report esponga gemmini_reference_delta con convergence, "
                "reference_config e throughput_comparison per un candidato dense GEMM."
            ),
            "requirement": "Voglio una NPU INT8 da 10 TOPS per dense GEMM con batch 1.",
            "num_candidates": 1,
            "generator_backend": "heuristic",
            "llm_model": None,
            "expected_candidate_id": "balanced",
            "expected_generator_backend": "heuristic",
            "expected_requested_backend": "heuristic",
            "expected_summary_values": [
                (("gemmini_reference_delta", "reference_name"), "gemmini_medium"),
                (("gemmini_reference_delta", "reference_config", "dataflow"), "weight_stationary"),
                (("gemmini_reference_delta", "candidate_vs_reference", "dataflow_match"), True),
                (("gemmini_reference_delta", "convergence"), "converges"),
                (("gemmini_reference_delta", "reference_config", "architecture_family"), "tiled_systolic_array"),
            ],
            "required_supporting_files": ["compiled_program.json"],
        },
        {
            "case_id": "dispatch_schedule_gemm",
            "description": (
                "Verifica che compiled_program esponga dispatch_schedule con entries, barrier_count, "
                "RAW hazard e engine_utilization per un candidato dense GEMM."
            ),
            "requirement": "Voglio una NPU INT8 da 4 TOPS per dense GEMM con batch 1.",
            "num_candidates": 1,
            "generator_backend": "heuristic",
            "llm_model": None,
            "expected_candidate_id": "balanced",
            "expected_generator_backend": "heuristic",
            "expected_requested_backend": "heuristic",
            "expected_summary_values": [
                (("compiled_program", "dispatch_schedule", "total_issue_cycles"), 2),
                (("compiled_program", "dispatch_schedule", "barrier_count"), 1),
                (("compiled_program", "dispatch_schedule", "raw_hazard_count"), 1),
                (("compiled_program", "dispatch_schedule", "entries", 0, "engine"), "compute"),
                (("compiled_program", "dispatch_schedule", "entries", 0, "barrier_before"), False),
                (("compiled_program", "dispatch_schedule", "entries", 1, "engine"), "vector"),
                (("compiled_program", "dispatch_schedule", "entries", 1, "barrier_before"), True),
            ],
            "required_supporting_files": ["compiled_program.json"],
        },
        {
            "case_id": "lowered_program_ops",
            "description": (
                "Verifica che compiled_program esponga lowered_program con Gemm e Relu "
                "per un candidato dense GEMM, con hardware_primitive e fusability corretti."
            ),
            "requirement": "Voglio una NPU INT8 da 4 TOPS per dense GEMM con batch 1.",
            "num_candidates": 1,
            "generator_backend": "heuristic",
            "llm_model": None,
            "expected_candidate_id": "balanced",
            "expected_generator_backend": "heuristic",
            "expected_requested_backend": "heuristic",
            "expected_summary_values": [
                (("compiled_program", "lowered_program", 0, "op_type"), "gemm"),
                (("compiled_program", "lowered_program", 0, "hardware_primitive"), "systolic_gemm"),
                (("compiled_program", "lowered_program", 0, "fusable_with_next"), True),
                (("compiled_program", "lowered_program", 1, "op_type"), "relu"),
                (("compiled_program", "lowered_program", 1, "hardware_primitive"), "vector_elementwise"),
            ],
            "required_supporting_files": ["compiled_program.json"],
        },
        {
            "case_id": "memory_planner_gemm",
            "description": (
                "Verifica che compiled_program esponga memory_plan con peak SRAM no-reuse, "
                "with-reuse e reuse_savings_pct per un candidato dense GEMM."
            ),
            "requirement": "Voglio una NPU INT8 da 4 TOPS per dense GEMM con batch 1.",
            "num_candidates": 1,
            "generator_backend": "heuristic",
            "llm_model": None,
            "expected_candidate_id": "balanced",
            "expected_generator_backend": "heuristic",
            "expected_requested_backend": "heuristic",
            "expected_summary_values": [
                (("compiled_program", "memory_plan", "operator_count"), 1),
                (("compiled_program", "memory_plan", "tensor_count"), 3),
                (("compiled_program", "memory_plan", "peak_sram_bytes_no_reuse"), 2048),
                (("compiled_program", "memory_plan", "peak_sram_bytes_with_reuse"), 2048),
                (("compiled_program", "memory_plan", "reuse_savings_pct"), 0.0),
            ],
            "required_supporting_files": ["compiled_program.json"],
        },
        {
            "case_id": "llm_fallback_capture",
            "description": "Regressione del fallback LLM con artifact di richiesta persistiti.",
            "requirement": "Voglio una NPU INT8 da 10 TOPS per dense GEMM.",
            "num_candidates": 1,
            "generator_backend": "llm",
            "llm_model": selected_llm_model,
            "expected_candidate_id": "balanced",
            "expected_generator_backend": "heuristic",
            "expected_requested_backend": "llm",
            "expected_summary_values": [
                (("top_level_case_count",), 4),
                (("total_cycles",), 45),
                (("compute_path", "compute_cycles"), 5),
                (("memory_path", "store_cycles"), 4),
                (("top_npu_throughput", "estimated_effective_tops"), 1.137778),
            ],
            "required_supporting_files": ["llm_request.json"],
        },
    ]


def _validate_case(
    case: Dict[str, Any],
    result: PipelineResult,
    require_full_toolchain: bool,
) -> List[str]:
    failures: List[str] = []

    if result.architecture.candidate_id != case["expected_candidate_id"]:
        failures.append(
            "Candidato selezionato inatteso: "
            f"atteso {case['expected_candidate_id']}, ottenuto {result.architecture.candidate_id}."
        )

    if result.generated.generator_backend != case["expected_generator_backend"]:
        failures.append(
            "Backend effettivo inatteso: "
            f"atteso {case['expected_generator_backend']}, ottenuto {result.generated.generator_backend}."
        )

    requested_backend = result.environment.get("llm", {}).get("requested_backend")
    if requested_backend != case["expected_requested_backend"]:
        failures.append(
            "Backend richiesto inatteso in environment.llm: "
            f"atteso {case['expected_requested_backend']}, ottenuto {requested_backend}."
        )

    if not result.report.get("available"):
        failures.append("Execution report non disponibile.")
    else:
        summary = result.report.get("summary", {})
        for path, expected in case["expected_summary_values"]:
            actual = _get_nested_value(summary, path)
            if not _values_match(actual, expected):
                failures.append(
                    "Summary inatteso per "
                    f"{'.'.join(str(part) for part in path)}: atteso {expected}, ottenuto {actual}."
                )

    tool_results = {tool.name: tool for tool in result.tool_results}
    python_reference = tool_results.get("python_reference")
    if not python_reference or python_reference.passed is not True:
        failures.append("python_reference non valido.")
    reference_coverage = tool_results.get("reference_coverage")
    if not reference_coverage or reference_coverage.passed is not True:
        failures.append("reference_coverage non valido.")

    if require_full_toolchain:
        failures.extend(_validate_required_toolchain(tool_results))

    supporting_files = [Path(path).name for path in result.generated.supporting_files]
    for required_name in case["required_supporting_files"]:
        if required_name not in supporting_files:
            failures.append(
                "Artifact richiesto assente nei supporting files: "
                f"{required_name}."
            )

    return failures


def _validate_required_toolchain(tool_results: Dict[str, ToolResult]) -> List[str]:
    failures = []
    for tool_name in ("python_reference", "verilator_lint", "iverilog_sim", "yosys_synth"):
        tool = tool_results.get(tool_name)
        if tool is None:
            failures.append(f"Tool result assente: {tool_name}.")
            continue
        if not tool.available:
            failures.append(f"Tool non disponibile: {tool_name}.")
            continue
        if tool.passed is not True:
            failures.append(f"Tool fallito: {tool_name}.")
    return failures


def _get_nested_value(payload: Dict[str, Any], path: Sequence[Any]) -> Any:
    current: Any = payload
    for key in path:
        if isinstance(key, int):
            if not isinstance(current, list):
                return None
            if key < 0 or key >= len(current):
                return None
            current = current[key]
            continue
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _values_match(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        if actual is None:
            return False
        return abs(float(actual) - expected) <= 1e-6
    return actual == expected
