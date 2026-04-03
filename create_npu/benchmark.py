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
                (("top_level_case_count",), 3),
                (("total_cycles",), 25),
                (("busy_cycles",), 19),
                (("done_cycles",), 3),
                (("idle_cycles",), 3),
                (("memory_path", "dma_cycles"), 8),
                (("memory_path", "load_cycles"), 6),
                (("memory_path", "working_set_utilization"), 0.5),
                (("memory_path", "total_dma_bits_transferred"), 128),
                (("memory_path", "peak_external_bandwidth_gb_per_s"), 2.0),
                (("compute_path", "compute_cycles"), 4),
                (("compute_path", "clear_cycles"), 1),
                (("compute_path", "estimated_mac_operations"), 20),
                (("top_npu_throughput", "estimated_effective_tops"), 8.192),
                (("top_npu_throughput", "theoretical_peak_tops"), 51.2),
            ],
            "required_supporting_files": [],
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
                (("top_level_case_count",), 3),
                (("total_cycles",), 25),
                (("compute_path", "compute_cycles"), 4),
                (("top_npu_throughput", "estimated_effective_tops"), 1.6384),
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
                    f"{'.'.join(path)}: atteso {expected}, ottenuto {actual}."
                )

    tool_results = {tool.name: tool for tool in result.tool_results}
    python_reference = tool_results.get("python_reference")
    if not python_reference or python_reference.passed is not True:
        failures.append("python_reference non valido.")

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


def _get_nested_value(payload: Dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for key in path:
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
