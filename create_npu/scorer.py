import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from create_npu.candidate_space import canonical_candidate_profile
from create_npu.models import ArchitectureCandidate, RequirementSpec, ToolResult
from create_npu.workloads import compatible_families_for_spec, resolve_family_for_spec


def score_design(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
    tool_results: List[ToolResult],
    report_summary: Optional[Dict[str, Any]] = None,
    supporting_files: Optional[List[str]] = None,
) -> float:
    return float(
        build_score_breakdown(
            spec=spec,
            architecture=architecture,
            tool_results=tool_results,
            report_summary=report_summary,
            supporting_files=supporting_files,
        )["total_score"]
    )


def build_score_breakdown(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
    tool_results: List[ToolResult],
    report_summary: Optional[Dict[str, Any]] = None,
    supporting_files: Optional[List[str]] = None,
) -> Dict[str, Any]:
    score = 10.0
    components: Dict[str, Dict[str, Any]] = {}
    tool_components: Dict[str, Dict[str, Any]] = {}
    tool_result_map = {result.name: result for result in tool_results}

    coverage_report = _load_json_artifact(
        supporting_files=supporting_files,
        artifact_name="coverage_report.json",
    )
    simulation_wrapper_report = _load_json_artifact(
        supporting_files=supporting_files,
        artifact_name="simulation_wrapper_report.json",
    )
    coverage_summary = coverage_report.get("summary", {})
    simulation_summary = (
        (report_summary or {}).get("simulation_wrapper")
        or simulation_wrapper_report.get("summary", {})
    )

    throughput_fit = _throughput_fit_component(spec=spec, architecture=architecture)
    score += throughput_fit["contribution"]
    components["throughput_fit"] = throughput_fit

    workload_alignment = {
        "contribution": _workload_family_bonus(spec=spec, architecture=architecture),
        "expected_family": resolve_family_for_spec(
            workload_type=spec.workload_type,
            preferred_dataflow=spec.preferred_dataflow,
        ),
        "compatible_families": compatible_families_for_spec(
            workload_type=spec.workload_type,
            preferred_dataflow=spec.preferred_dataflow,
        ),
        "selected_family": architecture.family,
    }
    score += workload_alignment["contribution"]
    components["workload_alignment"] = workload_alignment

    structured_requirement = {
        "contribution": _structured_requirement_bonus(spec=spec, architecture=architecture),
        "optimization_priority": spec.optimization_priority,
        "offchip_memory_type": spec.offchip_memory_type,
        "execution_mode": spec.execution_mode,
        "sparsity_support": spec.sparsity_support,
        "candidate_profile": canonical_candidate_profile(architecture.candidate_id),
    }
    score += structured_requirement["contribution"]
    components["structured_requirement"] = structured_requirement

    bus_width_bonus = 5.0 if architecture.bus_width_bits >= 512 else 0.0
    local_sram_bonus = 5.0 if architecture.local_sram_kb_per_tile >= 512 else 0.0
    score += bus_width_bonus
    score += local_sram_bonus
    components["memory_shape"] = {
        "contribution": round(bus_width_bonus + local_sram_bonus, 2),
        "bus_width_bits": architecture.bus_width_bits,
        "local_sram_kb_per_tile": architecture.local_sram_kb_per_tile,
    }

    power_budget = _power_budget_component(spec=spec, architecture=architecture)
    area_budget = _area_budget_component(spec=spec, architecture=architecture)
    memory_budget = _memory_budget_component(spec=spec, architecture=architecture)
    bandwidth_budget = _bandwidth_budget_component(spec=spec, architecture=architecture)
    ambiguity_component = _ambiguity_component(spec=spec)

    for name, component in (
        ("power_budget", power_budget),
        ("area_budget", area_budget),
        ("memory_budget", memory_budget),
        ("bandwidth_budget", bandwidth_budget),
        ("ambiguities", ambiguity_component),
    ):
        score += component["contribution"]
        components[name] = component

    tool_base_weights = {
        "python_reference": 15.0,
        "reference_coverage": 1.0,
        "simulation_wrapper": 1.0,
        "verilator_lint": 20.0,
        "iverilog_sim": 25.0,
        "yosys_synth": 10.0,
    }
    for tool_name, weight in tool_base_weights.items():
        tool_result = tool_result_map.get(tool_name)
        contribution = 0.0
        status = "missing"
        if tool_result is not None:
            if tool_result.available and tool_result.passed:
                contribution = weight
                status = "passed"
            elif tool_result.available and tool_result.passed is False:
                contribution = -weight
                status = "failed"
            elif not tool_result.available:
                status = "unavailable"
        score += contribution
        tool_components[tool_name] = {
            "contribution": round(contribution, 2),
            "weight": weight,
            "status": status,
        }

    coverage_quality = _coverage_quality_component(
        coverage_summary=coverage_summary,
        tool_result=tool_result_map.get("reference_coverage"),
    )
    simulation_quality = _simulation_quality_component(
        simulation_summary=simulation_summary,
        tool_result=tool_result_map.get("simulation_wrapper"),
    )
    score += coverage_quality["contribution"]
    score += simulation_quality["contribution"]
    components["coverage_quality"] = coverage_quality
    components["simulation_quality"] = simulation_quality
    components["tooling"] = tool_components

    quality_signals = {
        "coverage_score_pct": float(coverage_summary.get("coverage_score", 0.0)),
        "simulation_wrapper_case_pass_ratio": float(simulation_quality["case_pass_ratio"]),
        "simulation_wrapper_issue_class_ratio": float(simulation_quality["issue_class_ratio"]),
        "simulation_wrapper_detected_issue_classes": int(simulation_quality["detected_issue_classes"]),
        "simulation_wrapper_case_count": int(simulation_summary.get("case_count", 0) or 0),
        "selected_candidate_profile": canonical_candidate_profile(architecture.candidate_id),
    }

    raw_score = round(score, 6)
    total_score = round(max(0.0, min(raw_score, 100.0)), 2)
    return {
        "base_score": 10.0,
        "raw_score": raw_score,
        "total_score": total_score,
        "components": components,
        "quality_signals": quality_signals,
    }


def _load_json_artifact(
    supporting_files: Optional[List[str]],
    artifact_name: str,
) -> Dict[str, Any]:
    for candidate_path in supporting_files or []:
        path = Path(candidate_path)
        if path.name != artifact_name or not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _throughput_fit_component(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
) -> Dict[str, Any]:
    target_tops = spec.throughput_value or 1.0
    fit_error_ratio = abs(architecture.estimated_tops - target_tops) / target_tops
    return {
        "contribution": round(max(0.0, 25.0 - (fit_error_ratio * 40.0)), 2),
        "target_tops": float(target_tops),
        "estimated_tops": float(architecture.estimated_tops),
        "fit_error_ratio": round(float(fit_error_ratio), 6),
    }


def _power_budget_component(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
) -> Dict[str, Any]:
    if spec.power_budget_watts is None:
        return {"contribution": 0.0, "budget_watts": None}
    if architecture.estimated_power_watts <= spec.power_budget_watts:
        return {
            "contribution": 10.0,
            "budget_watts": float(spec.power_budget_watts),
            "estimated_power_watts": float(architecture.estimated_power_watts),
            "within_budget": True,
        }
    overflow_ratio = (
        architecture.estimated_power_watts - spec.power_budget_watts
    ) / spec.power_budget_watts
    return {
        "contribution": round(-min(20.0, overflow_ratio * 25.0), 2),
        "budget_watts": float(spec.power_budget_watts),
        "estimated_power_watts": float(architecture.estimated_power_watts),
        "overflow_ratio": round(float(overflow_ratio), 6),
        "within_budget": False,
    }


def _area_budget_component(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
) -> Dict[str, Any]:
    if spec.area_budget_mm2 is None:
        return {"contribution": 0.0, "budget_mm2": None}
    if architecture.estimated_area_mm2 <= spec.area_budget_mm2:
        return {
            "contribution": 5.0,
            "budget_mm2": float(spec.area_budget_mm2),
            "estimated_area_mm2": float(architecture.estimated_area_mm2),
            "within_budget": True,
        }
    overflow_ratio = (
        architecture.estimated_area_mm2 - spec.area_budget_mm2
    ) / spec.area_budget_mm2
    return {
        "contribution": round(-min(10.0, overflow_ratio * 10.0), 2),
        "budget_mm2": float(spec.area_budget_mm2),
        "estimated_area_mm2": float(architecture.estimated_area_mm2),
        "overflow_ratio": round(float(overflow_ratio), 6),
        "within_budget": False,
    }


def _memory_budget_component(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
) -> Dict[str, Any]:
    if spec.available_memory_mb is None:
        return {"contribution": 0.0, "budget_mb": None}
    estimated_memory_mb = _estimate_total_local_memory_mb(architecture)
    if estimated_memory_mb <= spec.available_memory_mb:
        return {
            "contribution": 5.0,
            "budget_mb": float(spec.available_memory_mb),
            "estimated_memory_mb": float(estimated_memory_mb),
            "within_budget": True,
        }
    overflow_ratio = (estimated_memory_mb - spec.available_memory_mb) / spec.available_memory_mb
    return {
        "contribution": round(-min(10.0, overflow_ratio * 15.0), 2),
        "budget_mb": float(spec.available_memory_mb),
        "estimated_memory_mb": float(estimated_memory_mb),
        "overflow_ratio": round(float(overflow_ratio), 6),
        "within_budget": False,
    }


def _bandwidth_budget_component(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
) -> Dict[str, Any]:
    if spec.memory_bandwidth_gb_per_s is None:
        return {"contribution": 0.0, "target_bandwidth_gb_per_s": None}
    estimated_bandwidth_gb_per_s = _estimate_bus_bandwidth_gb_per_s(architecture)
    if estimated_bandwidth_gb_per_s >= spec.memory_bandwidth_gb_per_s:
        return {
            "contribution": 8.0,
            "target_bandwidth_gb_per_s": float(spec.memory_bandwidth_gb_per_s),
            "estimated_bandwidth_gb_per_s": float(estimated_bandwidth_gb_per_s),
            "meets_target": True,
        }
    deficit_ratio = (
        spec.memory_bandwidth_gb_per_s - estimated_bandwidth_gb_per_s
    ) / spec.memory_bandwidth_gb_per_s
    return {
        "contribution": round(-min(15.0, deficit_ratio * 20.0), 2),
        "target_bandwidth_gb_per_s": float(spec.memory_bandwidth_gb_per_s),
        "estimated_bandwidth_gb_per_s": float(estimated_bandwidth_gb_per_s),
        "deficit_ratio": round(float(deficit_ratio), 6),
        "meets_target": False,
    }


def _ambiguity_component(spec: RequirementSpec) -> Dict[str, Any]:
    contribution = -min(10.0, float(len(spec.ambiguities) * 2)) if spec.ambiguities else 0.0
    return {
        "contribution": round(contribution, 2),
        "ambiguity_count": len(spec.ambiguities),
    }


def _coverage_quality_component(
    coverage_summary: Dict[str, Any],
    tool_result: Optional[ToolResult],
) -> Dict[str, Any]:
    coverage_score = float(coverage_summary.get("coverage_score", 0.0))
    if tool_result is None or not tool_result.available or not coverage_summary:
        return {
            "contribution": 0.0,
            "coverage_score_pct": coverage_score,
            "passes": False,
        }
    if tool_result.passed is True:
        contribution = (coverage_score / 100.0) * 4.0
    elif tool_result.passed is False:
        contribution = -((100.0 - coverage_score) / 100.0) * 4.0
    else:
        contribution = 0.0
    return {
        "contribution": round(contribution, 2),
        "coverage_score_pct": coverage_score,
        "passes": bool(coverage_summary.get("passes", False)),
        "covered_scheduler_states": len(
            coverage_summary.get("scheduler_state_coverage", {}).get("covered", [])
        ),
    }


def _simulation_quality_component(
    simulation_summary: Dict[str, Any],
    tool_result: Optional[ToolResult],
) -> Dict[str, Any]:
    case_count = max(1, int(simulation_summary.get("case_count", 0) or 0))
    pass_count = int(simulation_summary.get("pass_count", 0) or 0)
    pass_ratio = pass_count / case_count
    detected_issue_classes = sum(
        1
        for key in (
            "detected_shape_mismatch_count",
            "detected_sram_overflow_count",
            "detected_reuse_hazard_count",
            "detected_dma_compute_overlap_count",
        )
        if int(simulation_summary.get(key, 0) or 0) > 0
    )
    issue_class_ratio = detected_issue_classes / 4.0
    if tool_result is None or not tool_result.available or not simulation_summary:
        return {
            "contribution": 0.0,
            "case_pass_ratio": round(pass_ratio, 6),
            "issue_class_ratio": round(issue_class_ratio, 6),
            "detected_issue_classes": detected_issue_classes,
        }
    if tool_result.passed is True:
        contribution = pass_ratio * 2.5 + issue_class_ratio * 1.5
    elif tool_result.passed is False:
        contribution = -((1.0 - pass_ratio) * 2.5 + (1.0 - issue_class_ratio) * 1.5)
    else:
        contribution = 0.0
    return {
        "contribution": round(contribution, 2),
        "case_pass_ratio": round(pass_ratio, 6),
        "issue_class_ratio": round(issue_class_ratio, 6),
        "detected_issue_classes": detected_issue_classes,
        "actual_program_dma_compute_overlap_cycles": int(
            simulation_summary.get("actual_program_dma_compute_overlap_cycles", 0) or 0
        ),
    }


def _estimate_total_local_memory_mb(architecture: ArchitectureCandidate) -> float:
    return architecture.global_buffer_mb + (
        architecture.local_sram_kb_per_tile * architecture.tile_count
    ) / 1024.0


def _estimate_bus_bandwidth_gb_per_s(architecture: ArchitectureCandidate) -> float:
    return (architecture.bus_width_bits * architecture.target_frequency_mhz) / 8000.0


def _workload_family_bonus(spec: RequirementSpec, architecture: ArchitectureCandidate) -> float:
    compatible_families = compatible_families_for_spec(
        workload_type=spec.workload_type,
        preferred_dataflow=spec.preferred_dataflow,
    )
    expected_family = resolve_family_for_spec(
        workload_type=spec.workload_type,
        preferred_dataflow=spec.preferred_dataflow,
    )
    if architecture.family in compatible_families:
        if architecture.family == expected_family:
            return 12.0
        return 10.0
    return 0.0


def _structured_requirement_bonus(spec: RequirementSpec, architecture: ArchitectureCandidate) -> float:
    candidate_profile = canonical_candidate_profile(architecture.candidate_id)
    bonus = 0.0

    if spec.optimization_priority == "throughput" and candidate_profile == "throughput_max":
        bonus += 6.0
    elif spec.optimization_priority == "latency" and architecture.target_frequency_mhz >= 1000.0:
        bonus += 4.0
    elif spec.optimization_priority in ("efficiency", "area") and candidate_profile == "efficiency":
        bonus += 6.0
    elif spec.optimization_priority == "balanced" and candidate_profile == "balanced":
        bonus += 4.0

    if spec.offchip_memory_type == "HBM" and architecture.bus_width_bits >= 1024:
        bonus += 4.0
    elif spec.offchip_memory_type == "LPDDR" and architecture.bus_width_bits <= 512:
        bonus += 2.0

    if spec.execution_mode == "training" and architecture.local_sram_kb_per_tile >= 768:
        bonus += 3.0
    if spec.sparsity_support != "dense" and architecture.family == "sparse_pe_mesh":
        bonus += 4.0

    return bonus
