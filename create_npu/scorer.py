from typing import List

from create_npu.models import ArchitectureCandidate, RequirementSpec, ToolResult
from create_npu.workloads import compatible_families_for_spec, resolve_family_for_spec


def score_design(
    spec: RequirementSpec, architecture: ArchitectureCandidate, tool_results: List[ToolResult]
) -> float:
    score = 10.0

    target_tops = spec.throughput_value or 1.0
    fit_error_ratio = abs(architecture.estimated_tops - target_tops) / target_tops
    score += max(0.0, 25.0 - (fit_error_ratio * 40.0))

    score += _workload_family_bonus(spec=spec, architecture=architecture)
    score += _structured_requirement_bonus(spec=spec, architecture=architecture)

    score += 5.0 if architecture.bus_width_bits >= 512 else 0.0
    score += 5.0 if architecture.local_sram_kb_per_tile >= 512 else 0.0

    if spec.power_budget_watts is not None:
        if architecture.estimated_power_watts <= spec.power_budget_watts:
            score += 10.0
        else:
            overflow_ratio = (
                architecture.estimated_power_watts - spec.power_budget_watts
            ) / spec.power_budget_watts
            score -= min(20.0, overflow_ratio * 25.0)

    if spec.area_budget_mm2 is not None:
        if architecture.estimated_area_mm2 <= spec.area_budget_mm2:
            score += 5.0
        else:
            score -= min(
                10.0,
                ((architecture.estimated_area_mm2 - spec.area_budget_mm2) / spec.area_budget_mm2)
                * 10.0,
            )

    if spec.available_memory_mb is not None:
        estimated_memory_mb = _estimate_total_local_memory_mb(architecture)
        if estimated_memory_mb <= spec.available_memory_mb:
            score += 5.0
        else:
            overflow_ratio = (estimated_memory_mb - spec.available_memory_mb) / spec.available_memory_mb
            score -= min(10.0, overflow_ratio * 15.0)

    if spec.memory_bandwidth_gb_per_s is not None:
        estimated_bandwidth_gb_per_s = _estimate_bus_bandwidth_gb_per_s(architecture)
        if estimated_bandwidth_gb_per_s >= spec.memory_bandwidth_gb_per_s:
            score += 8.0
        else:
            deficit_ratio = (
                spec.memory_bandwidth_gb_per_s - estimated_bandwidth_gb_per_s
            ) / spec.memory_bandwidth_gb_per_s
            score -= min(15.0, deficit_ratio * 20.0)

    tool_weights = {
        "python_reference": 15.0,
        "verilator_lint": 20.0,
        "iverilog_sim": 25.0,
        "yosys_synth": 10.0,
    }
    for result in tool_results:
        weight = tool_weights.get(result.name, 0.0)
        if result.available and result.passed:
            score += weight
        elif result.available and result.passed is False:
            score -= weight

    if spec.ambiguities:
        score -= min(10.0, float(len(spec.ambiguities) * 2))

    return round(max(0.0, min(score, 100.0)), 2)


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
    bonus = 0.0

    if spec.optimization_priority == "throughput" and architecture.candidate_id == "throughput_max":
        bonus += 6.0
    elif spec.optimization_priority == "latency" and architecture.target_frequency_mhz >= 1000.0:
        bonus += 4.0
    elif spec.optimization_priority in ("efficiency", "area") and architecture.candidate_id == "efficiency":
        bonus += 6.0
    elif spec.optimization_priority == "balanced" and architecture.candidate_id == "balanced":
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
