from typing import List

from create_npu.models import ArchitectureCandidate, RequirementSpec, ToolResult


def score_design(
    spec: RequirementSpec, architecture: ArchitectureCandidate, tool_results: List[ToolResult]
) -> float:
    score = 10.0

    target_tops = spec.throughput_value or 1.0
    fit_error_ratio = abs(architecture.estimated_tops - target_tops) / target_tops
    score += max(0.0, 25.0 - (fit_error_ratio * 40.0))

    if architecture.family.startswith("tiled_systolic") and spec.workload_type in (
        "transformer",
        "dense_gemm",
    ):
        score += 10.0

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
