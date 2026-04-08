import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from create_npu.golden_model import scheduler_state_name

REQUIRED_MODULE_SECTIONS = [
    "mac_unit",
    "processing_element",
    "systolic_tile",
    "dma_engine",
    "scratchpad_controller",
    "accumulator_buffer",
    "scheduler",
    "cluster_control",
    "cluster_interconnect",
    "tile_compute_unit",
    "top_npu",
]
REQUIRED_SCHEDULER_STATES = [
    "IDLE",
    "DMA_ACT",
    "DMA_WGT",
    "LOAD",
    "COMPUTE",
    "STORE",
    "FLUSH",
    "CLEAR",
    "DONE",
]
REQUIRED_STRESS_TAGS = [
    "backpressure",
    "flush",
    "multi_tile",
    "control_routing",
    "store_fanout",
    "restart",
]
REQUIRED_COMPILED_PROGRAM_FIELDS = [
    "tiling_strategy",
    "problem_shape",
    "operator_descriptors",
    "mapping_plan",
    "estimated_mac_operations",
    "tensor_descriptors",
    "memory_plan",
    "lowered_program",
    "dispatch_schedule",
]


def evaluate_reference_coverage(reference_cases_path: str) -> Tuple[bool, Dict[str, Any]]:
    payload = json.loads(Path(reference_cases_path).read_text(encoding="utf-8"))
    report = build_reference_coverage_report(payload)
    report["reference_cases_path"] = reference_cases_path
    return not report["failures"], report


def build_reference_coverage_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    module_case_counts = {
        section: len(payload.get(section, [])) for section in REQUIRED_MODULE_SECTIONS
    }
    module_step_counts = {
        section: _case_step_count(payload.get(section, [])) for section in REQUIRED_MODULE_SECTIONS
    }
    stress_section_case_counts = {
        "top_npu_stress": len(payload.get("top_npu_stress", [])),
        "scheduler_stress": len(payload.get("scheduler_stress", [])),
        "cluster_control_stress": len(payload.get("cluster_control_stress", [])),
        "cluster_interconnect_stress": len(payload.get("cluster_interconnect_stress", [])),
    }

    scheduler_states = _collect_scheduler_states(payload)
    activation_banks, weight_banks, dma_banks = _collect_bank_coverage(payload)
    tile_counts = _collect_tile_counts(payload)
    stress_tags = _collect_stress_tags(payload)
    compiled_program = payload.get("compiled_program", {})
    compiled_program_fields_present = sorted(
        field_name for field_name in REQUIRED_COMPILED_PROGRAM_FIELDS if field_name in compiled_program
    )
    operator_descriptors = compiled_program.get("operator_descriptors", [])
    operator_types = sorted(
        {
            str(operator.get("op_type", "unknown"))
            for operator in operator_descriptors
            if isinstance(operator, dict)
        }
    )

    module_ratio = _ratio(sum(1 for count in module_case_counts.values() if count > 0), len(REQUIRED_MODULE_SECTIONS))
    scheduler_ratio = _ratio(
        len(set(scheduler_states).intersection(REQUIRED_SCHEDULER_STATES)),
        len(REQUIRED_SCHEDULER_STATES),
    )
    bank_ratio = _ratio(
        len(set(activation_banks).intersection({0, 1}))
        + len(set(weight_banks).intersection({0, 1}))
        + len(set(dma_banks).intersection({0, 1})),
        6,
    )
    tile_ratio = _ratio(len(set(tile_counts).intersection({1, 2, 3})), 3)
    stress_ratio = _ratio(len(set(stress_tags).intersection(REQUIRED_STRESS_TAGS)), len(REQUIRED_STRESS_TAGS))
    compiled_ratio = _ratio(
        len(compiled_program_fields_present) + (1 if operator_types else 0),
        len(REQUIRED_COMPILED_PROGRAM_FIELDS) + 1,
    )
    coverage_score = round(
        (
            module_ratio
            + scheduler_ratio
            + bank_ratio
            + tile_ratio
            + stress_ratio
            + compiled_ratio
        )
        / 6.0
        * 100.0,
        2,
    )

    failures = _coverage_failures(
        module_case_counts=module_case_counts,
        scheduler_states=scheduler_states,
        activation_banks=activation_banks,
        weight_banks=weight_banks,
        dma_banks=dma_banks,
        tile_counts=tile_counts,
        stress_tags=stress_tags,
        compiled_program=compiled_program,
        compiled_program_fields_present=compiled_program_fields_present,
        operator_types=operator_types,
    )
    summary = {
        "coverage_score": coverage_score,
        "module_case_counts": module_case_counts,
        "module_step_counts": module_step_counts,
        "stress_section_case_counts": stress_section_case_counts,
        "scheduler_state_coverage": {
            "required": REQUIRED_SCHEDULER_STATES,
            "covered": scheduler_states,
            "missing": sorted(set(REQUIRED_SCHEDULER_STATES) - set(scheduler_states)),
        },
        "bank_select_coverage": {
            "activation_banks": activation_banks,
            "weight_banks": weight_banks,
            "dma_banks": dma_banks,
        },
        "tile_count_coverage": {
            "covered": tile_counts,
            "required": [1, 2, 3],
            "missing": sorted({1, 2, 3} - set(tile_counts)),
        },
        "stress_tag_coverage": {
            "required": REQUIRED_STRESS_TAGS,
            "covered": stress_tags,
            "missing": sorted(set(REQUIRED_STRESS_TAGS) - set(stress_tags)),
        },
        "compiled_program_coverage": {
            "required_fields": REQUIRED_COMPILED_PROGRAM_FIELDS,
            "present_fields": compiled_program_fields_present,
            "missing_fields": sorted(
                set(REQUIRED_COMPILED_PROGRAM_FIELDS) - set(compiled_program_fields_present)
            ),
            "operator_count": len(operator_descriptors),
            "operator_types": operator_types,
        },
        "passes": not failures,
    }
    return {
        "summary": summary,
        "failures": failures,
    }


def format_reference_coverage_summary(report: Dict[str, Any]) -> str:
    summary = report.get("summary", {})
    scheduler_coverage = summary.get("scheduler_state_coverage", {})
    bank_coverage = summary.get("bank_select_coverage", {})
    tile_coverage = summary.get("tile_count_coverage", {})
    return (
        "Coverage score "
        f"{summary.get('coverage_score', 0.0):.2f}%: "
        f"moduli={sum(1 for count in summary.get('module_case_counts', {}).values() if count > 0)}"
        f"/{len(REQUIRED_MODULE_SECTIONS)}, "
        f"scheduler_states={len(scheduler_coverage.get('covered', []))}/{len(REQUIRED_SCHEDULER_STATES)}, "
        f"activation_banks={bank_coverage.get('activation_banks', [])}, "
        f"weight_banks={bank_coverage.get('weight_banks', [])}, "
        f"dma_banks={bank_coverage.get('dma_banks', [])}, "
        f"tile_counts={tile_coverage.get('covered', [])}."
    )


def _coverage_failures(
    module_case_counts: Dict[str, int],
    scheduler_states: List[str],
    activation_banks: List[int],
    weight_banks: List[int],
    dma_banks: List[int],
    tile_counts: List[int],
    stress_tags: List[str],
    compiled_program: Dict[str, Any],
    compiled_program_fields_present: List[str],
    operator_types: List[str],
) -> List[str]:
    failures: List[str] = []
    missing_modules = sorted(section for section, count in module_case_counts.items() if count <= 0)
    if missing_modules:
        failures.append(f"Moduli senza casi di riferimento: {', '.join(missing_modules)}.")

    missing_states = sorted(set(REQUIRED_SCHEDULER_STATES) - set(scheduler_states))
    if missing_states:
        failures.append(f"Scheduler states non coperti: {', '.join(missing_states)}.")

    if set(activation_banks) != {0, 1}:
        failures.append(f"Copertura activation bank incompleta: {activation_banks}.")
    if set(weight_banks) != {0, 1}:
        failures.append(f"Copertura weight bank incompleta: {weight_banks}.")
    if set(dma_banks) != {0, 1}:
        failures.append(f"Copertura DMA bank incompleta: {dma_banks}.")

    missing_tile_counts = sorted({1, 2, 3} - set(tile_counts))
    if missing_tile_counts:
        failures.append(f"Tile count non coperti: {', '.join(str(value) for value in missing_tile_counts)}.")

    missing_stress_tags = sorted(set(REQUIRED_STRESS_TAGS) - set(stress_tags))
    if missing_stress_tags:
        failures.append(f"Stress tag non coperti: {', '.join(missing_stress_tags)}.")

    missing_compiled_fields = sorted(set(REQUIRED_COMPILED_PROGRAM_FIELDS) - set(compiled_program_fields_present))
    if missing_compiled_fields:
        failures.append(
            "Campi compiled_program mancanti: " + ", ".join(missing_compiled_fields) + "."
        )
    if not compiled_program:
        failures.append("compiled_program assente nel manifest dei casi di riferimento.")
    if not operator_types:
        failures.append("Operator plan assente nel compiled_program.")

    return failures


def _collect_scheduler_states(payload: Dict[str, Any]) -> List[str]:
    covered = set()
    for section_name, state_field in (
        ("scheduler", "state_o"),
        ("scheduler_stress", "state_o"),
        ("top_npu", "scheduler_state_o"),
        ("top_npu_stress", "scheduler_state_o"),
    ):
        for case in payload.get(section_name, []):
            for step in case.get("steps", []):
                expected = step.get("expected", {})
                if state_field in expected:
                    covered.add(scheduler_state_name(int(expected[state_field])))
    return sorted(covered)


def _collect_bank_coverage(payload: Dict[str, Any]) -> Tuple[List[int], List[int], List[int]]:
    activation_banks = set()
    weight_banks = set()
    dma_banks = set()

    for case in payload.get("scratchpad_controller", []):
        for step in case.get("steps", []):
            activation_banks.add(int(step.get("activation_write_bank_i", 0)))
            activation_banks.add(int(step.get("activation_read_bank_i", 0)))
            weight_banks.add(int(step.get("weight_write_bank_i", 0)))
            weight_banks.add(int(step.get("weight_read_bank_i", 0)))

    for section_name in ("cluster_control", "cluster_control_stress", "cluster_interconnect", "cluster_interconnect_stress"):
        for case in payload.get(section_name, []):
            for step in case.get("steps", []):
                expected = step.get("expected", {})
                dma_banks.add(int(expected.get("dma_bank_select_o", 0)))
                activation_banks.add(int(expected.get("activation_read_bank_select_o", 0)))
                weight_banks.add(int(expected.get("weight_read_bank_select_o", 0)))

    return sorted(activation_banks), sorted(weight_banks), sorted(dma_banks)


def _collect_tile_counts(payload: Dict[str, Any]) -> List[int]:
    counts = set()
    for section_name in (
        "cluster_control",
        "cluster_control_stress",
        "cluster_interconnect",
        "cluster_interconnect_stress",
        "top_npu",
        "top_npu_stress",
    ):
        for case in payload.get(section_name, []):
            counts.add(int(case.get("tile_count", 1)))
    return sorted(counts)


def _collect_stress_tags(payload: Dict[str, Any]) -> List[str]:
    tags = set()
    for section_name in (
        "top_npu_stress",
        "scheduler_stress",
        "cluster_control_stress",
        "cluster_interconnect_stress",
    ):
        for case in payload.get(section_name, []):
            tags.update(str(tag) for tag in case.get("stress_tags", []))
    return sorted(tags)


def _case_step_count(cases: Iterable[Dict[str, Any]]) -> int:
    total = 0
    for case in cases:
        if "steps" in case:
            total += len(case.get("steps", []))
        else:
            total += 1
    return total


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)
