import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from create_npu.golden_model import scheduler_reference, scheduler_state_name, top_npu_reference
from create_npu.models import ArchitectureCandidate, GeneratedDesignBundle


def generate_execution_report(
    bundle: GeneratedDesignBundle,
    output_dir: Path,
    architecture: Optional[ArchitectureCandidate] = None,
) -> Dict[str, Any]:
    report_path = output_dir / "execution_report.json"
    report_payload = _build_execution_report(bundle=bundle, architecture=architecture)
    report_path.write_text(
        json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "available": bool(report_payload["available"]),
        "path": str(report_path),
        "summary": report_payload["summary"],
    }


def _build_execution_report(
    bundle: GeneratedDesignBundle,
    architecture: Optional[ArchitectureCandidate] = None,
) -> Dict[str, Any]:
    if not bundle.reference_cases_path:
        return {
            "available": False,
            "candidate_id": bundle.candidate_id,
            "generator_backend": bundle.generator_backend,
            "primary_module": bundle.primary_module,
            "summary": {
                "reason": "Manifest dei casi di riferimento assente.",
            },
            "cases": [],
        }

    payload = json.loads(Path(bundle.reference_cases_path).read_text(encoding="utf-8"))
    top_level_cases = payload.get("top_npu", [])
    if not top_level_cases:
        return {
            "available": False,
            "candidate_id": bundle.candidate_id,
            "generator_backend": bundle.generator_backend,
            "primary_module": bundle.primary_module,
            "summary": {
                "reason": "Sezione `top_npu` assente nei casi di riferimento.",
            },
            "cases": [],
        }

    case_reports = [
        _build_case_report(
            case,
            architecture=architecture,
            operand_width_bits=bundle.operand_width_bits,
        )
        for case in top_level_cases
    ]
    summary = _summarize_case_reports(
        case_reports,
        architecture=architecture,
        operand_width_bits=bundle.operand_width_bits,
    )

    return {
        "available": bool(case_reports),
        "candidate_id": bundle.candidate_id,
        "generator_backend": bundle.generator_backend,
        "primary_module": bundle.primary_module,
        "summary": summary,
        "cases": case_reports,
    }


def _build_case_report(
    case: Dict[str, Any],
    architecture: Optional[ArchitectureCandidate] = None,
    operand_width_bits: int = 8,
) -> Dict[str, Any]:
    rows = int(case.get("rows", 2))
    cols = int(case.get("cols", 2))
    depth = int(case.get("depth", 4))
    tile_count = int(case.get("tile_count", 1))
    steps = case["steps"]
    scheduler_snapshots = scheduler_reference(steps=steps, rows=rows, cols=cols)
    top_snapshots = top_npu_reference(
        steps=steps,
        rows=rows,
        cols=cols,
        depth=depth,
        tile_count=tile_count,
    )

    trace = []
    for cycle, (scheduler_snapshot, top_snapshot) in enumerate(
        zip(scheduler_snapshots, top_snapshots)
    ):
        state_id = int(scheduler_snapshot["state_o"])
        valids = [int(value) for value in top_snapshot["valids_o"]]
        psums = [int(value) for value in top_snapshot["psums_o"]]
        valid_lane_count = sum(valids)
        trace.append(
            {
                "cycle": cycle,
                "scheduler_state": {
                    "id": state_id,
                    "name": scheduler_state_name(state_id),
                },
                "busy": int(scheduler_snapshot["busy_o"]),
                "done": int(scheduler_snapshot["done_o"]),
                "event_tags": _event_tags(scheduler_snapshot=scheduler_snapshot, valid_lane_count=valid_lane_count),
                "memory_path": {
                    "dma_valid": int(scheduler_snapshot["dma_valid_o"]),
                    "dma_write_weights": int(scheduler_snapshot["dma_write_weights_o"]),
                    "dma_addr": int(scheduler_snapshot["dma_addr_o"]),
                    "dma_bank": int(scheduler_snapshot["dma_addr_o"]) % 2,
                    "dma_payload": [int(value) for value in scheduler_snapshot["dma_payload_o"]],
                    "load_vector_en": int(scheduler_snapshot["load_vector_en_o"]),
                    "activation_read_bank": int(scheduler_snapshot["activation_read_addr_o"]) % 2,
                    "activation_read_addr": int(scheduler_snapshot["activation_read_addr_o"]),
                    "weight_read_bank": int(scheduler_snapshot["weight_read_addr_o"]) % 2,
                    "weight_read_addr": int(scheduler_snapshot["weight_read_addr_o"]),
                },
                "compute_path": {
                    "compute_en": int(scheduler_snapshot["compute_en_o"]),
                    "clear_acc": int(scheduler_snapshot["clear_acc_o"]),
                    "psums": psums,
                    "valids": valids,
                    "valid_lane_count": valid_lane_count,
                },
            }
        )

    case_summary = _summarize_trace(
        trace=trace,
        rows=rows,
        cols=cols,
        depth=depth,
        operand_width_bits=operand_width_bits,
        architecture=architecture,
    )

    return {
        "name": case["name"],
        "tile_count": tile_count,
        "rows": rows,
        "cols": cols,
        "depth": depth,
        "program": _extract_program_inputs(steps),
        "summary": case_summary,
        "top_npu_throughput": _estimate_top_npu_throughput(
            summary=case_summary,
            architecture=architecture,
            seed_peak_macs_per_cycle=rows * cols * tile_count,
        ),
        "trace": trace,
    }


def _extract_program_inputs(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not steps:
        return {}

    first_step = steps[0]
    program_fields = [
        "activation_slot0_i",
        "activation_slot1_i",
        "weight_slot0_i",
        "weight_slot1_i",
        "tile_enable_i",
        "slot_count_i",
        "load_iterations_i",
        "compute_iterations_i",
        "clear_on_done_i",
    ]
    program = {}
    for field in program_fields:
        if field in first_step:
            value = first_step[field]
            if isinstance(value, list):
                program[field] = [int(item) for item in value]
            else:
                program[field] = int(value)
    return program


def _event_tags(scheduler_snapshot: Dict[str, Any], valid_lane_count: int) -> List[str]:
    tags = []

    if int(scheduler_snapshot["dma_valid_o"]):
        if int(scheduler_snapshot["dma_write_weights_o"]):
            tags.append("dma_write_weights")
        else:
            tags.append("dma_write_activations")

    if int(scheduler_snapshot["load_vector_en_o"]):
        tags.append("load_vector")

    if int(scheduler_snapshot["compute_en_o"]):
        tags.append("compute")

    if int(scheduler_snapshot["clear_acc_o"]):
        tags.append("clear")

    if valid_lane_count:
        tags.append("outputs_valid")

    return tags


def _empty_summary() -> Dict[str, Any]:
    return {
        "top_level_case_count": 0,
        "total_cycles": 0,
        "busy_cycles": 0,
        "done_cycles": 0,
        "idle_cycles": 0,
        "scheduler_state_sequence": [],
        "scheduler_state_histogram": {},
        "memory_path": {
            "dma_cycles": 0,
            "dma_activation_cycles": 0,
            "dma_weight_cycles": 0,
            "load_cycles": 0,
            "activation_elements_transferred": 0,
            "weight_elements_transferred": 0,
            "max_scratchpad_depth": 0,
            "activation_slots_touched": 0,
            "weight_slots_touched": 0,
            "peak_activation_slots_live": 0,
            "peak_weight_slots_live": 0,
            "working_set_utilization": 0.0,
            "total_dma_bits_transferred": 0,
            "average_dma_bits_per_dma_cycle": 0.0,
            "peak_dma_bits_per_cycle": 0,
            "effective_external_bandwidth_gb_per_s": 0.0,
            "peak_external_bandwidth_gb_per_s": 0.0,
            "theoretical_bus_bandwidth_gb_per_s": 0.0,
            "bus_bandwidth_utilization": 0.0,
            "peak_bus_bandwidth_utilization": 0.0,
        },
        "compute_path": {
            "compute_cycles": 0,
            "clear_cycles": 0,
            "output_valid_cycles": 0,
            "peak_valid_lanes": 0,
            "peak_abs_psum": 0,
            "estimated_mac_operations": 0,
        },
    }


def _summarize_case_reports(
    case_reports: List[Dict[str, Any]],
    architecture: Optional[ArchitectureCandidate] = None,
    operand_width_bits: int = 8,
) -> Dict[str, Any]:
    summary = _empty_summary()
    summary["top_level_case_count"] = len(case_reports)
    activation_slots_touched = set()
    weight_slots_touched = set()

    for case_report in case_reports:
        rows = int(case_report["rows"])
        cols = int(case_report["cols"])
        depth = int(case_report.get("depth", 0))
        activation_slots_live = set()
        weight_slots_live = set()
        summary["memory_path"]["max_scratchpad_depth"] = max(
            summary["memory_path"]["max_scratchpad_depth"],
            depth,
        )
        for step in case_report["trace"]:
            state_name = str(step["scheduler_state"]["name"])
            summary["total_cycles"] += 1
            summary["busy_cycles"] += int(step["busy"])
            summary["done_cycles"] += int(step["done"])
            if state_name == "IDLE":
                summary["idle_cycles"] += 1

            summary["scheduler_state_sequence"].append(state_name)
            histogram = summary["scheduler_state_histogram"]
            histogram[state_name] = histogram.get(state_name, 0) + 1

            memory_path = step["memory_path"]
            compute_path = step["compute_path"]

            if memory_path["dma_valid"]:
                summary["memory_path"]["dma_cycles"] += 1
                dma_bits_this_cycle = 0
                if memory_path["dma_write_weights"]:
                    summary["memory_path"]["dma_weight_cycles"] += 1
                    summary["memory_path"]["weight_elements_transferred"] += cols
                    weight_slots_live.add(int(memory_path["dma_addr"]))
                    weight_slots_touched.add(int(memory_path["dma_addr"]))
                    dma_bits_this_cycle = cols * operand_width_bits
                else:
                    summary["memory_path"]["dma_activation_cycles"] += 1
                    summary["memory_path"]["activation_elements_transferred"] += rows
                    activation_slots_live.add(int(memory_path["dma_addr"]))
                    activation_slots_touched.add(int(memory_path["dma_addr"]))
                    dma_bits_this_cycle = rows * operand_width_bits
                summary["memory_path"]["total_dma_bits_transferred"] += dma_bits_this_cycle
                summary["memory_path"]["peak_dma_bits_per_cycle"] = max(
                    summary["memory_path"]["peak_dma_bits_per_cycle"],
                    dma_bits_this_cycle,
                )

            if memory_path["load_vector_en"]:
                summary["memory_path"]["load_cycles"] += 1
                activation_slots_touched.add(int(memory_path["activation_read_addr"]))
                weight_slots_touched.add(int(memory_path["weight_read_addr"]))

            if compute_path["compute_en"]:
                summary["compute_path"]["compute_cycles"] += 1
                summary["compute_path"]["estimated_mac_operations"] += int(
                    compute_path["valid_lane_count"]
                )

            if compute_path["clear_acc"]:
                summary["compute_path"]["clear_cycles"] += 1

            if compute_path["valid_lane_count"] > 0:
                summary["compute_path"]["output_valid_cycles"] += 1

            summary["compute_path"]["peak_valid_lanes"] = max(
                summary["compute_path"]["peak_valid_lanes"],
                int(compute_path["valid_lane_count"]),
            )
            summary["compute_path"]["peak_abs_psum"] = max(
                summary["compute_path"]["peak_abs_psum"],
                max((abs(int(value)) for value in compute_path["psums"]), default=0),
            )
            summary["memory_path"]["peak_activation_slots_live"] = max(
                summary["memory_path"]["peak_activation_slots_live"],
                len(activation_slots_live),
            )
            summary["memory_path"]["peak_weight_slots_live"] = max(
                summary["memory_path"]["peak_weight_slots_live"],
                len(weight_slots_live),
            )

    summary["memory_path"]["activation_slots_touched"] = len(activation_slots_touched)
    summary["memory_path"]["weight_slots_touched"] = len(weight_slots_touched)
    max_scratchpad_depth = int(summary["memory_path"]["max_scratchpad_depth"])
    if max_scratchpad_depth > 0:
        summary["memory_path"]["working_set_utilization"] = round(
            (
                len(activation_slots_touched) + len(weight_slots_touched)
            )
            / float(2 * max_scratchpad_depth),
            6,
        )

    dma_cycles = int(summary["memory_path"]["dma_cycles"])
    if dma_cycles > 0:
        summary["memory_path"]["average_dma_bits_per_dma_cycle"] = round(
            summary["memory_path"]["total_dma_bits_transferred"] / float(dma_cycles),
            6,
        )

    if architecture is not None:
        theoretical_bandwidth = _estimate_bus_bandwidth_gb_per_s(architecture)
        effective_bandwidth = _estimate_external_bandwidth_gb_per_s(
            transferred_bits=int(summary["memory_path"]["total_dma_bits_transferred"]),
            total_cycles=int(summary["total_cycles"]),
            target_frequency_mhz=float(architecture.target_frequency_mhz),
        )
        peak_bandwidth = _estimate_peak_external_bandwidth_gb_per_s(
            peak_dma_bits_per_cycle=int(summary["memory_path"]["peak_dma_bits_per_cycle"]),
            target_frequency_mhz=float(architecture.target_frequency_mhz),
        )
        summary["memory_path"]["effective_external_bandwidth_gb_per_s"] = effective_bandwidth
        summary["memory_path"]["peak_external_bandwidth_gb_per_s"] = peak_bandwidth
        summary["memory_path"]["theoretical_bus_bandwidth_gb_per_s"] = theoretical_bandwidth
        if theoretical_bandwidth > 0.0:
            summary["memory_path"]["bus_bandwidth_utilization"] = round(
                effective_bandwidth / theoretical_bandwidth,
                6,
            )
            summary["memory_path"]["peak_bus_bandwidth_utilization"] = round(
                peak_bandwidth / theoretical_bandwidth,
                6,
            )

    summary["top_npu_throughput"] = _estimate_top_npu_throughput(
        summary=summary,
        architecture=architecture,
        seed_peak_macs_per_cycle=int(summary["compute_path"]["peak_valid_lanes"]),
    )
    return summary


def _summarize_trace(
    trace: List[Dict[str, Any]],
    rows: int,
    cols: int,
    depth: int,
    operand_width_bits: int,
    architecture: Optional[ArchitectureCandidate],
) -> Dict[str, Any]:
    case_report = {
        "rows": rows,
        "cols": cols,
        "depth": depth,
        "trace": trace,
    }
    summary = _summarize_case_reports(
        [case_report],
        architecture=architecture,
        operand_width_bits=operand_width_bits,
    )
    summary.pop("top_npu_throughput", None)
    return summary


def _estimate_top_npu_throughput(
    summary: Dict[str, Any],
    architecture: Optional[ArchitectureCandidate],
    seed_peak_macs_per_cycle: Optional[int] = None,
) -> Dict[str, Any]:
    total_cycles = int(summary.get("total_cycles", 0))
    busy_cycles = int(summary.get("busy_cycles", 0))
    compute_cycles = int(summary.get("compute_path", {}).get("compute_cycles", 0))
    estimated_mac_operations = int(
        summary.get("compute_path", {}).get("estimated_mac_operations", 0)
    )

    throughput_summary: Dict[str, Any] = {
        "available": False,
        "estimation_model": "peak_scaled_by_compute_duty_cycle",
        "total_cycles": total_cycles,
        "compute_cycles": compute_cycles,
        "scheduler_overhead_cycles": max(0, total_cycles - compute_cycles),
        "seed_effective_macs_per_cycle": round(
            (estimated_mac_operations / float(total_cycles)) if total_cycles else 0.0,
            6,
        ),
        "seed_effective_ops_per_cycle": round(
            ((estimated_mac_operations * 2.0) / float(total_cycles)) if total_cycles else 0.0,
            6,
        ),
    }

    if seed_peak_macs_per_cycle is not None and seed_peak_macs_per_cycle > 0:
        seed_peak_macs = int(seed_peak_macs_per_cycle)
        throughput_summary["seed_peak_macs_per_cycle"] = seed_peak_macs
        throughput_summary["seed_compute_array_utilization"] = round(
            ((estimated_mac_operations / float(total_cycles * seed_peak_macs)) if total_cycles else 0.0),
            6,
        )

    if architecture is None:
        throughput_summary["reason"] = "Architettura candidata assente."
        return throughput_summary

    compute_duty_cycle = (compute_cycles / float(total_cycles)) if total_cycles else 0.0
    compute_duty_cycle_while_busy = (compute_cycles / float(busy_cycles)) if busy_cycles else 0.0
    theoretical_peak_ops_per_cycle = float(architecture.pe_count * 2)
    theoretical_peak_tops = float(architecture.estimated_tops)

    throughput_summary.update(
        {
            "available": True,
            "target_frequency_mhz": round(float(architecture.target_frequency_mhz), 6),
            "architecture_pe_count": int(architecture.pe_count),
            "theoretical_peak_ops_per_cycle": round(theoretical_peak_ops_per_cycle, 6),
            "theoretical_peak_tops": round(theoretical_peak_tops, 6),
            "compute_duty_cycle": round(compute_duty_cycle, 6),
            "compute_duty_cycle_while_busy": round(compute_duty_cycle_while_busy, 6),
            "estimated_effective_ops_per_cycle": round(
                theoretical_peak_ops_per_cycle * compute_duty_cycle,
                6,
            ),
            "estimated_effective_tops": round(theoretical_peak_tops * compute_duty_cycle, 6),
        }
    )
    return throughput_summary


def _estimate_external_bandwidth_gb_per_s(
    transferred_bits: int,
    total_cycles: int,
    target_frequency_mhz: float,
) -> float:
    if total_cycles <= 0:
        return 0.0
    return round(
        (transferred_bits * target_frequency_mhz) / (8000.0 * total_cycles),
        6,
    )


def _estimate_peak_external_bandwidth_gb_per_s(
    peak_dma_bits_per_cycle: int,
    target_frequency_mhz: float,
) -> float:
    return round((peak_dma_bits_per_cycle * target_frequency_mhz) / 8000.0, 6)


def _estimate_bus_bandwidth_gb_per_s(architecture: ArchitectureCandidate) -> float:
    return round(
        (architecture.bus_width_bits * architecture.target_frequency_mhz) / 8000.0,
        6,
    )
