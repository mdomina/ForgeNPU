import math
from typing import List

from create_npu.models import ArchitectureCandidate, RequirementSpec


DEFAULT_CANDIDATE_IDS = ["balanced", "throughput_max", "efficiency"]


def generate_candidate_architectures(
    spec: RequirementSpec, max_candidates: int = 3
) -> List[ArchitectureCandidate]:
    candidate_ids = DEFAULT_CANDIDATE_IDS[: max(1, max_candidates)]
    return [plan_architecture(spec, candidate_id=candidate_id) for candidate_id in candidate_ids]


def plan_architecture(spec: RequirementSpec, candidate_id: str = "balanced") -> ArchitectureCandidate:
    throughput_tops = _normalize_throughput_to_tops(spec)
    target_frequency_mhz = _resolve_frequency(spec=spec, candidate_id=candidate_id)
    target_frequency_hz = target_frequency_mhz * 1_000_000.0
    macs_required = max(1, math.ceil((throughput_tops * 1_000_000_000_000.0) / (2.0 * target_frequency_hz)))

    family = _choose_family(spec)
    tile_edge = _choose_tile_edge(throughput_tops)
    logical_edge = max(tile_edge, math.ceil(math.sqrt(macs_required)))
    pe_rows = _round_up(logical_edge, tile_edge)
    pe_cols = _round_up(math.ceil(macs_required / float(pe_rows)), tile_edge)
    pe_count = pe_rows * pe_cols

    tile_count = math.ceil(pe_count / float(tile_edge * tile_edge))
    local_sram_kb_per_tile = _choose_local_sram(spec, candidate_id)
    global_buffer_mb = max(4, tile_count // 4)
    bus_width_bits = _choose_bus_width(throughput_tops, candidate_id)
    estimated_tops = (pe_count * 2.0 * target_frequency_hz) / 1_000_000_000_000.0
    estimated_power_watts = _estimate_power(
        pe_count=pe_count,
        target_frequency_mhz=target_frequency_mhz,
        bus_width_bits=bus_width_bits,
        candidate_id=candidate_id,
    )
    estimated_area_mm2 = _estimate_area(
        pe_count=pe_count,
        local_sram_kb_per_tile=local_sram_kb_per_tile,
        tile_count=tile_count,
        candidate_id=candidate_id,
    )

    modules = [
        "mac_unit",
        "processing_element",
        "systolic_tile",
        "scratchpad_controller",
        "tile_compute_unit",
        "dma_engine",
        "scheduler",
        "top_npu",
    ]
    if candidate_id == "throughput_max":
        modules.append("prefetch_buffer")
    if candidate_id == "efficiency":
        modules.append("clock_gating_unit")

    rationale = [
        f"Candidato valutato: {candidate_id}.",
        f"Famiglia architetturale scelta: {family}.",
        (
            "Numero di MAC stimato con formula throughput / (2 * frequenza), "
            f"pari a circa {macs_required} unita'."
        ),
        (
            f"Dimensionamento iniziale in mesh {pe_rows}x{pe_cols} "
            f"con tile {tile_edge}x{tile_edge}."
        ),
        (
            f"Stimato throughput teorico di {estimated_tops:.2f} TOPS "
            f"a {target_frequency_mhz:.0f} MHz."
        ),
        f"Potenza stimata iniziale: {estimated_power_watts:.1f} W.",
        f"Area stimata iniziale: {estimated_area_mm2:.1f} mm2.",
    ]

    if spec.batch_max > 1:
        rationale.append(
            f"Batch massimo {spec.batch_max}: aumento buffer locale e ampiezza del fabric."
        )

    if spec.power_budget_watts is not None:
        rationale.append(
            f"Budget di potenza rilevato: {spec.power_budget_watts:.0f} W."
        )

    rationale.append(_candidate_rationale(candidate_id))

    return ArchitectureCandidate(
        candidate_id=candidate_id,
        family=family,
        tile_rows=tile_edge,
        tile_cols=tile_edge,
        tile_count=tile_count,
        pe_rows=pe_rows,
        pe_cols=pe_cols,
        pe_count=pe_count,
        local_sram_kb_per_tile=local_sram_kb_per_tile,
        global_buffer_mb=global_buffer_mb,
        bus_width_bits=bus_width_bits,
        target_frequency_mhz=target_frequency_mhz,
        estimated_tops=estimated_tops,
        estimated_power_watts=estimated_power_watts,
        estimated_area_mm2=estimated_area_mm2,
        modules=modules,
        rationale=rationale,
    )


def _normalize_throughput_to_tops(spec: RequirementSpec) -> float:
    if spec.throughput_value is None:
        return 1.0
    return spec.throughput_value


def _choose_family(spec: RequirementSpec) -> str:
    if spec.workload_type == "transformer":
        return "tiled_systolic_transformer"
    if spec.workload_type == "convolution":
        return "weight_stationary_array"
    if spec.workload_type == "sparse_linear_algebra":
        return "sparse_pe_mesh"
    return "tiled_systolic_array"


def _choose_tile_edge(throughput_tops: float) -> int:
    if throughput_tops >= 500.0:
        return 64
    if throughput_tops >= 50.0:
        return 32
    if throughput_tops >= 5.0:
        return 16
    return 8


def _choose_local_sram(spec: RequirementSpec, candidate_id: str) -> int:
    base_kb = 256 if spec.numeric_precision.startswith("INT") else 512
    if spec.workload_type == "transformer":
        base_kb += 256
    if spec.batch_max > 8:
        base_kb += 256
    if candidate_id == "throughput_max":
        base_kb += 128
    if candidate_id == "efficiency":
        base_kb += 64
    return base_kb


def _choose_bus_width(throughput_tops: float, candidate_id: str) -> int:
    if throughput_tops >= 500.0:
        base_width = 2048
    elif throughput_tops >= 50.0:
        base_width = 1024
    elif throughput_tops >= 5.0:
        base_width = 512
    else:
        base_width = 256

    if candidate_id == "throughput_max":
        return base_width + 512
    if candidate_id == "efficiency":
        return max(256, base_width - 256)
    return base_width


def _resolve_frequency(spec: RequirementSpec, candidate_id: str) -> float:
    base_frequency = spec.target_frequency_mhz or 1000.0
    if candidate_id == "throughput_max":
        return round(base_frequency * 1.15, 2)
    if candidate_id == "efficiency":
        return round(base_frequency * 0.88, 2)
    return base_frequency


def _estimate_power(
    pe_count: int, target_frequency_mhz: float, bus_width_bits: int, candidate_id: str
) -> float:
    base_power = max(20.0, (pe_count * target_frequency_mhz) / 1_500_000.0 + (bus_width_bits / 128.0))
    if candidate_id == "throughput_max":
        return round(base_power * 1.18, 2)
    if candidate_id == "efficiency":
        return round(base_power * 0.78, 2)
    return round(base_power, 2)


def _estimate_area(
    pe_count: int, local_sram_kb_per_tile: int, tile_count: int, candidate_id: str
) -> float:
    base_area = (pe_count / 6500.0) + ((local_sram_kb_per_tile * tile_count) / 16384.0)
    if candidate_id == "throughput_max":
        return round(base_area * 1.12, 2)
    if candidate_id == "efficiency":
        return round(base_area * 0.93, 2)
    return round(base_area, 2)


def _candidate_rationale(candidate_id: str) -> str:
    if candidate_id == "throughput_max":
        return "Profilo throughput_max: privilegio frequenza e banda rispetto al costo energetico."
    if candidate_id == "efficiency":
        return "Profilo efficiency: riduco frequenza e banda per rientrare piu' facilmente nel budget di potenza."
    return "Profilo balanced: compromesso iniziale tra throughput, banda e consumi."


def _round_up(value: int, multiple: int) -> int:
    return int(math.ceil(value / float(multiple)) * multiple)
