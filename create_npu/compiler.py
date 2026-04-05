import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from create_npu.models import ArchitectureCandidate, RequirementSpec


@dataclass
class CompiledProgram:
    name: str
    workload_type: str
    execution_mode: str
    tiling_strategy: str
    tile_rows: int
    tile_cols: int
    tile_count: int
    active_tile_count: int
    tile_enable_mask: List[int]
    slot_count: int
    load_iterations: int
    compute_iterations: int
    store_burst_count: int
    activation_base_addr: int
    weight_base_addr: int
    result_base_addr: int
    slot_stride: int
    store_stride: int
    clear_on_done: bool
    activation_slots: List[List[int]]
    weight_slots: List[List[int]]
    estimated_activation_bytes: int
    estimated_weight_bytes: int
    estimated_result_bytes: int
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compile_seed_program(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
) -> CompiledProgram:
    tile_rows = max(1, int(architecture.tile_rows))
    tile_cols = max(1, int(architecture.tile_cols))
    tile_count = max(1, int(architecture.tile_count))
    active_tile_count = _active_tile_count(spec=spec, architecture=architecture)
    tile_enable_mask = [1 for _ in range(active_tile_count)] + [
        0 for _ in range(max(0, tile_count - active_tile_count))
    ]

    slot_count = _slot_count(spec)
    load_iterations = _load_iterations(spec)
    compute_iterations = _compute_iterations(spec)
    store_burst_count = _store_burst_count(spec=spec, tile_rows=tile_rows, active_tile_count=active_tile_count)
    slot_stride = _descriptor_stride(spec=spec, architecture=architecture)
    store_stride = _store_stride(spec=spec, store_burst_count=store_burst_count)
    activation_base_addr = 0
    weight_base_addr = _weight_base_addr(spec)
    result_base_addr = _result_base_addr(spec)
    activation_slots = _activation_slots(spec)
    weight_slots = _weight_slots(spec)
    clear_on_done = spec.execution_mode != "training"
    tiling_strategy = _tiling_strategy(spec)
    operand_width_bits = 8 if spec.numeric_precision.startswith("INT8") else 16

    estimated_activation_bytes = slot_count * tile_rows * math.ceil(operand_width_bits / 8.0)
    estimated_weight_bytes = slot_count * tile_cols * math.ceil(operand_width_bits / 8.0)
    estimated_result_bytes = store_burst_count * tile_rows * 4

    rationale = [
        f"Programma seed compilato per workload `{spec.workload_type}` con strategia `{tiling_strategy}`.",
        f"Slot DMA/load: {slot_count}, iterazioni load: {load_iterations}, iterazioni compute: {compute_iterations}.",
        f"Tile attivi nel programma seed: {active_tile_count}/{tile_count}.",
        (
            "Descrittori memoria compilati con base activation="
            f"{activation_base_addr}, weight={weight_base_addr}, result={result_base_addr}, "
            f"slot_stride={slot_stride}, store_stride={store_stride}."
        ),
    ]
    if spec.sequence_length is not None:
        rationale.append(f"Sequence length {spec.sequence_length} riflessa nel conteggio compute.")
    if spec.kernel_size is not None:
        rationale.append(f"Kernel size {spec.kernel_size}x{spec.kernel_size} riflessa nel tiling convolution.")
    if spec.batch_max > 1:
        rationale.append(f"Batch massimo {spec.batch_max} usato per aumentare parallelismo e burst di store.")
    if not clear_on_done:
        rationale.append("`clear_on_done` disabilitato per preservare stato tra iterazioni stile training.")

    return CompiledProgram(
        name=f"{spec.workload_type}_{spec.execution_mode}_seed_program",
        workload_type=spec.workload_type,
        execution_mode=spec.execution_mode,
        tiling_strategy=tiling_strategy,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        tile_count=tile_count,
        active_tile_count=active_tile_count,
        tile_enable_mask=tile_enable_mask,
        slot_count=slot_count,
        load_iterations=load_iterations,
        compute_iterations=compute_iterations,
        store_burst_count=store_burst_count,
        activation_base_addr=activation_base_addr,
        weight_base_addr=weight_base_addr,
        result_base_addr=result_base_addr,
        slot_stride=slot_stride,
        store_stride=store_stride,
        clear_on_done=clear_on_done,
        activation_slots=activation_slots,
        weight_slots=weight_slots,
        estimated_activation_bytes=estimated_activation_bytes,
        estimated_weight_bytes=estimated_weight_bytes,
        estimated_result_bytes=estimated_result_bytes,
        rationale=rationale,
    )


def compiled_program_seed_vectors(program: CompiledProgram) -> Dict[str, object]:
    activation_slots = list(program.activation_slots) + [[0, 0], [0, 0]]
    weight_slots = list(program.weight_slots) + [[0, 0], [0, 0]]
    return {
        "activation_slot0_i": [int(value) for value in activation_slots[0]],
        "activation_slot1_i": [int(value) for value in activation_slots[1]],
        "weight_slot0_i": [int(value) for value in weight_slots[0]],
        "weight_slot1_i": [int(value) for value in weight_slots[1]],
        "tile_enable_i": [int(value) for value in program.tile_enable_mask],
        "slot_count_i": int(program.slot_count),
        "load_iterations_i": int(program.load_iterations),
        "compute_iterations_i": int(program.compute_iterations),
        "activation_base_addr_i": int(program.activation_base_addr),
        "weight_base_addr_i": int(program.weight_base_addr),
        "result_base_addr_i": int(program.result_base_addr),
        "slot_stride_i": int(program.slot_stride),
        "store_stride_i": int(program.store_stride),
        "store_burst_count_i": int(program.store_burst_count),
        "clear_on_done_i": 1 if program.clear_on_done else 0,
    }


def _slot_count(spec: RequirementSpec) -> int:
    if spec.workload_type == "sparse_linear_algebra":
        return 1
    if spec.batch_max <= 1 and spec.sequence_length is None and spec.kernel_size is None:
        return 1
    return 2


def _load_iterations(spec: RequirementSpec) -> int:
    if spec.workload_type in {"transformer", "convolution"}:
        return 2
    if spec.batch_max > 1:
        return 2
    return 1


def _compute_iterations(spec: RequirementSpec) -> int:
    base = 1
    if spec.workload_type == "dense_gemm":
        base = 2
    elif spec.workload_type == "transformer":
        base = 2
        if spec.sequence_length is not None and spec.sequence_length >= 2048:
            base += 1
    elif spec.workload_type == "convolution":
        base = 2
        if spec.kernel_size is not None and spec.kernel_size >= 5:
            base += 1
    elif spec.workload_type == "sparse_linear_algebra":
        base = 1
    batch_bonus = 1 if spec.batch_max >= 8 else 0
    return max(1, min(15, base + batch_bonus))


def _store_burst_count(
    spec: RequirementSpec,
    tile_rows: int,
    active_tile_count: int,
) -> int:
    if tile_rows <= 1:
        return 1
    if spec.workload_type in {"transformer", "convolution"}:
        return 2
    if active_tile_count > 1 or spec.batch_max > 1:
        return 2
    return 1


def _descriptor_stride(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
) -> int:
    if spec.workload_type == "convolution":
        return max(1, min(7, spec.kernel_size or 1))
    if spec.workload_type == "transformer" and spec.sequence_length is not None:
        return max(1, min(7, math.ceil(spec.sequence_length / 2048.0)))
    return 1


def _store_stride(spec: RequirementSpec, store_burst_count: int) -> int:
    if spec.workload_type == "convolution" and store_burst_count > 1:
        return 2
    return 1


def _weight_base_addr(spec: RequirementSpec) -> int:
    if spec.workload_type == "convolution":
        return 1
    return 0


def _result_base_addr(spec: RequirementSpec) -> int:
    if spec.workload_type == "convolution":
        return 3
    return 2


def _active_tile_count(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
) -> int:
    tile_count = max(1, int(architecture.tile_count))
    if spec.workload_type in {"transformer", "convolution"} and tile_count > 1:
        return min(2, tile_count)
    if spec.optimization_priority == "throughput" and tile_count > 1:
        return min(2, tile_count)
    return 1


def _tiling_strategy(spec: RequirementSpec) -> str:
    if spec.workload_type == "transformer":
        return "sequence_blocking"
    if spec.workload_type == "convolution":
        return "weight_stationary_window"
    if spec.workload_type == "sparse_linear_algebra":
        return "sparse_stream_compaction"
    return "gemm_blocking"


def _activation_slots(spec: RequirementSpec) -> List[List[int]]:
    if spec.workload_type == "convolution":
        kernel = max(1, int(spec.kernel_size or 3))
        return [[kernel, 1], [1, kernel]]
    if spec.workload_type == "sparse_linear_algebra":
        return [[1, 0], [0, 2]]
    return [[1, 2], [3, 4]]


def _weight_slots(spec: RequirementSpec) -> List[List[int]]:
    if spec.workload_type == "convolution":
        kernel = max(1, int(spec.kernel_size or 3))
        return [[kernel + 2, kernel + 3], [kernel + 4, kernel + 5]]
    if spec.workload_type == "sparse_linear_algebra":
        return [[4, 0], [0, 5]]
    return [[5, 6], [7, 8]]
