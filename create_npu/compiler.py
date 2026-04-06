import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from create_npu.models import ArchitectureCandidate, RequirementSpec
from create_npu.workloads import resolve_dataflow_for_family


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
    estimated_mac_operations: int = 0
    problem_shape: Dict[str, Any] = field(default_factory=dict)
    operator_descriptors: List[Dict[str, Any]] = field(default_factory=list)
    mapping_plan: Dict[str, Any] = field(default_factory=dict)
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
    problem_shape = _problem_shape(
        spec=spec,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        active_tile_count=active_tile_count,
        slot_count=slot_count,
    )
    operator_descriptors = _operator_descriptors(spec=spec, problem_shape=problem_shape)
    mapping_plan = _mapping_plan(
        spec=spec,
        architecture=architecture,
        problem_shape=problem_shape,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        active_tile_count=active_tile_count,
    )
    slot_stride = _descriptor_stride(spec=spec, architecture=architecture)
    store_stride = _store_stride(spec=spec, store_burst_count=store_burst_count)
    activation_base_addr = 0
    weight_base_addr = _weight_base_addr(spec)
    result_base_addr = _result_base_addr(spec)
    activation_slots = _activation_slots(spec)
    weight_slots = _weight_slots(spec)
    clear_on_done = spec.execution_mode != "training"
    tiling_strategy = _tiling_strategy(spec=spec, mapping_plan=mapping_plan)
    operand_width_bits = 8 if spec.numeric_precision.startswith("INT8") else 16

    estimated_activation_bytes = slot_count * tile_rows * math.ceil(operand_width_bits / 8.0)
    estimated_weight_bytes = slot_count * tile_cols * math.ceil(operand_width_bits / 8.0)
    estimated_result_bytes = store_burst_count * tile_rows * 4
    estimated_mac_operations = sum(int(operator.get("macs", 0)) for operator in operator_descriptors)

    rationale = [
        f"Programma seed compilato per workload `{spec.workload_type}` con strategia `{tiling_strategy}`.",
        f"Slot DMA/load: {slot_count}, iterazioni load: {load_iterations}, iterazioni compute: {compute_iterations}.",
        f"Tile attivi nel programma seed: {active_tile_count}/{tile_count}.",
        (
            "Descrittori memoria compilati con base activation="
            f"{activation_base_addr}, weight={weight_base_addr}, result={result_base_addr}, "
            f"slot_stride={slot_stride}, store_stride={store_stride}."
        ),
        (
            "Shape workload compilata: "
            + ", ".join(f"{key}={value}" for key, value in problem_shape.items())
        ),
        (
            "Operatori compilati: "
            + ", ".join(str(operator.get("name", operator.get("op_type", "op"))) for operator in operator_descriptors)
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
    rationale.append(
        "Dataflow compilato: "
        f"{mapping_plan.get('dataflow', 'systolic')} "
        f"(output_stationary_i={int(mapping_plan.get('output_stationary_enabled', 0))}, "
        f"preload_en_i={int(mapping_plan.get('preload_enabled', 0))}, "
        f"transpose_inputs_i={int(mapping_plan.get('transpose_inputs', 0))})."
    )

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
        estimated_mac_operations=estimated_mac_operations,
        problem_shape=problem_shape,
        operator_descriptors=operator_descriptors,
        mapping_plan=mapping_plan,
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
        "output_stationary_i": int(program.mapping_plan.get("output_stationary_enabled", 0)),
        "preload_en_i": int(program.mapping_plan.get("preload_enabled", 0)),
        "transpose_inputs_i": int(program.mapping_plan.get("transpose_inputs", 0)),
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


def _tiling_strategy(spec: RequirementSpec, mapping_plan: Dict[str, Any]) -> str:
    dataflow = str(mapping_plan.get("dataflow", "systolic"))
    if spec.workload_type == "transformer":
        if dataflow == "output_stationary":
            return "output_stationary_sequence_blocking"
        return "sequence_blocking"
    if spec.workload_type == "convolution":
        if dataflow == "output_stationary":
            return "output_stationary_window"
        return "weight_stationary_window"
    if spec.workload_type == "sparse_linear_algebra":
        return "sparse_stream_compaction"
    if dataflow == "output_stationary":
        return "output_stationary_blocking"
    return "gemm_blocking"


def _problem_shape(
    spec: RequirementSpec,
    tile_rows: int,
    tile_cols: int,
    active_tile_count: int,
    slot_count: int,
) -> Dict[str, int]:
    batch = max(1, int(spec.batch_max))
    if spec.workload_type == "transformer":
        sequence_length = _align_up(int(spec.sequence_length or 1024), tile_rows)
        head_count = max(4, min(32, _align_up(active_tile_count * 8, 4)))
        head_dim = _align_up(max(tile_cols, 64), tile_cols)
        hidden_dim = _align_up(head_count * head_dim, tile_cols)
        projection_dim = _align_up(hidden_dim * 3, tile_cols)
        return {
            "batch": batch,
            "sequence_length": sequence_length,
            "head_count": head_count,
            "head_dim": head_dim,
            "hidden_dim": hidden_dim,
            "projection_dim": projection_dim,
        }
    if spec.workload_type == "convolution":
        kernel = max(1, int(spec.kernel_size or 3))
        input_height = 32 if kernel <= 3 else 16
        input_width = input_height
        output_height = max(1, input_height - kernel + 1)
        output_width = max(1, input_width - kernel + 1)
        input_channels = _align_up(max(tile_cols * max(1, slot_count), 16), tile_cols)
        output_channels = _align_up(max(input_channels * 2, tile_cols * max(2, active_tile_count)), tile_cols)
        return {
            "batch": batch,
            "input_height": input_height,
            "input_width": input_width,
            "input_channels": input_channels,
            "output_height": output_height,
            "output_width": output_width,
            "output_channels": output_channels,
            "kernel_height": kernel,
            "kernel_width": kernel,
            "stride": 1,
        }
    if spec.workload_type == "sparse_linear_algebra":
        density_percent = 50 if spec.sparsity_support == "structured" else 25
        m_dim = _align_up(tile_rows * max(2, batch), tile_rows)
        k_dim = _align_up(tile_cols * 8, tile_cols)
        n_dim = _align_up(tile_cols * max(2, active_tile_count * 2), tile_cols)
        nnz = max(tile_rows, int(m_dim * k_dim * density_percent / 100.0))
        return {
            "batch": batch,
            "m": m_dim,
            "k": k_dim,
            "n": n_dim,
            "density_percent": density_percent,
            "nnz": nnz,
        }
    return {
        "batch": batch,
        "m": _align_up(tile_rows * max(2, batch), tile_rows),
        "k": _align_up(tile_cols * max(4, slot_count * 2), tile_cols),
        "n": _align_up(tile_cols * max(2, active_tile_count * 2), tile_cols),
    }


def _operator_descriptors(
    spec: RequirementSpec,
    problem_shape: Dict[str, int],
) -> List[Dict[str, Any]]:
    if spec.workload_type == "transformer":
        batch = int(problem_shape["batch"])
        sequence_length = int(problem_shape["sequence_length"])
        head_count = int(problem_shape["head_count"])
        head_dim = int(problem_shape["head_dim"])
        hidden_dim = int(problem_shape["hidden_dim"])
        projection_dim = int(problem_shape["projection_dim"])
        token_count = batch * sequence_length
        return [
            {
                "name": "qkv_projection",
                "op_type": "gemm",
                "m": token_count,
                "k": hidden_dim,
                "n": projection_dim,
                "macs": token_count * hidden_dim * projection_dim,
            },
            {
                "name": "attention_scores",
                "op_type": "batched_gemm",
                "batch_groups": batch * head_count,
                "m": sequence_length,
                "k": head_dim,
                "n": sequence_length,
                "macs": batch * head_count * sequence_length * head_dim * sequence_length,
            },
            {
                "name": "attention_value_mix",
                "op_type": "batched_gemm",
                "batch_groups": batch * head_count,
                "m": sequence_length,
                "k": sequence_length,
                "n": head_dim,
                "macs": batch * head_count * sequence_length * sequence_length * head_dim,
            },
            {
                "name": "output_projection",
                "op_type": "gemm",
                "m": token_count,
                "k": hidden_dim,
                "n": hidden_dim,
                "macs": token_count * hidden_dim * hidden_dim,
            },
        ]
    if spec.workload_type == "convolution":
        batch = int(problem_shape["batch"])
        output_height = int(problem_shape["output_height"])
        output_width = int(problem_shape["output_width"])
        input_channels = int(problem_shape["input_channels"])
        output_channels = int(problem_shape["output_channels"])
        kernel_height = int(problem_shape["kernel_height"])
        kernel_width = int(problem_shape["kernel_width"])
        return [
            {
                "name": "conv2d_main",
                "op_type": "conv2d",
                "batch": batch,
                "input_height": int(problem_shape["input_height"]),
                "input_width": int(problem_shape["input_width"]),
                "input_channels": input_channels,
                "output_height": output_height,
                "output_width": output_width,
                "output_channels": output_channels,
                "kernel_height": kernel_height,
                "kernel_width": kernel_width,
                "stride": int(problem_shape["stride"]),
                "macs": (
                    batch
                    * output_height
                    * output_width
                    * output_channels
                    * input_channels
                    * kernel_height
                    * kernel_width
                ),
            }
        ]
    if spec.workload_type == "sparse_linear_algebra":
        return [
            {
                "name": "spmm_main",
                "op_type": "spmm",
                "m": int(problem_shape["m"]),
                "k": int(problem_shape["k"]),
                "n": int(problem_shape["n"]),
                "nnz": int(problem_shape["nnz"]),
                "density_percent": int(problem_shape["density_percent"]),
                "block_structure": "2:4" if spec.sparsity_support == "structured" else "unstructured",
                "macs": int(problem_shape["nnz"]) * int(problem_shape["n"]),
            }
        ]
    return [
        {
            "name": "gemm_main",
            "op_type": "gemm",
            "m": int(problem_shape["m"]),
            "k": int(problem_shape["k"]),
            "n": int(problem_shape["n"]),
            "macs": int(problem_shape["m"]) * int(problem_shape["k"]) * int(problem_shape["n"]),
        }
    ]


def _mapping_plan(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
    problem_shape: Dict[str, int],
    tile_rows: int,
    tile_cols: int,
    active_tile_count: int,
) -> Dict[str, Any]:
    resolved_dataflow = _resolved_mapping_dataflow(spec=spec, architecture=architecture)
    preload_enabled = int(resolved_dataflow == "output_stationary")
    transpose_inputs = int(
        resolved_dataflow == "output_stationary"
        and spec.workload_type in {"dense_gemm", "transformer"}
    )
    if spec.workload_type == "transformer":
        return {
            "dataflow": resolved_dataflow,
            "loop_order": ["batch", "sequence_block", "head", "k_block"],
            "sequence_block": tile_rows,
            "projection_block": tile_cols,
            "head_parallelism": min(int(problem_shape["head_count"]), active_tile_count),
            "active_tile_count": active_tile_count,
            "architecture_family": architecture.family,
            "output_stationary_enabled": int(resolved_dataflow == "output_stationary"),
            "preload_enabled": preload_enabled,
            "transpose_inputs": transpose_inputs,
        }
    if spec.workload_type == "convolution":
        loop_order = (
            ["batch", "spatial_block", "output_channel_block", "kernel"]
            if resolved_dataflow == "output_stationary"
            else ["batch", "output_channel_block", "spatial_block", "kernel"]
        )
        return {
            "dataflow": resolved_dataflow,
            "loop_order": loop_order,
            "output_channel_block": tile_cols * max(1, active_tile_count),
            "spatial_block": tile_rows,
            "kernel_window": int(problem_shape["kernel_height"]) * int(problem_shape["kernel_width"]),
            "active_tile_count": active_tile_count,
            "architecture_family": architecture.family,
            "output_stationary_enabled": int(resolved_dataflow == "output_stationary"),
            "preload_enabled": preload_enabled,
            "transpose_inputs": transpose_inputs,
        }
    if spec.workload_type == "sparse_linear_algebra":
        return {
            "dataflow": "sparse_streaming",
            "loop_order": ["row_block", "nnz_block", "col_block"],
            "row_block": tile_rows,
            "col_block": tile_cols * max(1, active_tile_count),
            "nnz_block": min(int(problem_shape["k"]), tile_cols * 4),
            "sparsity_mode": spec.sparsity_support,
            "active_tile_count": active_tile_count,
            "architecture_family": architecture.family,
            "output_stationary_enabled": 0,
            "preload_enabled": 0,
            "transpose_inputs": 0,
        }
    return {
        "dataflow": resolved_dataflow,
        "loop_order": ["m_block", "n_block", "k_block"],
        "m_block": tile_rows,
        "n_block": tile_cols * max(1, active_tile_count),
        "k_block": tile_cols,
        "active_tile_count": active_tile_count,
        "architecture_family": architecture.family,
        "output_stationary_enabled": int(resolved_dataflow == "output_stationary"),
        "preload_enabled": preload_enabled,
        "transpose_inputs": transpose_inputs,
    }


def _resolved_mapping_dataflow(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
) -> str:
    if spec.workload_type == "sparse_linear_algebra":
        return "sparse_streaming"
    resolved_dataflow = resolve_dataflow_for_family(architecture.family)
    if resolved_dataflow == "sparse":
        return "sparse_streaming"
    return resolved_dataflow


def _align_up(value: int, granularity: int) -> int:
    safe_granularity = max(1, granularity)
    safe_value = max(1, value)
    return int(math.ceil(safe_value / safe_granularity) * safe_granularity)


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
