import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from create_npu.memory_planner import plan_memory
from create_npu.models import ArchitectureCandidate, RequirementSpec
from create_npu.workloads import resolve_dataflow_for_family


@dataclass
class TensorDescriptor:
    """Describes a single tensor operand in the compiled program.

    Fields are intentionally rich enough to drive a future ``graph_mode`` or
    ``operator_mode`` execution without re-deriving them from the requirement.
    """

    name: str
    role: str                    # "activation_input" | "weight_input" | "output"
    operator_name: str           # operator that owns this tensor
    dtype: str                   # "INT8" | "INT16" | "INT32" | "FP16"
    shape: List[int]             # logical dimensions, e.g. [M, K]
    size_bytes: int              # total bytes = prod(shape) * itemsize
    base_addr: int               # SRAM base address
    stride_bytes: int            # leading-dimension stride in bytes
    layout: str                  # "row_major" | "col_major" | "tiled"
    requires_transpose: bool = False
    quantization_scale: float = 1.0
    producer_op: Optional[str] = None
    consumer_ops: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TiledLoop:
    """A single dimension in a tiled loop nest for gemm or convolution."""

    dim: str
    full_size: int
    tile_size: int
    iterations: int
    buffer_slot: str  # "A" for activation/input, "B" for weight, "C" for output


@dataclass
class TiledLoopNest:
    """Complete tiled loop nest with double-buffering and occupancy estimates."""

    op_type: str  # "gemm" or "conv2d"
    loops: List[TiledLoop]
    double_buffering_enabled: bool
    prefetch_distance: int  # 0 = no prefetch, 1 = one tile ahead
    activation_tiles_in_flight: int
    weight_tiles_in_flight: int
    output_tiles_in_flight: int
    total_tile_iterations: int
    estimated_compute_cycles: int
    estimated_memory_cycles: int
    estimated_overlap_cycles: int
    cluster_occupancy_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LoweredOp:
    """A single operation in the lowered hardware execution plan.

    Maps each high-level op (Gemm, Conv, Relu, Reduce, Concat, Slice, Pool)
    to a concrete hardware primitive and tracks tensor flow and fusability.
    """

    name: str
    op_type: str            # gemm | conv2d | spmm | relu | gelu | reduce | concat | slice | pool
    hardware_primitive: str # systolic_gemm | im2col_conv | sparse_gemm |
                            # vector_elementwise | vector_reduce | vector_pool | noop
    inputs: List[str]       # tensor descriptor names consumed by this op
    outputs: List[str]      # tensor descriptor names produced by this op
    fusable_with_next: bool = False  # can be fused into the next op kernel
    params: Dict[str, Any] = field(default_factory=dict)  # op-specific parameters

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DispatchEntry:
    """Single issued operation in the hardware dispatch schedule.

    Models the scoreboard slot for one lowered op: which engine it runs on,
    which prior ops it depends on (RAW hazards), whether a barrier is needed
    before it issues, and estimated issue/completion cycles.
    """

    op_name: str
    engine: str                    # "dma" | "compute" | "vector" | "noop"
    depends_on: List[str]          # ops whose outputs this op reads (RAW deps)
    barrier_before: bool           # must drain prior engine ops before issuing
    issue_cycle: int               # cycle when this op is issued
    completion_cycle: int          # estimated cycle when this op completes
    stall_cycles: int              # cycles stalled waiting for dependencies

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DispatchSchedule:
    """Complete scoreboard-based dispatch plan for a lowered program.

    Summarises issue order, barriers, hazards and cycle estimates for all
    ops in the lowered program.
    """

    entries: List[Dict[str, Any]]   # serialised DispatchEntry list
    total_issue_cycles: int
    total_completion_cycles: int
    barrier_count: int
    stall_cycles: int
    raw_hazard_count: int
    engine_utilization: Dict[str, float]   # engine -> utilization fraction

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    tiled_loop_nest: Dict[str, Any] = field(default_factory=dict)
    cluster_occupancy: Dict[str, Any] = field(default_factory=dict)
    tensor_descriptors: List[Dict[str, Any]] = field(default_factory=list)
    memory_plan: Dict[str, Any] = field(default_factory=dict)
    lowered_program: List[Dict[str, Any]] = field(default_factory=list)
    dispatch_schedule: Dict[str, Any] = field(default_factory=dict)
    gemm_accumulation: Dict[str, Any] = field(default_factory=dict)
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

    tiled_nest = _build_tiled_loop_nest(
        spec=spec,
        problem_shape=problem_shape,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        active_tile_count=active_tile_count,
        slot_count=slot_count,
        compute_iterations=compute_iterations,
    )
    cluster_occupancy = _build_cluster_occupancy(
        tiled_nest=tiled_nest,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        active_tile_count=active_tile_count,
        tile_count=tile_count,
    )

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
        f"Tiled loop nest: {tiled_nest.op_type} con {tiled_nest.total_tile_iterations} iterazioni totali, "
        f"double buffering={'si' if tiled_nest.double_buffering_enabled else 'no'}, "
        f"occupancy cluster={tiled_nest.cluster_occupancy_percent}%."
    )
    rationale.append(
        "Dataflow compilato: "
        f"{mapping_plan.get('dataflow', 'systolic')} "
        f"(output_stationary_i={int(mapping_plan.get('output_stationary_enabled', 0))}, "
        f"preload_en_i={int(mapping_plan.get('preload_enabled', 0))}, "
        f"transpose_inputs_i={int(mapping_plan.get('transpose_inputs', 0))})."
    )

    tensor_descs = _tensor_descriptors(
        spec=spec,
        problem_shape=problem_shape,
        activation_base_addr=activation_base_addr,
        weight_base_addr=weight_base_addr,
        result_base_addr=result_base_addr,
        operator_descriptors=operator_descriptors,
    )
    rationale.append(
        f"Tensor descriptors compilati: {len(tensor_descs)} descrittori "
        f"({sum(1 for d in tensor_descs if d['role'] == 'activation_input')} activation, "
        f"{sum(1 for d in tensor_descs if d['role'] == 'weight_input')} weight, "
        f"{sum(1 for d in tensor_descs if d['role'] == 'output')} output)."
    )

    memory_plan = plan_memory(
        tensor_descriptors=tensor_descs,
        operator_descriptors=operator_descriptors,
    )
    rationale.append(
        f"Memory plan: peak SRAM senza riuso={memory_plan.get('peak_sram_bytes_no_reuse', 0)} B, "
        f"con riuso={memory_plan.get('peak_sram_bytes_with_reuse', 0)} B "
        f"(risparmio {memory_plan.get('reuse_savings_pct', 0)}%)."
    )

    lowered_program = _lower_operators(
        spec=spec,
        operator_descriptors=operator_descriptors,
        problem_shape=problem_shape,
    )
    fusable_count = sum(1 for op in lowered_program if op.get("fusable_with_next"))
    rationale.append(
        f"Lowered program: {len(lowered_program)} op "
        f"({', '.join(sorted({op['op_type'] for op in lowered_program}))}), "
        f"{fusable_count} fusable."
    )

    dispatch_schedule = _build_dispatch_schedule(lowered_program)
    rationale.append(
        f"Dispatch schedule: {dispatch_schedule.get('total_issue_cycles', 0)} issue, "
        f"{dispatch_schedule.get('barrier_count', 0)} barrier, "
        f"{dispatch_schedule.get('raw_hazard_count', 0)} RAW hazard, "
        f"{dispatch_schedule.get('stall_cycles', 0)} stall cycles."
    )
    gemm_accumulation = _gemm_accumulation_plan(
        spec=spec,
        problem_shape=problem_shape,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        active_tile_count=active_tile_count,
        store_burst_count=store_burst_count,
        lowered_program=lowered_program,
    )
    rationale.append(
        "GEMM accumulation path: "
        f"controller={gemm_accumulation.get('controller_module', 'gemm_ctrl')}, "
        f"k_tiles={gemm_accumulation.get('k_tile_count', 0)}, "
        f"persistent_psums={'si' if gemm_accumulation.get('partial_sum_persistence') else 'no'}, "
        f"deferred_writeback={'si' if gemm_accumulation.get('deferred_writeback') else 'no'}."
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
        tiled_loop_nest=tiled_nest.to_dict(),
        cluster_occupancy=cluster_occupancy,
        tensor_descriptors=tensor_descs,
        memory_plan=memory_plan,
        lowered_program=lowered_program,
        dispatch_schedule=dispatch_schedule,
        gemm_accumulation=gemm_accumulation,
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


def _build_tiled_loop_nest(
    spec: RequirementSpec,
    problem_shape: Dict[str, int],
    tile_rows: int,
    tile_cols: int,
    active_tile_count: int,
    slot_count: int,
    compute_iterations: int,
) -> TiledLoopNest:
    """Build a tiled loop nest for gemm or convolution with double-buffering."""
    if spec.workload_type == "convolution":
        return _build_conv_tiled_loops(
            problem_shape=problem_shape,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            active_tile_count=active_tile_count,
            slot_count=slot_count,
            compute_iterations=compute_iterations,
        )
    return _build_gemm_tiled_loops(
        problem_shape=problem_shape,
        tile_rows=tile_rows,
        tile_cols=tile_cols,
        active_tile_count=active_tile_count,
        slot_count=slot_count,
        compute_iterations=compute_iterations,
    )


def _build_gemm_tiled_loops(
    problem_shape: Dict[str, int],
    tile_rows: int,
    tile_cols: int,
    active_tile_count: int,
    slot_count: int,
    compute_iterations: int,
) -> TiledLoopNest:
    m = int(problem_shape.get("m", tile_rows))
    k = int(problem_shape.get("k", tile_cols))
    n = int(problem_shape.get("n", tile_cols))
    batch = int(problem_shape.get("batch", 1))

    m_tile = tile_rows
    k_tile = tile_cols
    n_tile = tile_cols * max(1, active_tile_count)

    m_iters = max(1, math.ceil(m / m_tile))
    k_iters = max(1, math.ceil(k / k_tile))
    n_iters = max(1, math.ceil(n / n_tile))

    loops = [
        TiledLoop(dim="batch", full_size=batch, tile_size=1, iterations=batch, buffer_slot="A"),
        TiledLoop(dim="m", full_size=m, tile_size=m_tile, iterations=m_iters, buffer_slot="C"),
        TiledLoop(dim="n", full_size=n, tile_size=n_tile, iterations=n_iters, buffer_slot="C"),
        TiledLoop(dim="k", full_size=k, tile_size=k_tile, iterations=k_iters, buffer_slot="A"),
    ]

    total_tile_iterations = batch * m_iters * n_iters * k_iters
    double_buffering = slot_count >= 2
    prefetch_distance = 1 if double_buffering else 0

    act_in_flight = 2 if double_buffering else 1
    wgt_in_flight = 2 if double_buffering else 1
    out_in_flight = 1

    compute_cycles = total_tile_iterations * m_tile * k_tile * compute_iterations
    memory_cycles = total_tile_iterations * (m_tile + k_tile + n_tile)
    overlap = min(memory_cycles, compute_cycles) if double_buffering else 0
    effective_cycles = max(1, compute_cycles + memory_cycles - overlap)

    peak_pe_active = m_tile * min(n_tile, tile_cols)
    total_pe = tile_rows * tile_cols * max(1, active_tile_count)
    occupancy = min(100.0, 100.0 * peak_pe_active * active_tile_count / max(1, total_pe))
    occupancy = round(occupancy * compute_cycles / max(1, effective_cycles), 1)

    return TiledLoopNest(
        op_type="gemm",
        loops=loops,
        double_buffering_enabled=double_buffering,
        prefetch_distance=prefetch_distance,
        activation_tiles_in_flight=act_in_flight,
        weight_tiles_in_flight=wgt_in_flight,
        output_tiles_in_flight=out_in_flight,
        total_tile_iterations=total_tile_iterations,
        estimated_compute_cycles=compute_cycles,
        estimated_memory_cycles=memory_cycles,
        estimated_overlap_cycles=overlap,
        cluster_occupancy_percent=occupancy,
    )


def _build_conv_tiled_loops(
    problem_shape: Dict[str, int],
    tile_rows: int,
    tile_cols: int,
    active_tile_count: int,
    slot_count: int,
    compute_iterations: int,
) -> TiledLoopNest:
    batch = int(problem_shape.get("batch", 1))
    oh = int(problem_shape.get("output_height", 1))
    ow = int(problem_shape.get("output_width", 1))
    ic = int(problem_shape.get("input_channels", tile_cols))
    oc = int(problem_shape.get("output_channels", tile_cols))
    kh = int(problem_shape.get("kernel_height", 3))
    kw = int(problem_shape.get("kernel_width", 3))

    spatial_tile = tile_rows
    oc_tile = tile_cols * max(1, active_tile_count)
    ic_tile = tile_cols

    spatial_total = oh * ow
    spatial_iters = max(1, math.ceil(spatial_total / spatial_tile))
    oc_iters = max(1, math.ceil(oc / oc_tile))
    ic_iters = max(1, math.ceil(ic / ic_tile))
    kernel_iters = kh * kw

    loops = [
        TiledLoop(dim="batch", full_size=batch, tile_size=1, iterations=batch, buffer_slot="A"),
        TiledLoop(dim="spatial", full_size=spatial_total, tile_size=spatial_tile, iterations=spatial_iters, buffer_slot="C"),
        TiledLoop(dim="oc", full_size=oc, tile_size=oc_tile, iterations=oc_iters, buffer_slot="C"),
        TiledLoop(dim="ic", full_size=ic, tile_size=ic_tile, iterations=ic_iters, buffer_slot="A"),
        TiledLoop(dim="kernel", full_size=kh * kw, tile_size=1, iterations=kernel_iters, buffer_slot="B"),
    ]

    total_tile_iterations = batch * spatial_iters * oc_iters * ic_iters * kernel_iters
    double_buffering = slot_count >= 2
    prefetch_distance = 1 if double_buffering else 0

    act_in_flight = 2 if double_buffering else 1
    wgt_in_flight = 2 if double_buffering else 1
    out_in_flight = 1

    compute_cycles = total_tile_iterations * spatial_tile * ic_tile * compute_iterations
    memory_cycles = total_tile_iterations * (spatial_tile + ic_tile + oc_tile)
    overlap = min(memory_cycles, compute_cycles) if double_buffering else 0
    effective_cycles = max(1, compute_cycles + memory_cycles - overlap)

    peak_pe_active = spatial_tile * min(oc_tile, tile_cols)
    total_pe = tile_rows * tile_cols * max(1, active_tile_count)
    occupancy = min(100.0, 100.0 * peak_pe_active * active_tile_count / max(1, total_pe))
    occupancy = round(occupancy * compute_cycles / max(1, effective_cycles), 1)

    return TiledLoopNest(
        op_type="conv2d",
        loops=loops,
        double_buffering_enabled=double_buffering,
        prefetch_distance=prefetch_distance,
        activation_tiles_in_flight=act_in_flight,
        weight_tiles_in_flight=wgt_in_flight,
        output_tiles_in_flight=out_in_flight,
        total_tile_iterations=total_tile_iterations,
        estimated_compute_cycles=compute_cycles,
        estimated_memory_cycles=memory_cycles,
        estimated_overlap_cycles=overlap,
        cluster_occupancy_percent=occupancy,
    )


def _build_cluster_occupancy(
    tiled_nest: TiledLoopNest,
    tile_rows: int,
    tile_cols: int,
    active_tile_count: int,
    tile_count: int,
) -> Dict[str, Any]:
    total_pe = tile_rows * tile_cols * max(1, tile_count)
    active_pe = tile_rows * tile_cols * max(1, active_tile_count)
    spatial_utilization = round(100.0 * active_pe / max(1, total_pe), 1)
    return {
        "total_pe": total_pe,
        "active_pe": active_pe,
        "spatial_utilization_percent": spatial_utilization,
        "compute_bound_occupancy_percent": tiled_nest.cluster_occupancy_percent,
        "double_buffering_enabled": tiled_nest.double_buffering_enabled,
        "estimated_compute_cycles": tiled_nest.estimated_compute_cycles,
        "estimated_memory_cycles": tiled_nest.estimated_memory_cycles,
        "estimated_overlap_cycles": tiled_nest.estimated_overlap_cycles,
        "effective_cycles": max(
            1,
            tiled_nest.estimated_compute_cycles
            + tiled_nest.estimated_memory_cycles
            - tiled_nest.estimated_overlap_cycles,
        ),
        "memory_compute_ratio": round(
            tiled_nest.estimated_memory_cycles / max(1, tiled_nest.estimated_compute_cycles), 3
        ),
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


def _build_dispatch_schedule(
    lowered_program: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build a scoreboard-based dispatch schedule for the lowered program.

    Algorithm
    ---------
    1. Assign each op to an engine based on its hardware_primitive.
    2. Detect RAW hazards: op B has a RAW dep on op A if any of B's inputs
       is produced by A (i.e. appears in A's outputs).
    3. Place a barrier_before flag when B's engine differs from A's engine
       and there is a RAW dependency.
    4. Estimate issue_cycle: max(prior_completion_cycles_of_deps) + 1.
    5. Estimate completion_cycle: issue_cycle + engine_latency_cycles.
    6. Compute stall_cycles = issue_cycle - (prev_issue_cycle + 1) if > 0.
    """
    _PRIMITIVE_TO_ENGINE: Dict[str, str] = {
        "systolic_gemm": "compute",
        "im2col_conv": "compute",
        "sparse_gemm": "compute",
        "vector_elementwise": "vector",
        "vector_reduce": "vector",
        "vector_pool": "vector",
        "noop": "noop",
    }
    _ENGINE_LATENCY: Dict[str, int] = {
        "compute": 4,
        "vector": 1,
        "noop": 0,
        "dma": 2,
    }

    # Map output tensor name → op name that produces it
    produced_by: Dict[str, str] = {}
    for op in lowered_program:
        for out in op.get("outputs", []):
            produced_by[out] = op["name"]

    entries: List[Dict[str, Any]] = []
    # Track latest completion cycle per op name
    completion_of: Dict[str, int] = {}
    prev_engine = ""
    current_cycle = 0

    engine_busy_cycles: Dict[str, int] = {}

    for op in lowered_program:
        primitive = op.get("hardware_primitive", "noop")
        engine = _PRIMITIVE_TO_ENGINE.get(primitive, "noop")
        latency = _ENGINE_LATENCY.get(engine, 1)

        # RAW dependencies
        raw_deps: List[str] = []
        for inp in op.get("inputs", []):
            producer = produced_by.get(inp)
            if producer is not None:
                raw_deps.append(producer)

        # Earliest possible issue: after all dependencies complete
        earliest = current_cycle
        for dep in raw_deps:
            dep_completion = completion_of.get(dep, 0)
            if dep_completion >= earliest:
                earliest = dep_completion + 1

        # Barrier when engine changes AND there is a RAW dep from a different engine
        barrier = False
        if raw_deps and prev_engine and prev_engine != engine and prev_engine != "noop":
            barrier = True
            # Barrier drains all in-flight compute ops: advance to latest completion
            max_completion = max(completion_of.values(), default=0)
            if max_completion + 1 > earliest:
                earliest = max_completion + 1

        stall = max(0, earliest - current_cycle)
        issue_cycle = earliest
        completion_cycle = issue_cycle + latency
        current_cycle = issue_cycle + 1

        engine_busy_cycles[engine] = engine_busy_cycles.get(engine, 0) + latency
        completion_of[op["name"]] = completion_cycle

        entries.append(
            DispatchEntry(
                op_name=op["name"],
                engine=engine,
                depends_on=raw_deps,
                barrier_before=barrier,
                issue_cycle=issue_cycle,
                completion_cycle=completion_cycle,
                stall_cycles=stall,
            ).to_dict()
        )
        if engine != "noop":
            prev_engine = engine

    total_issue = len(entries)
    total_completion = max((e["completion_cycle"] for e in entries), default=0)
    barrier_count = sum(1 for e in entries if e["barrier_before"])
    total_stall = sum(e["stall_cycles"] for e in entries)
    raw_hazard_count = sum(1 for e in entries if e["depends_on"])

    # Utilization: busy / total_completion
    util: Dict[str, float] = {}
    for eng, busy in engine_busy_cycles.items():
        if eng == "noop":
            continue
        util[eng] = round(busy / max(1, total_completion), 3)

    return DispatchSchedule(
        entries=entries,
        total_issue_cycles=total_issue,
        total_completion_cycles=total_completion,
        barrier_count=barrier_count,
        stall_cycles=total_stall,
        raw_hazard_count=raw_hazard_count,
        engine_utilization=util,
    ).to_dict()


def _lower_operators(
    spec: RequirementSpec,
    operator_descriptors: List[Dict[str, Any]],
    problem_shape: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Build the lowered operation sequence for the compiled program.

    Translates each primary compute op to a hardware primitive and inserts
    elementwise (Relu/Gelu), reduction (softmax), pooling (Pool), concat
    and slice ops where they naturally appear in the workload dataflow.

    Supported op_types: gemm, conv2d, spmm, relu, gelu, reduce, concat,
    slice, pool.
    Supported hardware_primitives: systolic_gemm, im2col_conv, sparse_gemm,
    vector_elementwise, vector_reduce, vector_pool, noop.
    """
    if spec.workload_type == "transformer":
        return _lower_transformer(problem_shape)
    if spec.workload_type == "convolution":
        return _lower_convolution(problem_shape)
    if spec.workload_type == "sparse_linear_algebra":
        return _lower_sparse(problem_shape)
    return _lower_gemm(problem_shape)


def _lower_gemm(problem_shape: Dict[str, int]) -> List[Dict[str, Any]]:
    """dense_gemm lowering: Gemm → Relu (fused)."""
    m = int(problem_shape.get("m", 1))
    k = int(problem_shape.get("k", 1))
    n = int(problem_shape.get("n", 1))
    return [
        LoweredOp(
            name="gemm_main",
            op_type="gemm",
            hardware_primitive="systolic_gemm",
            inputs=["gemm_main_activation_in", "gemm_main_weight_in"],
            outputs=["gemm_main_result_out"],
            fusable_with_next=True,
            params={"m": m, "k": k, "n": n},
        ).to_dict(),
        LoweredOp(
            name="relu_post_gemm",
            op_type="relu",
            hardware_primitive="vector_elementwise",
            inputs=["gemm_main_result_out"],
            outputs=["relu_post_gemm_out"],
            fusable_with_next=False,
            params={"shape": [m, n]},
        ).to_dict(),
    ]


def _gemm_accumulation_plan(
    spec: RequirementSpec,
    problem_shape: Dict[str, int],
    tile_rows: int,
    tile_cols: int,
    active_tile_count: int,
    store_burst_count: int,
    lowered_program: List[Dict[str, Any]],
) -> Dict[str, Any]:
    reduction_dim = _reduction_dim_for_accumulation(
        spec=spec,
        problem_shape=problem_shape,
        lowered_program=lowered_program,
    )
    k_tile_width = max(1, tile_cols)
    k_tile_count = max(1, math.ceil(reduction_dim / float(k_tile_width)))
    output_tile_cols = max(1, tile_cols * active_tile_count)
    return {
        "enabled": True,
        "controller_module": "gemm_ctrl",
        "accumulator_storage": "accumulator_buffer",
        "reduction_dim": reduction_dim,
        "k_tile_width": k_tile_width,
        "k_tile_count": k_tile_count,
        "partial_sum_persistence": k_tile_count > 1,
        "deferred_writeback": k_tile_count > 1,
        "writeback_policy": "explicit_final_k_tile" if k_tile_count > 1 else "per_output_tile",
        "output_tile_rows": max(1, tile_rows),
        "output_tile_cols": output_tile_cols,
        "accumulator_depth_tiles": max(1, store_burst_count),
    }


def _reduction_dim_for_accumulation(
    spec: RequirementSpec,
    problem_shape: Dict[str, int],
    lowered_program: List[Dict[str, Any]],
) -> int:
    for op in lowered_program:
        params = op.get("params", {})
        if "k" in params:
            return max(1, int(params["k"]))
        if op.get("op_type") == "conv2d":
            input_channels = int(problem_shape.get("input_channels", 1))
            kernel_height = int(problem_shape.get("kernel_height", 1))
            kernel_width = int(problem_shape.get("kernel_width", 1))
            return max(1, input_channels * kernel_height * kernel_width)
    if spec.workload_type == "transformer":
        return max(1, int(problem_shape.get("hidden_dim", 1)))
    return max(1, int(problem_shape.get("k", problem_shape.get("input_channels", 1))))


def _lower_convolution(problem_shape: Dict[str, int]) -> List[Dict[str, Any]]:
    """convolution lowering: Conv2d (im2col) → Relu (fused) → Pool."""
    oh = int(problem_shape.get("output_height", 1))
    ow = int(problem_shape.get("output_width", 1))
    oc = int(problem_shape.get("output_channels", 1))
    ic = int(problem_shape.get("input_channels", 1))
    kh = int(problem_shape.get("kernel_height", 1))
    kw = int(problem_shape.get("kernel_width", 1))
    pool_size = 2
    pool_oh = max(1, oh // pool_size)
    pool_ow = max(1, ow // pool_size)
    return [
        LoweredOp(
            name="conv2d_main",
            op_type="conv2d",
            hardware_primitive="im2col_conv",
            inputs=["conv2d_main_activation_in", "conv2d_main_weight_in"],
            outputs=["conv2d_main_result_out"],
            fusable_with_next=True,
            params={
                "output_height": oh, "output_width": ow, "output_channels": oc,
                "input_channels": ic, "kernel_height": kh, "kernel_width": kw,
                "via_im2col": True,
            },
        ).to_dict(),
        LoweredOp(
            name="relu_post_conv",
            op_type="relu",
            hardware_primitive="vector_elementwise",
            inputs=["conv2d_main_result_out"],
            outputs=["relu_post_conv_out"],
            fusable_with_next=True,
            params={"shape": [oh * ow, oc]},
        ).to_dict(),
        LoweredOp(
            name="pool_post_relu",
            op_type="pool",
            hardware_primitive="vector_pool",
            inputs=["relu_post_conv_out"],
            outputs=["pool_post_relu_out"],
            fusable_with_next=False,
            params={
                "pool_type": "max",
                "pool_size": pool_size,
                "input_shape": [oh, ow, oc],
                "output_shape": [pool_oh, pool_ow, oc],
            },
        ).to_dict(),
    ]


def _lower_transformer(problem_shape: Dict[str, int]) -> List[Dict[str, Any]]:
    """transformer lowering: QKV Gemm → Slice(Q,K,V) → Attn Gemm → Softmax
    Reduce → Value Gemm → Concat heads → Output Gemm → GeLU."""
    batch = int(problem_shape.get("batch", 1))
    seq = int(problem_shape.get("sequence_length", 1))
    hidden = int(problem_shape.get("hidden_dim", 1))
    head_count = int(problem_shape.get("head_count", 1))
    head_dim = int(problem_shape.get("head_dim", 1))
    proj = int(problem_shape.get("projection_dim", 1))
    token_count = batch * seq
    return [
        LoweredOp(
            name="qkv_projection",
            op_type="gemm",
            hardware_primitive="systolic_gemm",
            inputs=["activation_in", "weight_in"],
            outputs=["qkv_combined"],
            fusable_with_next=True,
            params={"m": token_count, "k": hidden, "n": proj},
        ).to_dict(),
        LoweredOp(
            name="slice_q",
            op_type="slice",
            hardware_primitive="noop",
            inputs=["qkv_combined"],
            outputs=["q_slice"],
            fusable_with_next=False,
            params={"axis": 1, "start": 0, "end": hidden, "shape": [token_count, hidden]},
        ).to_dict(),
        LoweredOp(
            name="slice_k",
            op_type="slice",
            hardware_primitive="noop",
            inputs=["qkv_combined"],
            outputs=["k_slice"],
            fusable_with_next=False,
            params={"axis": 1, "start": hidden, "end": 2 * hidden, "shape": [token_count, hidden]},
        ).to_dict(),
        LoweredOp(
            name="slice_v",
            op_type="slice",
            hardware_primitive="noop",
            inputs=["qkv_combined"],
            outputs=["v_slice"],
            fusable_with_next=False,
            params={"axis": 1, "start": 2 * hidden, "end": proj, "shape": [token_count, head_dim]},
        ).to_dict(),
        LoweredOp(
            name="attention_scores",
            op_type="gemm",
            hardware_primitive="systolic_gemm",
            inputs=["q_slice", "k_slice"],
            outputs=["attn_scores_out"],
            fusable_with_next=True,
            params={"m": seq, "k": head_dim, "n": seq, "batch_groups": batch * head_count},
        ).to_dict(),
        LoweredOp(
            name="softmax_reduce",
            op_type="reduce",
            hardware_primitive="vector_reduce",
            inputs=["attn_scores_out"],
            outputs=["softmax_out"],
            fusable_with_next=True,
            params={"op": "softmax", "axis": -1, "shape": [batch * head_count, seq, seq]},
        ).to_dict(),
        LoweredOp(
            name="attention_value_mix",
            op_type="gemm",
            hardware_primitive="systolic_gemm",
            inputs=["softmax_out", "v_slice"],
            outputs=["attn_value_out"],
            fusable_with_next=True,
            params={"m": seq, "k": seq, "n": head_dim, "batch_groups": batch * head_count},
        ).to_dict(),
        LoweredOp(
            name="concat_heads",
            op_type="concat",
            hardware_primitive="noop",
            inputs=["attn_value_out"],
            outputs=["multi_head_out"],
            fusable_with_next=True,
            params={"axis": -1, "head_count": head_count, "output_shape": [token_count, hidden]},
        ).to_dict(),
        LoweredOp(
            name="output_projection",
            op_type="gemm",
            hardware_primitive="systolic_gemm",
            inputs=["multi_head_out", "weight_in"],
            outputs=["output_proj_out"],
            fusable_with_next=True,
            params={"m": token_count, "k": hidden, "n": hidden},
        ).to_dict(),
        LoweredOp(
            name="gelu_output",
            op_type="gelu",
            hardware_primitive="vector_elementwise",
            inputs=["output_proj_out"],
            outputs=["gelu_out"],
            fusable_with_next=False,
            params={"shape": [token_count, hidden]},
        ).to_dict(),
    ]


def _lower_sparse(problem_shape: Dict[str, int]) -> List[Dict[str, Any]]:
    """sparse_linear_algebra lowering: SpMM → Relu (fused)."""
    m = int(problem_shape.get("m", 1))
    nnz = int(problem_shape.get("nnz", 1))
    n = int(problem_shape.get("n", 1))
    return [
        LoweredOp(
            name="spmm_main",
            op_type="spmm",
            hardware_primitive="sparse_gemm",
            inputs=["spmm_main_activation_in", "spmm_main_weight_in"],
            outputs=["spmm_main_result_out"],
            fusable_with_next=True,
            params={"m": m, "nnz": nnz, "n": n},
        ).to_dict(),
        LoweredOp(
            name="relu_post_spmm",
            op_type="relu",
            hardware_primitive="vector_elementwise",
            inputs=["spmm_main_result_out"],
            outputs=["relu_post_spmm_out"],
            fusable_with_next=False,
            params={"shape": [m, n]},
        ).to_dict(),
    ]


def _tensor_descriptors(
    spec: RequirementSpec,
    problem_shape: Dict[str, int],
    activation_base_addr: int,
    weight_base_addr: int,
    result_base_addr: int,
    operator_descriptors: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build TensorDescriptor records for all operands in the compiled program.

    Produces one activation, one weight and one output descriptor for the seed
    LOAD/COMPUTE/STORE program, plus per-operator descriptors drawn from
    *operator_descriptors*.  All descriptors are returned as plain dicts so
    they serialise cleanly to JSON.
    """
    bytes_per_element = 1 if spec.numeric_precision.startswith("INT8") else 2
    result_bytes_per_element = 4  # INT32 accumulator

    def _itemsize(dtype: str) -> int:
        return 1 if dtype in ("INT8", "UINT8") else (2 if dtype in ("INT16", "FP16") else 4)

    descriptors: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Seed-level descriptors  (one activation, one weight, one output)
    # ------------------------------------------------------------------
    if spec.workload_type == "transformer":
        m = int(problem_shape.get("sequence_length", 1))
        k = int(problem_shape.get("hidden_dim", 1))
        n = int(problem_shape.get("projection_dim", 1))
    elif spec.workload_type == "convolution":
        oh = int(problem_shape.get("output_height", 1))
        ow = int(problem_shape.get("output_width", 1))
        ic = int(problem_shape.get("input_channels", 1))
        kh = int(problem_shape.get("kernel_height", 1))
        kw = int(problem_shape.get("kernel_width", 1))
        oc = int(problem_shape.get("output_channels", 1))
        m = oh * ow
        k = ic * kh * kw   # im2col K dimension
        n = oc
    elif spec.workload_type == "sparse_linear_algebra":
        m = int(problem_shape.get("m", 1))
        k = int(problem_shape.get("nnz", 1))
        n = int(problem_shape.get("n", 1))
    else:
        m = int(problem_shape.get("m", 1))
        k = int(problem_shape.get("k", 1))
        n = int(problem_shape.get("n", 1))

    op_name = operator_descriptors[0].get("name", "seed_op") if operator_descriptors else "seed_op"
    act_layout = "im2col_row_major" if spec.workload_type == "convolution" else "row_major"

    act_size = m * k * bytes_per_element
    wgt_size = k * n * bytes_per_element
    res_size = m * n * result_bytes_per_element

    descriptors.append(
        TensorDescriptor(
            name="activation_in",
            role="activation_input",
            operator_name=op_name,
            dtype=spec.numeric_precision,
            shape=[m, k],
            size_bytes=act_size,
            base_addr=activation_base_addr,
            stride_bytes=k * bytes_per_element,
            layout=act_layout,
            requires_transpose=False,
            quantization_scale=1.0,
            producer_op=None,
            consumer_ops=[op_name],
        ).to_dict()
    )
    descriptors.append(
        TensorDescriptor(
            name="weight_in",
            role="weight_input",
            operator_name=op_name,
            dtype=spec.numeric_precision,
            shape=[k, n],
            size_bytes=wgt_size,
            base_addr=weight_base_addr,
            stride_bytes=n * bytes_per_element,
            layout="row_major",
            requires_transpose=False,
            quantization_scale=1.0,
            producer_op=None,
            consumer_ops=[op_name],
        ).to_dict()
    )
    descriptors.append(
        TensorDescriptor(
            name="result_out",
            role="output",
            operator_name=op_name,
            dtype="INT32",
            shape=[m, n],
            size_bytes=res_size,
            base_addr=result_base_addr,
            stride_bytes=n * result_bytes_per_element,
            layout="row_major",
            requires_transpose=False,
            quantization_scale=1.0,
            producer_op=op_name,
            consumer_ops=[],
        ).to_dict()
    )

    # ------------------------------------------------------------------
    # Per-operator descriptors (one input-pair + output per operator)
    # ------------------------------------------------------------------
    for op in operator_descriptors:
        op_type = str(op.get("op_type", "gemm"))
        o_name = str(op.get("name", op_type))
        op_m = int(op.get("m", m))
        op_k = int(op.get("k", k))
        op_n = int(op.get("n", n))
        if op_type == "conv2d":
            op_m = int(op.get("output_height", 1)) * int(op.get("output_width", 1))
            op_k = (
                int(op.get("input_channels", 1))
                * int(op.get("kernel_height", 1))
                * int(op.get("kernel_width", 1))
            )
            op_n = int(op.get("output_channels", 1))
        elif op_type == "spmm":
            op_m = int(op.get("m", 1))
            op_k = int(op.get("nnz", 1))
            op_n = int(op.get("n", 1))
        op_act_size = op_m * op_k * bytes_per_element
        op_wgt_size = op_k * op_n * bytes_per_element
        op_res_size = op_m * op_n * result_bytes_per_element
        descriptors.append(
            TensorDescriptor(
                name=f"{o_name}_activation_in",
                role="activation_input",
                operator_name=o_name,
                dtype=spec.numeric_precision,
                shape=[op_m, op_k],
                size_bytes=op_act_size,
                base_addr=activation_base_addr,
                stride_bytes=op_k * bytes_per_element,
                layout="row_major" if op_type != "conv2d" else "im2col_row_major",
                requires_transpose=False,
                quantization_scale=1.0,
                producer_op=None,
                consumer_ops=[o_name],
            ).to_dict()
        )
        descriptors.append(
            TensorDescriptor(
                name=f"{o_name}_weight_in",
                role="weight_input",
                operator_name=o_name,
                dtype=spec.numeric_precision,
                shape=[op_k, op_n],
                size_bytes=op_wgt_size,
                base_addr=weight_base_addr,
                stride_bytes=op_n * bytes_per_element,
                layout="row_major",
                requires_transpose=False,
                quantization_scale=1.0,
                producer_op=None,
                consumer_ops=[o_name],
            ).to_dict()
        )
        descriptors.append(
            TensorDescriptor(
                name=f"{o_name}_result_out",
                role="output",
                operator_name=o_name,
                dtype="INT32",
                shape=[op_m, op_n],
                size_bytes=op_res_size,
                base_addr=result_base_addr,
                stride_bytes=op_n * result_bytes_per_element,
                layout="row_major",
                requires_transpose=False,
                quantization_scale=1.0,
                producer_op=o_name,
                consumer_ops=[],
            ).to_dict()
        )

    return descriptors


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
