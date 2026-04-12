import json
from pathlib import Path
from typing import Dict, List, Tuple

SCHEDULER_IDLE = 0
SCHEDULER_DMA_ACT = 1
SCHEDULER_DMA_WGT = 2
SCHEDULER_LOAD = 3
SCHEDULER_COMPUTE = 4
SCHEDULER_STORE = 5
SCHEDULER_FLUSH = 6
SCHEDULER_CLEAR = 7
SCHEDULER_DONE = 8

SCHEDULER_STATE_NAMES = {
    SCHEDULER_IDLE: "IDLE",
    SCHEDULER_DMA_ACT: "DMA_ACT",
    SCHEDULER_DMA_WGT: "DMA_WGT",
    SCHEDULER_LOAD: "LOAD",
    SCHEDULER_COMPUTE: "COMPUTE",
    SCHEDULER_STORE: "STORE",
    SCHEDULER_FLUSH: "FLUSH",
    SCHEDULER_CLEAR: "CLEAR",
    SCHEDULER_DONE: "DONE",
}


def mac_reference(a: int, b: int, acc_in: int) -> int:
    return acc_in + (a * b)


def processing_element_reference(
    activation: int,
    weight: int,
    psum_in: int,
    compute_en: bool,
    clear_acc: bool,
) -> Tuple[int, bool]:
    if clear_acc:
        return 0, False
    if compute_en:
        return mac_reference(activation, weight, psum_in), True
    return psum_in, False


def _load_tile_operands(
    prev_activation_regs: List[List[int]],
    prev_weight_regs: List[List[int]],
    activations_west: List[int],
    weights_north: List[int],
    rows: int,
    cols: int,
    preload_en: bool,
    transpose_inputs: bool,
) -> Tuple[List[List[int]], List[List[int]]]:
    next_activations = [[0 for _ in range(cols)] for _ in range(rows)]
    next_weights = [[0 for _ in range(cols)] for _ in range(rows)]

    if preload_en:
        for row in range(rows):
            for col in range(cols):
                if transpose_inputs:
                    next_activations[row][col] = (
                        int(weights_north[row]) if row < len(weights_north) else 0
                    )
                    next_weights[row][col] = (
                        int(activations_west[col]) if col < len(activations_west) else 0
                    )
                else:
                    next_activations[row][col] = (
                        int(activations_west[row]) if row < len(activations_west) else 0
                    )
                    next_weights[row][col] = (
                        int(weights_north[col]) if col < len(weights_north) else 0
                    )
        return next_activations, next_weights

    for row in range(rows):
        for col in range(cols):
            if col == 0:
                next_activations[row][col] = (
                    int(weights_north[row])
                    if transpose_inputs and row < len(weights_north)
                    else int(activations_west[row]) if row < len(activations_west) else 0
                )
            else:
                next_activations[row][col] = prev_activation_regs[row][col - 1]

    for row in range(rows):
        for col in range(cols):
            if row == 0:
                next_weights[row][col] = (
                    int(activations_west[col])
                    if transpose_inputs and col < len(activations_west)
                    else int(weights_north[col]) if col < len(weights_north) else 0
                )
            else:
                next_weights[row][col] = prev_weight_regs[row - 1][col]

    return next_activations, next_weights


def _shift_output_stationary_operands(
    prev_activation_regs: List[List[int]],
    prev_weight_regs: List[List[int]],
    rows: int,
    cols: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    next_activations = [[0 for _ in range(cols)] for _ in range(rows)]
    next_weights = [[0 for _ in range(cols)] for _ in range(rows)]

    for row in range(rows):
        for col in range(cols - 1, -1, -1):
            if col == 0:
                next_activations[row][col] = 0
            else:
                next_activations[row][col] = prev_activation_regs[row][col - 1]
    for row in range(rows - 1, -1, -1):
        for col in range(cols):
            if row == 0:
                next_weights[row][col] = 0
            else:
                next_weights[row][col] = prev_weight_regs[row - 1][col]

    return next_activations, next_weights


def systolic_tile_reference(
    steps: List[Dict[str, object]],
    rows: int = 2,
    cols: int = 2,
) -> List[Dict[str, List[int]]]:
    activation_regs = [[0 for _ in range(cols)] for _ in range(rows)]
    weight_regs = [[0 for _ in range(cols)] for _ in range(rows)]
    psum_regs = [[0 for _ in range(cols)] for _ in range(rows)]
    valid_regs = [[False for _ in range(cols)] for _ in range(rows)]
    snapshots: List[Dict[str, List[int]]] = []

    for step in steps:
        load_inputs_en = bool(step["load_inputs_en"])
        compute_en = bool(step["compute_en"])
        clear_acc = bool(step["clear_acc"])
        flush_pipeline = bool(step.get("flush_pipeline", 0))
        output_stationary = bool(step.get("output_stationary_i", 0))
        preload_en = bool(step.get("preload_en_i", 0))
        transpose_inputs = bool(step.get("transpose_inputs_i", 0))
        activations_west = [int(value) for value in step["activations_west_i"]]
        weights_north = [int(value) for value in step["weights_north_i"]]

        if flush_pipeline:
            activation_regs = [[0 for _ in range(cols)] for _ in range(rows)]
            weight_regs = [[0 for _ in range(cols)] for _ in range(rows)]
        elif load_inputs_en:
            activation_regs, weight_regs = _load_tile_operands(
                prev_activation_regs=activation_regs,
                prev_weight_regs=weight_regs,
                activations_west=activations_west,
                weights_north=weights_north,
                rows=rows,
                cols=cols,
                preload_en=preload_en,
                transpose_inputs=transpose_inputs,
            )

        if clear_acc:
            psum_regs = [[0 for _ in range(cols)] for _ in range(rows)]
            valid_regs = [[False for _ in range(cols)] for _ in range(rows)]
        elif compute_en:
            next_psums = [[0 for _ in range(cols)] for _ in range(rows)]
            next_valids = [[True for _ in range(cols)] for _ in range(rows)]
            for row in range(rows):
                for col in range(cols):
                    next_psums[row][col] = psum_regs[row][col] + (
                        activation_regs[row][col] * weight_regs[row][col]
                    )
            psum_regs = next_psums
            valid_regs = next_valids
            if output_stationary and not load_inputs_en:
                activation_regs, weight_regs = _shift_output_stationary_operands(
                    prev_activation_regs=activation_regs,
                    prev_weight_regs=weight_regs,
                    rows=rows,
                    cols=cols,
                )
        else:
            valid_regs = [[False for _ in range(cols)] for _ in range(rows)]

        snapshots.append(
            {
                "psums_o": _flatten_matrix(psum_regs),
                "valids_o": [int(value) for value in _flatten_matrix(valid_regs)],
                "activations_state": _flatten_matrix(activation_regs),
                "weights_state": _flatten_matrix(weight_regs),
            }
        )

    return snapshots


def scratchpad_controller_reference(
    steps: List[Dict[str, object]],
    rows: int = 2,
    cols: int = 2,
    depth: int = 4,
    bank_count: int = 2,
) -> List[Dict[str, List[int]]]:
    activation_bank = [
        [[0 for _ in range(rows)] for _ in range(depth)] for _ in range(bank_count)
    ]
    weight_bank = [
        [[0 for _ in range(cols)] for _ in range(depth)] for _ in range(bank_count)
    ]
    activation_valid = [[False for _ in range(depth)] for _ in range(bank_count)]
    weight_valid = [[False for _ in range(depth)] for _ in range(bank_count)]
    snapshots: List[Dict[str, List[int]]] = []

    for step in steps:
        activation_read_bank = _sanitize_bank_index(step.get("activation_read_bank_i", 0), bank_count)
        activation_read_addr = int(step["activation_read_addr_i"])
        weight_read_bank = _sanitize_bank_index(step.get("weight_read_bank_i", 0), bank_count)
        weight_read_addr = int(step["weight_read_addr_i"])
        snapshots.append(
            {
                "activations_west_o": list(
                    activation_bank[activation_read_bank][activation_read_addr]
                ),
                "weights_north_o": list(weight_bank[weight_read_bank][weight_read_addr]),
                "vector_valid_o": int(
                    bool(step["load_vector_en_i"])
                    and activation_valid[activation_read_bank][activation_read_addr]
                    and weight_valid[weight_read_bank][weight_read_addr]
                ),
            }
        )

        if bool(step["write_activations_en_i"]):
            activation_write_bank = _sanitize_bank_index(
                step.get("activation_write_bank_i", 0), bank_count
            )
            activation_write_addr = int(step["activation_write_addr_i"])
            activation_bank[activation_write_bank][activation_write_addr] = [
                int(value) for value in step["activations_write_data_i"]
            ]
            activation_valid[activation_write_bank][activation_write_addr] = True
        if bool(step["write_weights_en_i"]):
            weight_write_bank = _sanitize_bank_index(
                step.get("weight_write_bank_i", 0), bank_count
            )
            weight_write_addr = int(step["weight_write_addr_i"])
            weight_bank[weight_write_bank][weight_write_addr] = [
                int(value) for value in step["weights_write_data_i"]
            ]
            weight_valid[weight_write_bank][weight_write_addr] = True

    return snapshots


def dma_engine_reference(
    steps: List[Dict[str, object]],
    rows: int = 2,
    cols: int = 2,
) -> List[Dict[str, object]]:
    pending_valid = False
    pending_write_weights = False
    pending_addr = 0
    pending_payload = [0 for _ in range(max(rows, cols))]
    snapshots: List[Dict[str, object]] = []

    for step in steps:
        if bool(step["dma_valid_i"]):
            pending_valid = True
            pending_write_weights = bool(step["dma_write_weights_i"])
            pending_addr = int(step["dma_addr_i"])
            pending_payload = [int(value) for value in step["dma_payload_i"]]
        else:
            pending_valid = False

        snapshot = {
            "write_activations_en_o": int(pending_valid and not pending_write_weights),
            "activation_write_addr_o": pending_addr,
            "activations_write_data_o": list(pending_payload[:rows]),
            "write_weights_en_o": int(pending_valid and pending_write_weights),
            "weight_write_addr_o": pending_addr,
            "weights_write_data_o": list(pending_payload[:cols]),
            "dma_done_o": int(pending_valid),
        }
        snapshots.append(snapshot)

    return snapshots


def tile_compute_unit_reference(
    steps: List[Dict[str, object]],
    rows: int = 2,
    cols: int = 2,
    depth: int = 4,
    bank_count: int = 2,
) -> List[Dict[str, List[int]]]:
    activation_bank = [
        [[0 for _ in range(rows)] for _ in range(depth)] for _ in range(bank_count)
    ]
    weight_bank = [
        [[0 for _ in range(cols)] for _ in range(depth)] for _ in range(bank_count)
    ]
    activation_valid = [[False for _ in range(depth)] for _ in range(bank_count)]
    weight_valid = [[False for _ in range(depth)] for _ in range(bank_count)]
    activation_regs = [[0 for _ in range(cols)] for _ in range(rows)]
    weight_regs = [[0 for _ in range(cols)] for _ in range(rows)]
    psum_regs = [[0 for _ in range(cols)] for _ in range(rows)]
    valid_regs = [[False for _ in range(cols)] for _ in range(rows)]
    snapshots: List[Dict[str, List[int]]] = []
    pending_dma_valid = False
    pending_dma_write_weights = False
    pending_dma_addr = 0
    pending_dma_bank = 0
    pending_dma_payload = [0 for _ in range(max(rows, cols))]

    for step in steps:
        activation_read_bank = _sanitize_bank_index(step.get("activation_read_bank_i", 0), bank_count)
        activation_read_addr = int(step["activation_read_addr_i"])
        weight_read_bank = _sanitize_bank_index(step.get("weight_read_bank_i", 0), bank_count)
        weight_read_addr = int(step["weight_read_addr_i"])
        activations_west = list(activation_bank[activation_read_bank][activation_read_addr])
        weights_north = list(weight_bank[weight_read_bank][weight_read_addr])
        scratchpad_vector_valid = (
            bool(step["load_vector_en_i"])
            and activation_valid[activation_read_bank][activation_read_addr]
            and weight_valid[weight_read_bank][weight_read_addr]
        )
        compute_en = bool(step["compute_en_i"])
        clear_acc = bool(step["clear_acc_i"])
        flush_pipeline = bool(step.get("flush_pipeline_i", 0))
        output_stationary = bool(step.get("output_stationary_i", 0))
        preload_en = bool(step.get("preload_en_i", 0))
        transpose_inputs = bool(step.get("transpose_inputs_i", 0))

        prev_activation_regs = [row[:] for row in activation_regs]
        prev_weight_regs = [row[:] for row in weight_regs]
        next_activation_regs = [row[:] for row in prev_activation_regs]
        next_weight_regs = [row[:] for row in prev_weight_regs]
        next_psum_regs = [row[:] for row in psum_regs]
        next_valid_regs = [row[:] for row in valid_regs]

        if flush_pipeline:
            next_activation_regs = [[0 for _ in range(cols)] for _ in range(rows)]
            next_weight_regs = [[0 for _ in range(cols)] for _ in range(rows)]
            next_valid_regs = [[False for _ in range(cols)] for _ in range(rows)]
        elif scratchpad_vector_valid:
            next_activation_regs, next_weight_regs = _load_tile_operands(
                prev_activation_regs=prev_activation_regs,
                prev_weight_regs=prev_weight_regs,
                activations_west=activations_west,
                weights_north=weights_north,
                rows=rows,
                cols=cols,
                preload_en=preload_en,
                transpose_inputs=transpose_inputs,
            )

        if clear_acc:
            next_psum_regs = [[0 for _ in range(cols)] for _ in range(rows)]
            next_valid_regs = [[False for _ in range(cols)] for _ in range(rows)]
        elif compute_en:
            for row in range(rows):
                for col in range(cols):
                    next_psum_regs[row][col] = psum_regs[row][col] + (
                        prev_activation_regs[row][col] * prev_weight_regs[row][col]
                    )
                    next_valid_regs[row][col] = True
            if output_stationary and not scratchpad_vector_valid:
                next_activation_regs, next_weight_regs = _shift_output_stationary_operands(
                    prev_activation_regs=prev_activation_regs,
                    prev_weight_regs=prev_weight_regs,
                    rows=rows,
                    cols=cols,
                )
        else:
            next_valid_regs = [[False for _ in range(cols)] for _ in range(rows)]

        activation_regs = next_activation_regs
        weight_regs = next_weight_regs
        psum_regs = next_psum_regs
        valid_regs = next_valid_regs

        snapshots.append(
            {
                "psums_o": _flatten_matrix(psum_regs),
                "valids_o": [int(value) for value in _flatten_matrix(valid_regs)],
                "scratchpad_vector_valid_o": int(scratchpad_vector_valid),
            }
        )

        if pending_dma_valid:
            if pending_dma_write_weights:
                weight_bank[pending_dma_bank][pending_dma_addr] = [
                    int(value) for value in pending_dma_payload[:cols]
                ]
                weight_valid[pending_dma_bank][pending_dma_addr] = True
            else:
                activation_bank[pending_dma_bank][pending_dma_addr] = [
                    int(value) for value in pending_dma_payload[:rows]
                ]
                activation_valid[pending_dma_bank][pending_dma_addr] = True

        if bool(step["dma_valid_i"]):
            pending_dma_valid = True
            pending_dma_write_weights = bool(step["dma_write_weights_i"])
            pending_dma_addr = int(step["dma_addr_i"])
            if pending_dma_write_weights:
                pending_dma_bank = _sanitize_bank_index(
                    step.get("weight_write_bank_i", 0), bank_count
                )
            else:
                pending_dma_bank = _sanitize_bank_index(
                    step.get("activation_write_bank_i", 0), bank_count
                )
            pending_dma_payload = [int(value) for value in step["dma_payload_i"]]
        else:
            pending_dma_valid = False
            pending_dma_write_weights = False
            pending_dma_addr = 0
            pending_dma_bank = 0
            pending_dma_payload = [0 for _ in range(max(rows, cols))]

    return snapshots


def accumulator_buffer_reference(
    steps: List[Dict[str, object]],
    rows: int = 2,
    cols: int = 2,
) -> List[Dict[str, object]]:
    pe_count = rows * cols
    stored_psums = [0 for _ in range(pe_count)]
    buffer_valid = False
    snapshots: List[Dict[str, object]] = []

    for step in steps:
        if bool(step.get("clear_i", 0)):
            stored_psums = [0 for _ in range(pe_count)]
            buffer_valid = False
        elif bool(step.get("capture_en_i", 0)):
            incoming_psums = [int(value) for value in step.get("psums_i", [])]
            stored_psums = list(incoming_psums[:pe_count]) + [0 for _ in range(max(0, pe_count - len(incoming_psums)))]
            buffer_valid = True

        scale_shift = int(step.get("read_scale_shift_i", 0))
        apply_bias = bool(step.get("apply_bias_i", 0))
        bias_values = [int(value) for value in step.get("bias_i", [])]
        bias_values = bias_values[:pe_count] + [0 for _ in range(max(0, pe_count - len(bias_values)))]
        readback = []
        for lane, stored_value in enumerate(stored_psums):
            scaled_value = int(stored_value) >> scale_shift
            if apply_bias:
                scaled_value += int(bias_values[lane])
            readback.append(scaled_value)

        snapshots.append(
            {
                "psums_o": list(stored_psums),
                "readback_o": readback,
                "readback_valid_mask_o": [
                    int(bool(step.get("store_en_i", 0)) and buffer_valid) for _ in range(pe_count)
                ],
                "buffer_valid_o": int(buffer_valid),
            }
        )

    return snapshots


def gemm_ctrl_reference(
    steps: List[Dict[str, object]],
    rows: int = 2,
    cols: int = 2,
) -> List[Dict[str, object]]:
    pe_count = rows * cols
    stored_psums = [0 for _ in range(pe_count)]
    accumulator_valid = False
    writeback_pending = False
    k_tiles_completed = 0
    snapshots: List[Dict[str, object]] = []

    for step in steps:
        clear = bool(step.get("clear_i", 0))
        issue = bool(step.get("issue_i", 0))
        accumulate = bool(step.get("accumulate_i", 0))
        final_k_tile = bool(step.get("final_k_tile_i", 0))

        if clear:
            stored_psums = [0 for _ in range(pe_count)]
            accumulator_valid = False
            writeback_pending = False
            k_tiles_completed = 0
        elif issue:
            incoming_psums = [int(value) for value in step.get("partial_psums_i", [])]
            incoming_psums = incoming_psums[:pe_count] + [0 for _ in range(max(0, pe_count - len(incoming_psums)))]
            if accumulate and accumulator_valid:
                stored_psums = [
                    int(stored_psums[lane]) + int(incoming_psums[lane])
                    for lane in range(pe_count)
                ]
                k_tiles_completed += 1
            else:
                stored_psums = list(incoming_psums)
                accumulator_valid = True
                k_tiles_completed = 1
                writeback_pending = False
            accumulator_valid = True
            if final_k_tile:
                writeback_pending = True

        scale_shift = int(step.get("read_scale_shift_i", 0))
        apply_bias = bool(step.get("apply_bias_i", 0))
        bias_values = [int(value) for value in step.get("bias_i", [])]
        bias_values = bias_values[:pe_count] + [0 for _ in range(max(0, pe_count - len(bias_values)))]
        writeback = []
        for lane, stored_value in enumerate(stored_psums):
            scaled_value = int(stored_value) >> scale_shift
            if apply_bias:
                scaled_value += int(bias_values[lane])
            writeback.append(scaled_value)

        writeback_fire = bool(step.get("writeback_en_i", 0)) and writeback_pending and accumulator_valid
        snapshots.append(
            {
                "accumulated_psums_o": list(stored_psums),
                "writeback_o": writeback,
                "writeback_valid_mask_o": [int(writeback_fire) for _ in range(pe_count)],
                "accumulator_valid_o": int(accumulator_valid),
                "writeback_pending_o": int(writeback_pending),
                "k_tiles_completed_o": int(k_tiles_completed),
            }
        )

    return snapshots


def scheduler_reference(
    steps: List[Dict[str, object]],
    rows: int = 2,
    cols: int = 2,
) -> List[Dict[str, object]]:
    state = SCHEDULER_IDLE
    slot_index = 0
    load_index = 0
    compute_index = 0
    store_index = 0
    program_slot_count = 2
    program_load_iterations = 2
    program_compute_iterations = 2
    program_clear_on_done = True
    program_store_burst_count = 2
    decoupled_mode = False
    load_issued = 0
    compute_issued = 0
    store_issued = 0
    snapshots: List[Dict[str, object]] = []

    for step in steps:
        (
            state,
            slot_index,
            load_index,
            compute_index,
            store_index,
            program_slot_count,
            program_load_iterations,
            program_compute_iterations,
            program_clear_on_done,
            program_store_burst_count,
            decoupled_mode,
            load_issued,
            compute_issued,
            store_issued,
        ) = _scheduler_next_context(
            state=state,
            slot_index=slot_index,
            load_index=load_index,
            compute_index=compute_index,
            store_index=store_index,
            step=step,
            rows=rows,
            program_slot_count=program_slot_count,
            program_load_iterations=program_load_iterations,
            program_compute_iterations=program_compute_iterations,
            program_clear_on_done=program_clear_on_done,
            program_store_burst_count=program_store_burst_count,
            decoupled_mode=decoupled_mode,
            load_issued=load_issued,
            compute_issued=compute_issued,
            store_issued=store_issued,
        )
        snapshots.append(
            _scheduler_outputs(
                state=state,
                step=step,
                rows=rows,
                cols=cols,
                slot_index=slot_index,
                load_index=load_index,
                store_index=store_index,
                program_slot_count=program_slot_count,
                program_load_iterations=program_load_iterations,
                program_compute_iterations=program_compute_iterations,
                program_store_burst_count=program_store_burst_count,
                decoupled_mode=decoupled_mode,
                load_issued=load_issued,
                compute_issued=compute_issued,
                store_issued=store_issued,
            )
        )

    return snapshots


def cluster_control_reference(
    steps: List[Dict[str, object]],
    rows: int = 2,
    cols: int = 2,
    tile_count: int = 1,
) -> List[Dict[str, object]]:
    scheduler_snapshots = scheduler_reference(steps=steps, rows=rows, cols=cols)
    sanitized_tile_count = _sanitize_tile_count(tile_count)
    tile_enable_masks = [
        _sanitize_tile_enable_mask(step.get("tile_enable_i"), sanitized_tile_count)
        for step in steps
    ]
    return [
        _cluster_control_outputs(
            scheduler_snapshot=scheduler_snapshot,
            tile_enable_mask=tile_enable_masks[step_index],
        )
        for step_index, scheduler_snapshot in enumerate(scheduler_snapshots)
    ]


def cluster_interconnect_reference(
    steps: List[Dict[str, object]],
    rows: int = 2,
    cols: int = 2,
    tile_count: int = 1,
) -> List[Dict[str, object]]:
    sanitized_tile_count = _sanitize_tile_count(tile_count)
    snapshots: List[Dict[str, object]] = []
    for step in steps:
        control_snapshot = {
            "tile_dma_valid_o": _sanitize_tile_enable_mask(
                step.get("tile_dma_valid_i"), sanitized_tile_count
            ),
            "dma_write_weights_o": int(step.get("dma_write_weights_i", 0)),
            "dma_bank_select_o": int(bool(step.get("dma_bank_select_i", 0))),
            "dma_local_addr_o": int(step.get("dma_local_addr_i", 0)),
            "tile_load_vector_en_o": _sanitize_tile_enable_mask(
                step.get("tile_load_vector_en_i"), sanitized_tile_count
            ),
            "activation_read_bank_select_o": int(
                bool(step.get("activation_read_bank_select_i", 0))
            ),
            "activation_local_read_addr_o": int(step.get("activation_local_read_addr_i", 0)),
            "weight_read_bank_select_o": int(bool(step.get("weight_read_bank_select_i", 0))),
            "weight_local_read_addr_o": int(step.get("weight_local_read_addr_i", 0)),
            "tile_compute_en_o": _sanitize_tile_enable_mask(
                step.get("tile_compute_en_i"), sanitized_tile_count
            ),
            "tile_flush_pipeline_o": _sanitize_tile_enable_mask(
                step.get("tile_flush_pipeline_i"), sanitized_tile_count
            ),
            "tile_clear_acc_o": _sanitize_tile_enable_mask(
                step.get("tile_clear_acc_i"), sanitized_tile_count
            ),
            "tile_store_results_en_o": _sanitize_tile_enable_mask(
                step.get("tile_store_results_en_i"), sanitized_tile_count
            ),
            "store_results_en_o": int(step.get("store_results_en_i", 0)),
            "result_write_addr_o": int(step.get("result_write_addr_i", 0)),
            "store_burst_index_o": int(step.get("store_burst_index_i", 0)),
        }
        snapshots.append(
            _cluster_interconnect_outputs(
                control_snapshot=control_snapshot,
                dma_payload=[int(value) for value in step.get("dma_payload_i", [])],
                psums=[int(value) for value in step.get("tile_psums_i", [])],
                rows=rows,
                cols=cols,
                tile_count=sanitized_tile_count,
                step=step,
            )
        )
    return snapshots


def top_npu_context_reference(
    steps: List[Dict[str, object]],
    rows: int = 2,
    cols: int = 2,
    depth: int = 4,
    tile_count: int = 1,
) -> Dict[str, List[Dict[str, object]]]:
    sanitized_tile_count = _sanitize_tile_count(tile_count)
    max_dim = max(rows, cols)
    state = SCHEDULER_IDLE
    slot_index = 0
    load_index = 0
    compute_index = 0
    store_index = 0
    program_slot_count = 2
    program_load_iterations = 2
    program_compute_iterations = 2
    program_clear_on_done = True
    program_store_burst_count = 2
    decoupled_mode = False
    load_issued = 0
    compute_issued = 0
    store_issued = 0
    scheduler_snapshots: List[Dict[str, object]] = []
    control_snapshots: List[Dict[str, object]] = []
    idle_scheduler_snapshot = {
        "state_o": SCHEDULER_IDLE,
        "busy_o": 0,
        "done_o": 0,
        "dma_valid_o": 0,
        "dma_write_weights_o": 0,
        "dma_addr_o": 0,
        "dma_payload_o": [0 for _ in range(max_dim)],
        "load_vector_en_o": 0,
        "activation_read_addr_o": 0,
        "weight_read_addr_o": 0,
        "compute_en_o": 0,
        "flush_pipeline_o": 0,
        "clear_acc_o": 0,
        "store_results_en_o": 0,
        "result_write_addr_o": 0,
        "store_burst_index_o": 0,
        "decoupled_mode_o": 0,
        "load_queue_depth_o": 0,
        "execute_queue_depth_o": 0,
        "store_queue_depth_o": 0,
        "hazard_wait_o": 0,
    }
    idle_control_snapshot = _cluster_control_outputs(
        scheduler_snapshot=idle_scheduler_snapshot,
        tile_enable_mask=[0 for _ in range(sanitized_tile_count)],
    )
    idle_step = {
        "tile_dma_ready_i": [1 for _ in range(sanitized_tile_count)],
        "tile_load_ready_i": [1 for _ in range(sanitized_tile_count)],
        "store_ready_i": 1,
    }
    prev_control_snapshot = idle_control_snapshot

    for step in steps:
        scheduler_accepts = _cluster_interconnect_accepts(
            control_snapshot=prev_control_snapshot,
            step=step,
            tile_count=sanitized_tile_count,
        )
        scheduler_step = dict(step)
        scheduler_step.update(scheduler_accepts)
        (
            state,
            slot_index,
            load_index,
            compute_index,
            store_index,
            program_slot_count,
            program_load_iterations,
            program_compute_iterations,
            program_clear_on_done,
            program_store_burst_count,
            decoupled_mode,
            load_issued,
            compute_issued,
            store_issued,
        ) = _scheduler_next_context(
            state=state,
            slot_index=slot_index,
            load_index=load_index,
            compute_index=compute_index,
            store_index=store_index,
            step=scheduler_step,
            rows=rows,
            program_slot_count=program_slot_count,
            program_load_iterations=program_load_iterations,
            program_compute_iterations=program_compute_iterations,
            program_clear_on_done=program_clear_on_done,
            program_store_burst_count=program_store_burst_count,
            decoupled_mode=decoupled_mode,
            load_issued=load_issued,
            compute_issued=compute_issued,
            store_issued=store_issued,
        )
        scheduler_snapshot = _scheduler_outputs(
            state=state,
            step=scheduler_step,
            rows=rows,
            cols=cols,
            slot_index=slot_index,
            load_index=load_index,
            store_index=store_index,
            program_slot_count=program_slot_count,
            program_load_iterations=program_load_iterations,
            program_compute_iterations=program_compute_iterations,
            program_store_burst_count=program_store_burst_count,
            decoupled_mode=decoupled_mode,
            load_issued=load_issued,
            compute_issued=compute_issued,
            store_issued=store_issued,
        )
        control_snapshot = _cluster_control_outputs(
            scheduler_snapshot=scheduler_snapshot,
            tile_enable_mask=_sanitize_tile_enable_mask(
                step.get("tile_enable_i"), sanitized_tile_count
            ),
        )
        scheduler_snapshots.append(scheduler_snapshot)
        control_snapshots.append(control_snapshot)
        prev_control_snapshot = control_snapshot

    idle_interconnect_snapshot = _cluster_interconnect_outputs(
        control_snapshot=idle_control_snapshot,
        dma_payload=[0 for _ in range(max_dim)],
        psums=[0 for _ in range(sanitized_tile_count * rows * cols)],
        rows=rows,
        cols=cols,
        tile_count=sanitized_tile_count,
        step=idle_step,
    )

    tile_snapshots_by_index: List[List[Dict[str, object]]] = []
    for tile_index in range(sanitized_tile_count):
        tile_steps = []
        for step_index in range(len(steps)):
            cycle_interconnect_snapshot = (
                _cluster_interconnect_outputs(
                    control_snapshot=control_snapshots[step_index - 1],
                    dma_payload=[
                        int(value)
                        for value in scheduler_snapshots[step_index - 1]["dma_payload_o"]
                    ],
                    psums=[0 for _ in range(sanitized_tile_count * rows * cols)],
                    rows=rows,
                    cols=cols,
                    tile_count=sanitized_tile_count,
                    step=steps[step_index],
                )
                if step_index > 0
                else idle_interconnect_snapshot
            )
            tile_steps.append(
                {
                    "dma_valid_i": cycle_interconnect_snapshot["tile_dma_valid_o"][tile_index],
                    "dma_write_weights_i": cycle_interconnect_snapshot["dma_write_weights_o"],
                    "dma_addr_i": cycle_interconnect_snapshot["dma_local_addr_o"],
                    "dma_payload_i": cycle_interconnect_snapshot["dma_payload_o"],
                    "activation_write_bank_i": cycle_interconnect_snapshot["dma_bank_select_o"],
                    "weight_write_bank_i": cycle_interconnect_snapshot["dma_bank_select_o"],
                    "load_vector_en_i": cycle_interconnect_snapshot["tile_load_vector_en_o"][tile_index],
                    "activation_read_bank_i": cycle_interconnect_snapshot["activation_read_bank_select_o"],
                    "activation_read_addr_i": cycle_interconnect_snapshot["activation_local_read_addr_o"],
                    "weight_read_bank_i": cycle_interconnect_snapshot["weight_read_bank_select_o"],
                    "weight_read_addr_i": cycle_interconnect_snapshot["weight_local_read_addr_o"],
                    "compute_en_i": cycle_interconnect_snapshot["tile_compute_en_o"][tile_index],
                    "flush_pipeline_i": cycle_interconnect_snapshot["tile_flush_pipeline_o"][tile_index],
                    "clear_acc_i": cycle_interconnect_snapshot["tile_clear_acc_o"][tile_index],
                    "output_stationary_i": int(step.get("output_stationary_i", 0)),
                    "preload_en_i": int(step.get("preload_en_i", 0)),
                    "transpose_inputs_i": int(step.get("transpose_inputs_i", 0)),
                }
            )
        tile_snapshots_by_index.append(
            tile_compute_unit_reference(
                steps=tile_steps,
                rows=rows,
                cols=cols,
                depth=depth,
            )
        )

    interconnect_snapshots: List[Dict[str, object]] = []
    top_snapshots: List[Dict[str, object]] = []
    for step_index, step in enumerate(steps):
        psums: List[int] = []
        valids: List[int] = []
        for tile_index in range(sanitized_tile_count):
            tile_snapshot = tile_snapshots_by_index[tile_index][step_index]
            psums.extend([int(value) for value in tile_snapshot["psums_o"]])
            valids.extend([int(value) for value in tile_snapshot["valids_o"]])
        interconnect_snapshot = _cluster_interconnect_outputs(
            control_snapshot=control_snapshots[step_index],
            dma_payload=[int(value) for value in scheduler_snapshots[step_index]["dma_payload_o"]],
            psums=psums,
            rows=rows,
            cols=cols,
            tile_count=sanitized_tile_count,
            step=step,
        )
        interconnect_snapshots.append(interconnect_snapshot)
        top_snapshots.append(
            {
                "scheduler_state_o": scheduler_snapshots[step_index]["state_o"],
                "busy_o": scheduler_snapshots[step_index]["busy_o"],
                "done_o": scheduler_snapshots[step_index]["done_o"],
                "result_write_valid_o": interconnect_snapshot["result_write_valid_o"],
                "result_write_addr_o": interconnect_snapshot["result_write_addr_o"],
                "result_write_payload_o": interconnect_snapshot["result_write_payload_o"],
                "result_write_valid_mask_o": interconnect_snapshot["result_write_valid_mask_o"],
                "psums_o": psums,
                "valids_o": valids,
            }
        )

    return {
        "scheduler": scheduler_snapshots,
        "control": control_snapshots,
        "interconnect": interconnect_snapshots,
        "top": top_snapshots,
    }


def top_npu_reference(
    steps: List[Dict[str, object]],
    rows: int = 2,
    cols: int = 2,
    depth: int = 4,
    tile_count: int = 1,
) -> List[Dict[str, object]]:
    return top_npu_context_reference(
        steps=steps,
        rows=rows,
        cols=cols,
        depth=depth,
        tile_count=tile_count,
    )["top"]


def scheduler_state_name(state: int) -> str:
    return SCHEDULER_STATE_NAMES.get(state, f"UNKNOWN_{state}")


def evaluate_reference_cases(reference_cases_path: str) -> Tuple[bool, str]:
    payload = json.loads(Path(reference_cases_path).read_text(encoding="utf-8"))
    failures: List[str] = []

    for case in payload.get("mac_unit", []):
        observed = mac_reference(
            a=case["inputs"]["a"],
            b=case["inputs"]["b"],
            acc_in=case["inputs"]["acc_in"],
        )
        if observed != case["expected"]["acc_out"]:
            failures.append(
                f"mac_unit/{case['name']}: expected {case['expected']['acc_out']}, got {observed}"
            )

    for case in payload.get("processing_element", []):
        observed_psum, observed_valid = processing_element_reference(
            activation=case["inputs"]["activation_i"],
            weight=case["inputs"]["weight_i"],
            psum_in=case["inputs"]["psum_i"],
            compute_en=bool(case["inputs"]["compute_en"]),
            clear_acc=bool(case["inputs"]["clear_acc"]),
        )
        if observed_psum != case["expected"]["psum_o"] or observed_valid != bool(
            case["expected"]["valid_o"]
        ):
            failures.append(
                "processing_element/"
                f"{case['name']}: expected ({case['expected']['psum_o']}, "
                f"{bool(case['expected']['valid_o'])}), got ({observed_psum}, {observed_valid})"
            )

    for case in payload.get("systolic_tile", []):
        rows = int(case.get("rows", 2))
        cols = int(case.get("cols", 2))
        observed_snapshots = systolic_tile_reference(
            steps=case["steps"],
            rows=rows,
            cols=cols,
        )
        if len(observed_snapshots) != len(case["steps"]):
            failures.append(
                f"systolic_tile/{case['name']}: numero di snapshot inatteso {len(observed_snapshots)}"
            )
            continue

        for step_idx, step in enumerate(case["steps"]):
            expected = step["expected"]
            observed = observed_snapshots[step_idx]
            if (
                observed["psums_o"] != expected["psums_o"]
                or observed["valids_o"] != expected["valids_o"]
            ):
                failures.append(
                    "systolic_tile/"
                    f"{case['name']}/step_{step_idx}: expected "
                    f"({expected['psums_o']}, {expected['valids_o']}), got "
                    f"({observed['psums_o']}, {observed['valids_o']})"
                )

    for case in payload.get("scratchpad_controller", []):
        rows = int(case.get("rows", 2))
        cols = int(case.get("cols", 2))
        depth = int(case.get("depth", 4))
        observed_snapshots = scratchpad_controller_reference(
            steps=case["steps"],
            rows=rows,
            cols=cols,
            depth=depth,
        )
        for step_idx, step in enumerate(case["steps"]):
            expected = step["expected"]
            observed = observed_snapshots[step_idx]
            if (
                observed["activations_west_o"] != expected["activations_west_o"]
                or observed["weights_north_o"] != expected["weights_north_o"]
                or observed["vector_valid_o"] != expected["vector_valid_o"]
            ):
                failures.append(
                    "scratchpad_controller/"
                    f"{case['name']}/step_{step_idx}: expected "
                    f"({expected['activations_west_o']}, {expected['weights_north_o']}, "
                    f"{expected['vector_valid_o']}), got "
                    f"({observed['activations_west_o']}, {observed['weights_north_o']}, "
                    f"{observed['vector_valid_o']})"
                )

    for case in payload.get("accumulator_buffer", []):
        rows = int(case.get("rows", 2))
        cols = int(case.get("cols", 2))
        observed_snapshots = accumulator_buffer_reference(
            steps=case["steps"],
            rows=rows,
            cols=cols,
        )
        for step_idx, step in enumerate(case["steps"]):
            expected = step["expected"]
            observed = observed_snapshots[step_idx]
            if (
                observed["psums_o"] != expected["psums_o"]
                or observed["readback_o"] != expected["readback_o"]
                or observed["readback_valid_mask_o"] != expected["readback_valid_mask_o"]
                or observed["buffer_valid_o"] != expected["buffer_valid_o"]
            ):
                failures.append(
                    "accumulator_buffer/"
                    f"{case['name']}/step_{step_idx}: expected "
                    f"({expected['psums_o']}, {expected['readback_o']}, "
                    f"{expected['readback_valid_mask_o']}, {expected['buffer_valid_o']}), got "
                    f"({observed['psums_o']}, {observed['readback_o']}, "
                    f"{observed['readback_valid_mask_o']}, {observed['buffer_valid_o']})"
                )

    for case in payload.get("gemm_ctrl", []):
        rows = int(case.get("rows", 2))
        cols = int(case.get("cols", 2))
        observed_snapshots = gemm_ctrl_reference(
            steps=case["steps"],
            rows=rows,
            cols=cols,
        )
        for step_idx, step in enumerate(case["steps"]):
            expected = step["expected"]
            observed = observed_snapshots[step_idx]
            if (
                observed["accumulated_psums_o"] != expected["accumulated_psums_o"]
                or observed["writeback_o"] != expected["writeback_o"]
                or observed["writeback_valid_mask_o"] != expected["writeback_valid_mask_o"]
                or observed["accumulator_valid_o"] != expected["accumulator_valid_o"]
                or observed["writeback_pending_o"] != expected["writeback_pending_o"]
                or observed["k_tiles_completed_o"] != expected["k_tiles_completed_o"]
            ):
                failures.append(
                    "gemm_ctrl/"
                    f"{case['name']}/step_{step_idx}: expected "
                    f"({expected['accumulated_psums_o']}, {expected['writeback_o']}, "
                    f"{expected['writeback_valid_mask_o']}, {expected['accumulator_valid_o']}, "
                    f"{expected['writeback_pending_o']}, {expected['k_tiles_completed_o']}), got "
                    f"({observed['accumulated_psums_o']}, {observed['writeback_o']}, "
                    f"{observed['writeback_valid_mask_o']}, {observed['accumulator_valid_o']}, "
                    f"{observed['writeback_pending_o']}, {observed['k_tiles_completed_o']})"
                )

    for case in payload.get("dma_engine", []):
        rows = int(case.get("rows", 2))
        cols = int(case.get("cols", 2))
        observed_snapshots = dma_engine_reference(
            steps=case["steps"],
            rows=rows,
            cols=cols,
        )
        for step_idx, step in enumerate(case["steps"]):
            expected = step["expected"]
            observed = observed_snapshots[step_idx]
            if (
                observed["write_activations_en_o"] != expected["write_activations_en_o"]
                or observed["activation_write_addr_o"] != expected["activation_write_addr_o"]
                or observed["activations_write_data_o"] != expected["activations_write_data_o"]
                or observed["write_weights_en_o"] != expected["write_weights_en_o"]
                or observed["weight_write_addr_o"] != expected["weight_write_addr_o"]
                or observed["weights_write_data_o"] != expected["weights_write_data_o"]
                or observed["dma_done_o"] != expected["dma_done_o"]
            ):
                failures.append(
                    "dma_engine/"
                    f"{case['name']}/step_{step_idx}: expected "
                    f"({expected['write_activations_en_o']}, {expected['activation_write_addr_o']}, "
                    f"{expected['activations_write_data_o']}, {expected['write_weights_en_o']}, "
                    f"{expected['weight_write_addr_o']}, {expected['weights_write_data_o']}, "
                    f"{expected['dma_done_o']}), got "
                    f"({observed['write_activations_en_o']}, {observed['activation_write_addr_o']}, "
                    f"{observed['activations_write_data_o']}, {observed['write_weights_en_o']}, "
                    f"{observed['weight_write_addr_o']}, {observed['weights_write_data_o']}, "
                    f"{observed['dma_done_o']})"
                )

    for case in list(payload.get("scheduler", [])) + list(payload.get("scheduler_stress", [])):
        rows = int(case.get("rows", 2))
        cols = int(case.get("cols", 2))
        observed_snapshots = scheduler_reference(
            steps=case["steps"],
            rows=rows,
            cols=cols,
        )
        for step_idx, step in enumerate(case["steps"]):
            expected = step["expected"]
            observed = observed_snapshots[step_idx]
            if (
                observed["state_o"] != expected["state_o"]
                or observed["busy_o"] != expected["busy_o"]
                or observed["done_o"] != expected["done_o"]
                or observed["dma_valid_o"] != expected["dma_valid_o"]
                or observed["dma_write_weights_o"] != expected["dma_write_weights_o"]
                or observed["dma_addr_o"] != expected["dma_addr_o"]
                or observed["dma_payload_o"] != expected["dma_payload_o"]
                or observed["load_vector_en_o"] != expected["load_vector_en_o"]
                or observed["activation_read_addr_o"] != expected["activation_read_addr_o"]
                or observed["weight_read_addr_o"] != expected["weight_read_addr_o"]
                or observed["store_results_en_o"] != expected["store_results_en_o"]
                or observed["result_write_addr_o"] != expected["result_write_addr_o"]
                or observed["store_burst_index_o"] != expected["store_burst_index_o"]
                or observed["compute_en_o"] != expected["compute_en_o"]
                or observed["flush_pipeline_o"] != expected["flush_pipeline_o"]
                or observed["clear_acc_o"] != expected["clear_acc_o"]
                or observed.get("decoupled_mode_o", 0) != expected.get("decoupled_mode_o", 0)
                or observed.get("load_queue_depth_o", 0) != expected.get("load_queue_depth_o", 0)
                or observed.get("execute_queue_depth_o", 0) != expected.get("execute_queue_depth_o", 0)
                or observed.get("store_queue_depth_o", 0) != expected.get("store_queue_depth_o", 0)
                or observed.get("hazard_wait_o", 0) != expected.get("hazard_wait_o", 0)
            ):
                failures.append(
                    "scheduler/"
                    f"{case['name']}/step_{step_idx}: expected "
                    f"({expected['state_o']}, {expected['busy_o']}, {expected['done_o']}, "
                    f"{expected['dma_valid_o']}, {expected['dma_write_weights_o']}, "
                    f"{expected['dma_addr_o']}, {expected['dma_payload_o']}, "
                    f"{expected['load_vector_en_o']}, {expected['activation_read_addr_o']}, "
                    f"{expected['weight_read_addr_o']}, {expected['store_results_en_o']}, "
                    f"{expected['result_write_addr_o']}, {expected['store_burst_index_o']}, "
                    f"{expected['compute_en_o']}, {expected['flush_pipeline_o']}, "
                    f"{expected['clear_acc_o']}, {expected.get('decoupled_mode_o', 0)}, "
                    f"{expected.get('load_queue_depth_o', 0)}, {expected.get('execute_queue_depth_o', 0)}, "
                    f"{expected.get('store_queue_depth_o', 0)}, {expected.get('hazard_wait_o', 0)}), got "
                    f"({observed['state_o']}, {observed['busy_o']}, {observed['done_o']}, "
                    f"{observed['dma_valid_o']}, {observed['dma_write_weights_o']}, "
                    f"{observed['dma_addr_o']}, {observed['dma_payload_o']}, "
                    f"{observed['load_vector_en_o']}, {observed['activation_read_addr_o']}, "
                    f"{observed['weight_read_addr_o']}, {observed['store_results_en_o']}, "
                    f"{observed['result_write_addr_o']}, {observed['store_burst_index_o']}, "
                    f"{observed['compute_en_o']}, {observed['flush_pipeline_o']}, "
                    f"{observed['clear_acc_o']}, {observed.get('decoupled_mode_o', 0)}, "
                    f"{observed.get('load_queue_depth_o', 0)}, {observed.get('execute_queue_depth_o', 0)}, "
                    f"{observed.get('store_queue_depth_o', 0)}, {observed.get('hazard_wait_o', 0)})"
                )

    for case in list(payload.get("cluster_control", [])) + list(payload.get("cluster_control_stress", [])):
        rows = int(case.get("rows", 2))
        cols = int(case.get("cols", 2))
        tile_count = int(case.get("tile_count", 1))
        observed_snapshots = cluster_control_reference(
            steps=case["steps"],
            rows=rows,
            cols=cols,
            tile_count=tile_count,
        )
        for step_idx, step in enumerate(case["steps"]):
            expected = step["expected"]
            observed = observed_snapshots[step_idx]
            if (
                observed["tile_dma_valid_o"] != expected["tile_dma_valid_o"]
                or observed["dma_write_weights_o"] != expected["dma_write_weights_o"]
                or observed["dma_bank_select_o"] != expected["dma_bank_select_o"]
                or observed["dma_local_addr_o"] != expected["dma_local_addr_o"]
                or observed["tile_load_vector_en_o"] != expected["tile_load_vector_en_o"]
                or observed["activation_read_bank_select_o"]
                != expected["activation_read_bank_select_o"]
                or observed["activation_local_read_addr_o"]
                != expected["activation_local_read_addr_o"]
                or observed["weight_read_bank_select_o"]
                != expected["weight_read_bank_select_o"]
                or observed["weight_local_read_addr_o"] != expected["weight_local_read_addr_o"]
                or observed["tile_compute_en_o"] != expected["tile_compute_en_o"]
                or observed["tile_flush_pipeline_o"] != expected["tile_flush_pipeline_o"]
                or observed["tile_clear_acc_o"] != expected["tile_clear_acc_o"]
                or observed["tile_store_results_en_o"] != expected["tile_store_results_en_o"]
                or observed["store_results_en_o"] != expected["store_results_en_o"]
                or observed["result_write_addr_o"] != expected["result_write_addr_o"]
                or observed["store_burst_index_o"] != expected["store_burst_index_o"]
            ):
                failures.append(
                    "cluster_control/"
                    f"{case['name']}/step_{step_idx}: expected "
                    f"({expected['tile_dma_valid_o']}, {expected['dma_write_weights_o']}, "
                    f"{expected['dma_bank_select_o']}, {expected['dma_local_addr_o']}, "
                    f"{expected['tile_load_vector_en_o']}, "
                    f"{expected['activation_read_bank_select_o']}, "
                    f"{expected['activation_local_read_addr_o']}, "
                    f"{expected['weight_read_bank_select_o']}, "
                    f"{expected['weight_local_read_addr_o']}, "
                    f"{expected['tile_compute_en_o']}, {expected['tile_flush_pipeline_o']}, "
                    f"{expected['tile_clear_acc_o']}, {expected['tile_store_results_en_o']}, "
                    f"{expected['store_results_en_o']}, {expected['result_write_addr_o']}, "
                    f"{expected['store_burst_index_o']}), got "
                    f"({observed['tile_dma_valid_o']}, {observed['dma_write_weights_o']}, "
                    f"{observed['dma_bank_select_o']}, {observed['dma_local_addr_o']}, "
                    f"{observed['tile_load_vector_en_o']}, "
                    f"{observed['activation_read_bank_select_o']}, "
                    f"{observed['activation_local_read_addr_o']}, "
                    f"{observed['weight_read_bank_select_o']}, "
                    f"{observed['weight_local_read_addr_o']}, "
                    f"{observed['tile_compute_en_o']}, {observed['tile_flush_pipeline_o']}, "
                    f"{observed['tile_clear_acc_o']}, {observed['tile_store_results_en_o']}, "
                    f"{observed['store_results_en_o']}, {observed['result_write_addr_o']}, "
                    f"{observed['store_burst_index_o']})"
                )

    for case in list(payload.get("cluster_interconnect", [])) + list(payload.get("cluster_interconnect_stress", [])):
        rows = int(case.get("rows", 2))
        cols = int(case.get("cols", 2))
        tile_count = int(case.get("tile_count", 1))
        observed_snapshots = cluster_interconnect_reference(
            steps=case["steps"],
            rows=rows,
            cols=cols,
            tile_count=tile_count,
        )
        for step_idx, step in enumerate(case["steps"]):
            expected = step["expected"]
            observed = observed_snapshots[step_idx]
            if (
                observed["dma_accept_o"] != expected["dma_accept_o"]
                or observed["load_accept_o"] != expected["load_accept_o"]
                or observed["store_accept_o"] != expected["store_accept_o"]
                or observed["dma_backpressure_o"] != expected["dma_backpressure_o"]
                or observed["load_backpressure_o"] != expected["load_backpressure_o"]
                or observed["store_backpressure_o"] != expected["store_backpressure_o"]
                or observed["tile_dma_valid_o"] != expected["tile_dma_valid_o"]
                or observed["dma_write_weights_o"] != expected["dma_write_weights_o"]
                or observed["dma_bank_select_o"] != expected["dma_bank_select_o"]
                or observed["dma_local_addr_o"] != expected["dma_local_addr_o"]
                or observed["dma_payload_o"] != expected["dma_payload_o"]
                or observed["tile_load_vector_en_o"] != expected["tile_load_vector_en_o"]
                or observed["activation_read_bank_select_o"]
                != expected["activation_read_bank_select_o"]
                or observed["activation_local_read_addr_o"]
                != expected["activation_local_read_addr_o"]
                or observed["weight_read_bank_select_o"]
                != expected["weight_read_bank_select_o"]
                or observed["weight_local_read_addr_o"] != expected["weight_local_read_addr_o"]
                or observed["tile_compute_en_o"] != expected["tile_compute_en_o"]
                or observed["tile_flush_pipeline_o"] != expected["tile_flush_pipeline_o"]
                or observed["tile_clear_acc_o"] != expected["tile_clear_acc_o"]
                or observed["tile_store_results_en_o"] != expected["tile_store_results_en_o"]
                or observed["result_write_valid_o"] != expected["result_write_valid_o"]
                or observed["result_write_addr_o"] != expected["result_write_addr_o"]
                or observed["result_write_payload_o"] != expected["result_write_payload_o"]
                or observed["result_write_valid_mask_o"] != expected["result_write_valid_mask_o"]
            ):
                failures.append(
                    "cluster_interconnect/"
                    f"{case['name']}/step_{step_idx}: expected "
                    f"({expected['dma_accept_o']}, {expected['load_accept_o']}, "
                    f"{expected['store_accept_o']}, {expected['dma_backpressure_o']}, "
                    f"{expected['load_backpressure_o']}, {expected['store_backpressure_o']}, "
                    f"{expected['tile_dma_valid_o']}, {expected['dma_write_weights_o']}, "
                    f"{expected['dma_bank_select_o']}, {expected['dma_local_addr_o']}, "
                    f"{expected['dma_payload_o']}, {expected['tile_load_vector_en_o']}, "
                    f"{expected['activation_read_bank_select_o']}, "
                    f"{expected['activation_local_read_addr_o']}, "
                    f"{expected['weight_read_bank_select_o']}, "
                    f"{expected['weight_local_read_addr_o']}, "
                    f"{expected['tile_compute_en_o']}, {expected['tile_flush_pipeline_o']}, "
                    f"{expected['tile_clear_acc_o']}, {expected['tile_store_results_en_o']}, "
                    f"{expected['result_write_valid_o']}, {expected['result_write_addr_o']}, "
                    f"{expected['result_write_payload_o']}, {expected['result_write_valid_mask_o']}), got "
                    f"({observed['dma_accept_o']}, {observed['load_accept_o']}, "
                    f"{observed['store_accept_o']}, {observed['dma_backpressure_o']}, "
                    f"{observed['load_backpressure_o']}, {observed['store_backpressure_o']}, "
                    f"{observed['tile_dma_valid_o']}, {observed['dma_write_weights_o']}, "
                    f"{observed['dma_bank_select_o']}, {observed['dma_local_addr_o']}, "
                    f"{observed['dma_payload_o']}, {observed['tile_load_vector_en_o']}, "
                    f"{observed['activation_read_bank_select_o']}, "
                    f"{observed['activation_local_read_addr_o']}, "
                    f"{observed['weight_read_bank_select_o']}, "
                    f"{observed['weight_local_read_addr_o']}, "
                    f"{observed['tile_compute_en_o']}, {observed['tile_flush_pipeline_o']}, "
                    f"{observed['tile_clear_acc_o']}, {observed['tile_store_results_en_o']}, "
                    f"{observed['result_write_valid_o']}, {observed['result_write_addr_o']}, "
                    f"{observed['result_write_payload_o']}, {observed['result_write_valid_mask_o']})"
                )

    for case in payload.get("tile_compute_unit", []):
        rows = int(case.get("rows", 2))
        cols = int(case.get("cols", 2))
        depth = int(case.get("depth", 4))
        observed_snapshots = tile_compute_unit_reference(
            steps=case["steps"],
            rows=rows,
            cols=cols,
            depth=depth,
        )
        for step_idx, step in enumerate(case["steps"]):
            expected = step["expected"]
            observed = observed_snapshots[step_idx]
            if (
                observed["psums_o"] != expected["psums_o"]
                or observed["valids_o"] != expected["valids_o"]
                or observed["scratchpad_vector_valid_o"] != expected["scratchpad_vector_valid_o"]
            ):
                failures.append(
                    "tile_compute_unit/"
                    f"{case['name']}/step_{step_idx}: expected "
                    f"({expected['psums_o']}, {expected['valids_o']}, "
                    f"{expected['scratchpad_vector_valid_o']}), got "
                    f"({observed['psums_o']}, {observed['valids_o']}, "
                    f"{observed['scratchpad_vector_valid_o']})"
                )

    for case in list(payload.get("top_npu", [])) + list(payload.get("top_npu_stress", [])):
        rows = int(case.get("rows", 2))
        cols = int(case.get("cols", 2))
        depth = int(case.get("depth", 4))
        tile_count = int(case.get("tile_count", 1))
        observed_snapshots = top_npu_reference(
            steps=case["steps"],
            rows=rows,
            cols=cols,
            depth=depth,
            tile_count=tile_count,
        )
        for step_idx, step in enumerate(case["steps"]):
            expected = step["expected"]
            observed = observed_snapshots[step_idx]
            if (
                observed["scheduler_state_o"] != expected["scheduler_state_o"]
                or observed["busy_o"] != expected["busy_o"]
                or observed["done_o"] != expected["done_o"]
                or observed["result_write_valid_o"] != expected["result_write_valid_o"]
                or observed["result_write_addr_o"] != expected["result_write_addr_o"]
                or observed["result_write_payload_o"] != expected["result_write_payload_o"]
                or observed["result_write_valid_mask_o"] != expected["result_write_valid_mask_o"]
                or observed["psums_o"] != expected["psums_o"]
                or observed["valids_o"] != expected["valids_o"]
            ):
                failures.append(
                    "top_npu/"
                    f"{case['name']}/step_{step_idx}: expected "
                    f"({expected['scheduler_state_o']}, {expected['busy_o']}, "
                    f"{expected['done_o']}, {expected['result_write_valid_o']}, "
                    f"{expected['result_write_addr_o']}, {expected['result_write_payload_o']}, "
                    f"{expected['result_write_valid_mask_o']}, "
                    f"{expected['psums_o']}, "
                    f"{expected['valids_o']}), got "
                    f"({observed['scheduler_state_o']}, {observed['busy_o']}, "
                    f"{observed['done_o']}, {observed['result_write_valid_o']}, "
                    f"{observed['result_write_addr_o']}, {observed['result_write_payload_o']}, "
                    f"{observed['result_write_valid_mask_o']}, "
                    f"{observed['psums_o']}, "
                    f"{observed['valids_o']})"
                )

    if failures:
        return False, "; ".join(failures)

    total_cases = (
        len(payload.get("mac_unit", []))
        + len(payload.get("processing_element", []))
        + sum(len(case["steps"]) for case in payload.get("systolic_tile", []))
        + sum(len(case["steps"]) for case in payload.get("dma_engine", []))
        + sum(len(case["steps"]) for case in payload.get("scratchpad_controller", []))
        + sum(len(case["steps"]) for case in payload.get("accumulator_buffer", []))
        + sum(len(case["steps"]) for case in payload.get("gemm_ctrl", []))
        + sum(len(case["steps"]) for case in payload.get("scheduler", []))
        + sum(len(case["steps"]) for case in payload.get("scheduler_stress", []))
        + sum(len(case["steps"]) for case in payload.get("cluster_control", []))
        + sum(len(case["steps"]) for case in payload.get("cluster_control_stress", []))
        + sum(len(case["steps"]) for case in payload.get("cluster_interconnect", []))
        + sum(len(case["steps"]) for case in payload.get("cluster_interconnect_stress", []))
        + sum(len(case["steps"]) for case in payload.get("tile_compute_unit", []))
        + sum(len(case["steps"]) for case in payload.get("top_npu", []))
        + sum(len(case["steps"]) for case in payload.get("top_npu_stress", []))
    )
    return True, f"Golden model Python valido su {total_cases} casi."


def _cluster_control_outputs(
    scheduler_snapshot: Dict[str, object],
    tile_enable_mask: List[int],
) -> Dict[str, object]:
    sanitized_tile_enable_mask = [int(bool(value)) for value in tile_enable_mask]
    dma_addr = int(scheduler_snapshot.get("dma_addr_o", 0))
    activation_read_addr = int(scheduler_snapshot.get("activation_read_addr_o", 0))
    weight_read_addr = int(scheduler_snapshot.get("weight_read_addr_o", 0))
    dma_valid = int(scheduler_snapshot.get("dma_valid_o", 0))
    load_vector_en = int(scheduler_snapshot.get("load_vector_en_o", 0))
    compute_en = int(scheduler_snapshot.get("compute_en_o", 0))
    flush_pipeline = int(scheduler_snapshot.get("flush_pipeline_o", 0))
    clear_acc = int(scheduler_snapshot.get("clear_acc_o", 0))

    return {
        "tile_dma_valid_o": [
            int(dma_valid and enabled) for enabled in sanitized_tile_enable_mask
        ],
        "dma_write_weights_o": int(scheduler_snapshot.get("dma_write_weights_o", 0)),
        "dma_bank_select_o": _slot_bank_select(dma_addr),
        "dma_local_addr_o": _slot_local_addr(dma_addr),
        "tile_load_vector_en_o": [
            int(load_vector_en and enabled) for enabled in sanitized_tile_enable_mask
        ],
        "activation_read_bank_select_o": _slot_bank_select(activation_read_addr),
        "activation_local_read_addr_o": _slot_local_addr(activation_read_addr),
        "weight_read_bank_select_o": _slot_bank_select(weight_read_addr),
        "weight_local_read_addr_o": _slot_local_addr(weight_read_addr),
        "tile_compute_en_o": [
            int(compute_en and enabled) for enabled in sanitized_tile_enable_mask
        ],
        "tile_flush_pipeline_o": [
            int(flush_pipeline and enabled) for enabled in sanitized_tile_enable_mask
        ],
        "tile_clear_acc_o": [
            int(clear_acc and enabled) for enabled in sanitized_tile_enable_mask
        ],
        "tile_store_results_en_o": [
            int(int(scheduler_snapshot.get("store_results_en_o", 0)) and enabled)
            for enabled in sanitized_tile_enable_mask
        ],
        "store_results_en_o": int(scheduler_snapshot.get("store_results_en_o", 0)),
        "result_write_addr_o": int(scheduler_snapshot.get("result_write_addr_o", 0)),
        "store_burst_index_o": int(scheduler_snapshot.get("store_burst_index_o", 0)),
    }


def _cluster_interconnect_outputs(
    control_snapshot: Dict[str, object],
    dma_payload: List[int],
    psums: List[int],
    rows: int,
    cols: int,
    tile_count: int,
    step: Dict[str, object],
) -> Dict[str, object]:
    sanitized_tile_count = _sanitize_tile_count(tile_count)
    sanitized_dma_payload = [int(value) for value in dma_payload[: max(rows, cols)]]
    if len(sanitized_dma_payload) < max(rows, cols):
        sanitized_dma_payload.extend([0 for _ in range(max(rows, cols) - len(sanitized_dma_payload))])
    accepts = _cluster_interconnect_accepts(
        control_snapshot=control_snapshot,
        step=step,
        tile_count=sanitized_tile_count,
    )
    dma_accept = accepts["dma_accept_i"]
    load_accept = accepts["load_accept_i"]
    store_accept = accepts["store_accept_i"]
    tile_dma_valid = _sanitize_tile_enable_mask(
        control_snapshot.get("tile_dma_valid_o"), sanitized_tile_count
    )
    tile_load_vector_en = _sanitize_tile_enable_mask(
        control_snapshot.get("tile_load_vector_en_o"), sanitized_tile_count
    )
    tile_store_results_en = _sanitize_tile_enable_mask(
        control_snapshot.get("tile_store_results_en_o"), sanitized_tile_count
    )
    store_payload, store_valid_mask = _store_segment_payload(
        psums=[int(value) for value in psums],
        tile_enable_mask=tile_store_results_en if store_accept else [0 for _ in range(sanitized_tile_count)],
        rows=rows,
        cols=cols,
        burst_index=int(control_snapshot.get("store_burst_index_o", 0)),
    )
    result_write_valid = int(control_snapshot.get("store_results_en_o", 0) and store_accept)
    if not result_write_valid:
        store_payload = [0 for _ in psums]
        store_valid_mask = [0 for _ in psums]
    return {
        "dma_accept_o": dma_accept,
        "load_accept_o": load_accept,
        "store_accept_o": store_accept,
        "dma_backpressure_o": int(any(tile_dma_valid) and not dma_accept),
        "load_backpressure_o": int(any(tile_load_vector_en) and not load_accept),
        "store_backpressure_o": int(bool(control_snapshot.get("store_results_en_o", 0)) and not store_accept),
        "tile_dma_valid_o": tile_dma_valid if dma_accept else [0 for _ in range(sanitized_tile_count)],
        "dma_write_weights_o": int(control_snapshot.get("dma_write_weights_o", 0)),
        "dma_bank_select_o": int(bool(control_snapshot.get("dma_bank_select_o", 0))),
        "dma_local_addr_o": int(control_snapshot.get("dma_local_addr_o", 0)),
        "dma_payload_o": sanitized_dma_payload,
        "tile_load_vector_en_o": (
            tile_load_vector_en if load_accept else [0 for _ in range(sanitized_tile_count)]
        ),
        "activation_read_bank_select_o": int(
            bool(control_snapshot.get("activation_read_bank_select_o", 0))
        ),
        "activation_local_read_addr_o": int(control_snapshot.get("activation_local_read_addr_o", 0)),
        "weight_read_bank_select_o": int(bool(control_snapshot.get("weight_read_bank_select_o", 0))),
        "weight_local_read_addr_o": int(control_snapshot.get("weight_local_read_addr_o", 0)),
        "tile_compute_en_o": _sanitize_tile_enable_mask(
            control_snapshot.get("tile_compute_en_o"), sanitized_tile_count
        ),
        "tile_flush_pipeline_o": _sanitize_tile_enable_mask(
            control_snapshot.get("tile_flush_pipeline_o"), sanitized_tile_count
        ),
        "tile_clear_acc_o": _sanitize_tile_enable_mask(
            control_snapshot.get("tile_clear_acc_o"), sanitized_tile_count
        ),
        "tile_store_results_en_o": tile_store_results_en if store_accept else [0 for _ in range(sanitized_tile_count)],
        "result_write_valid_o": result_write_valid,
        "result_write_addr_o": int(control_snapshot.get("result_write_addr_o", 0)),
        "result_write_payload_o": store_payload,
        "result_write_valid_mask_o": store_valid_mask,
    }


def _flatten_matrix(matrix: List[List[int]]) -> List[int]:
    return [value for row in matrix for value in row]


def _cluster_interconnect_accepts(
    control_snapshot: Dict[str, object],
    step: Dict[str, object],
    tile_count: int,
) -> Dict[str, int]:
    tile_dma_valid = _sanitize_tile_enable_mask(
        control_snapshot.get("tile_dma_valid_o"), tile_count
    )
    tile_load_vector_en = _sanitize_tile_enable_mask(
        control_snapshot.get("tile_load_vector_en_o"), tile_count
    )
    dma_ready_mask = _sanitize_ready_mask(step.get("tile_dma_ready_i"), tile_count)
    load_ready_mask = _sanitize_ready_mask(step.get("tile_load_ready_i"), tile_count)
    dma_accept = int(
        (not any(tile_dma_valid))
        or all((not tile_dma_valid[idx]) or dma_ready_mask[idx] for idx in range(tile_count))
    )
    load_accept = int(
        (not any(tile_load_vector_en))
        or all(
            (not tile_load_vector_en[idx]) or load_ready_mask[idx]
            for idx in range(tile_count)
        )
    )
    store_valid = int(control_snapshot.get("store_results_en_o", 0))
    store_accept = int((not store_valid) or _sanitize_accept(step.get("store_ready_i", 1)))
    return {
        "dma_accept_i": dma_accept,
        "load_accept_i": load_accept,
        "store_accept_i": store_accept,
    }


def _scheduler_next_context(
    state: int,
    slot_index: int,
    load_index: int,
    compute_index: int,
    store_index: int,
    step: Dict[str, object],
    rows: int,
    program_slot_count: int,
    program_load_iterations: int,
    program_compute_iterations: int,
    program_clear_on_done: bool,
    program_store_burst_count: int,
    decoupled_mode: bool,
    load_issued: int,
    compute_issued: int,
    store_issued: int,
) -> Tuple[int, int, int, int, int, int, int, int, bool, int, bool, int, int, int]:
    store_burst_count = _sanitize_store_burst_count(
        step.get("store_burst_count_i", rows),
        rows=rows,
    )
    dma_accept = _sanitize_accept(step.get("dma_accept_i", 1))
    load_accept = _sanitize_accept(step.get("load_accept_i", 1))
    store_accept = _sanitize_accept(step.get("store_accept_i", 1))
    state_d = state
    slot_index_d = slot_index
    load_index_d = load_index
    compute_index_d = compute_index
    store_index_d = store_index
    program_slot_count_d = program_slot_count
    program_load_iterations_d = program_load_iterations
    program_compute_iterations_d = program_compute_iterations
    program_clear_on_done_d = program_clear_on_done
    program_store_burst_count_d = program_store_burst_count
    decoupled_mode_d = decoupled_mode
    load_issued_d = load_issued
    compute_issued_d = compute_issued
    store_issued_d = store_issued

    if state == SCHEDULER_IDLE:
        if bool(step["start_i"]):
            state_d = SCHEDULER_DMA_ACT
            slot_index_d = 0
            load_index_d = 0
            compute_index_d = 0
            store_index_d = 0
            program_slot_count_d = _sanitize_slot_count(step.get("slot_count_i", 2))
            program_load_iterations_d = _sanitize_load_iterations(step.get("load_iterations_i", 2))
            program_compute_iterations_d = _sanitize_compute_iterations(step.get("compute_iterations_i", 2))
            program_clear_on_done_d = bool(step.get("clear_on_done_i", 1))
            program_store_burst_count_d = store_burst_count
            decoupled_mode_d = bool(step.get("decoupled_mode_i", 0))
            load_issued_d = 0
            compute_issued_d = 0
            store_issued_d = 0
    elif not decoupled_mode:
        if state == SCHEDULER_DMA_ACT:
            if dma_accept:
                state_d = SCHEDULER_DMA_WGT
        elif state == SCHEDULER_DMA_WGT:
            if dma_accept:
                if slot_index + 1 < program_slot_count:
                    slot_index_d = slot_index + 1
                    state_d = SCHEDULER_DMA_ACT
                else:
                    load_index_d = 0
                    store_index_d = 0
                    state_d = SCHEDULER_LOAD
        elif state == SCHEDULER_LOAD:
            if load_accept:
                load_issued_d = load_issued + 1
                if load_index + 1 < program_load_iterations:
                    load_index_d = load_index + 1
                    state_d = SCHEDULER_LOAD
                elif program_compute_iterations > 0:
                    compute_index_d = 0
                    store_index_d = 0
                    state_d = SCHEDULER_COMPUTE
                elif program_clear_on_done:
                    state_d = SCHEDULER_CLEAR
                else:
                    state_d = SCHEDULER_DONE
        elif state == SCHEDULER_COMPUTE:
            compute_issued_d = compute_issued + 1
            if compute_index + 1 < program_compute_iterations:
                compute_index_d = compute_index + 1
                state_d = SCHEDULER_COMPUTE
            else:
                store_index_d = 0
                state_d = SCHEDULER_STORE
        elif state == SCHEDULER_STORE:
            if store_accept:
                store_issued_d = store_issued + 1
                if store_index + 1 < store_burst_count:
                    store_index_d = store_index + 1
                    state_d = SCHEDULER_STORE
                else:
                    store_index_d = 0
                    state_d = SCHEDULER_FLUSH
        elif state == SCHEDULER_FLUSH:
            state_d = SCHEDULER_CLEAR if program_clear_on_done else SCHEDULER_DONE
        elif state == SCHEDULER_CLEAR:
            store_index_d = 0
            state_d = SCHEDULER_DONE
        elif state == SCHEDULER_DONE:
            store_index_d = 0
            state_d = SCHEDULER_IDLE
    else:
        if state == SCHEDULER_DMA_ACT:
            if dma_accept:
                state_d = SCHEDULER_DMA_WGT
        elif state == SCHEDULER_DMA_WGT:
            if dma_accept:
                if slot_index + 1 < program_slot_count:
                    slot_index_d = slot_index + 1
                    state_d = SCHEDULER_DMA_ACT
                else:
                    load_index_d = 0
                    compute_index_d = 0
                    store_index_d = 0
                    state_d = SCHEDULER_LOAD
        elif state == SCHEDULER_LOAD:
            if load_issued < program_load_iterations and load_accept:
                load_issued_d = load_issued + 1
                load_index_d = load_issued_d
            if compute_issued < program_compute_iterations and load_issued > compute_issued:
                compute_issued_d = compute_issued + 1
                compute_index_d = compute_issued_d
            if (
                store_issued < program_store_burst_count
                and compute_issued > store_issued
                and store_accept
            ):
                store_issued_d = store_issued + 1
                store_index_d = store_issued_d
            if load_issued_d >= program_load_iterations:
                if program_compute_iterations == 0:
                    state_d = SCHEDULER_CLEAR if program_clear_on_done else SCHEDULER_DONE
                elif compute_issued_d < program_compute_iterations:
                    state_d = SCHEDULER_COMPUTE
                elif store_issued_d < program_store_burst_count:
                    state_d = SCHEDULER_STORE
                else:
                    state_d = SCHEDULER_FLUSH
        elif state == SCHEDULER_COMPUTE:
            if compute_issued < program_compute_iterations and load_issued > compute_issued:
                compute_issued_d = compute_issued + 1
                compute_index_d = compute_issued_d
            if (
                store_issued < program_store_burst_count
                and compute_issued > store_issued
                and store_accept
            ):
                store_issued_d = store_issued + 1
                store_index_d = store_issued_d
            if compute_issued_d >= program_compute_iterations:
                if store_issued_d < program_store_burst_count:
                    state_d = SCHEDULER_STORE
                else:
                    state_d = SCHEDULER_FLUSH
        elif state == SCHEDULER_STORE:
            if (
                store_issued < program_store_burst_count
                and compute_issued > store_issued
                and store_accept
            ):
                store_issued_d = store_issued + 1
                store_index_d = store_issued_d
            if store_issued_d >= program_store_burst_count:
                state_d = SCHEDULER_FLUSH
        elif state == SCHEDULER_FLUSH:
            state_d = SCHEDULER_CLEAR if program_clear_on_done else SCHEDULER_DONE
        elif state == SCHEDULER_CLEAR:
            state_d = SCHEDULER_DONE
        elif state == SCHEDULER_DONE:
            state_d = SCHEDULER_IDLE

    return (
        state_d,
        slot_index_d,
        load_index_d,
        compute_index_d,
        store_index_d,
        program_slot_count_d,
        program_load_iterations_d,
        program_compute_iterations_d,
        program_clear_on_done_d,
        program_store_burst_count_d,
        decoupled_mode_d,
        load_issued_d,
        compute_issued_d,
        store_issued_d,
    )


def _scheduler_outputs(
    state: int,
    step: Dict[str, object],
    rows: int,
    cols: int,
    slot_index: int,
    load_index: int,
    store_index: int,
    program_slot_count: int,
    program_load_iterations: int,
    program_compute_iterations: int,
    program_store_burst_count: int,
    decoupled_mode: bool,
    load_issued: int,
    compute_issued: int,
    store_issued: int,
) -> Dict[str, object]:
    max_dim = max(rows, cols)
    slot_stride = _sanitize_stride(step.get("slot_stride_i", 1))
    snapshot = {
        "state_o": state,
        "busy_o": int(state not in (SCHEDULER_IDLE, SCHEDULER_DONE)),
        "done_o": int(state == SCHEDULER_DONE),
        "dma_valid_o": 0,
        "dma_write_weights_o": 0,
        "dma_addr_o": 0,
        "dma_payload_o": [0 for _ in range(max_dim)],
        "load_vector_en_o": 0,
        "activation_read_addr_o": 0,
        "weight_read_addr_o": 0,
        "store_results_en_o": 0,
        "result_write_addr_o": 0,
        "store_burst_index_o": 0,
        "compute_en_o": 0,
        "flush_pipeline_o": 0,
        "clear_acc_o": 0,
        "decoupled_mode_o": int(decoupled_mode),
        "load_queue_depth_o": max(0, int(program_load_iterations) - int(load_issued)),
        "execute_queue_depth_o": max(0, int(load_issued) - int(compute_issued)),
        "store_queue_depth_o": max(0, int(compute_issued) - int(store_issued)),
        "hazard_wait_o": 0,
    }

    if decoupled_mode and state in (SCHEDULER_LOAD, SCHEDULER_COMPUTE, SCHEDULER_STORE):
        if (compute_issued < program_compute_iterations) and not (load_issued > compute_issued):
            snapshot["hazard_wait_o"] = 1
        if (store_issued < program_store_burst_count) and not (compute_issued > store_issued):
            snapshot["hazard_wait_o"] = 1

    if state == SCHEDULER_DMA_ACT:
        snapshot["dma_valid_o"] = 1
        snapshot["dma_addr_o"] = _descriptor_addr(
            base_addr=step.get("activation_base_addr_i", 0),
            index=slot_index,
            stride=slot_stride,
        )
        snapshot["dma_payload_o"] = _select_activation_slot(step=step, slot_index=slot_index, width=max_dim)
    elif state == SCHEDULER_DMA_WGT:
        snapshot["dma_valid_o"] = 1
        snapshot["dma_write_weights_o"] = 1
        snapshot["dma_addr_o"] = _descriptor_addr(
            base_addr=step.get("weight_base_addr_i", 0),
            index=slot_index,
            stride=slot_stride,
        )
        snapshot["dma_payload_o"] = _select_weight_slot(step=step, slot_index=slot_index, width=max_dim)
    elif state == SCHEDULER_LOAD:
        snapshot["load_vector_en_o"] = 1
        snapshot["activation_read_addr_o"] = _descriptor_addr(
            base_addr=step.get("activation_base_addr_i", 0),
            index=_effective_load_addr(
                load_index=load_issued if decoupled_mode else load_index,
                program_slot_count=program_slot_count,
            ),
            stride=slot_stride,
        )
        snapshot["weight_read_addr_o"] = _descriptor_addr(
            base_addr=step.get("weight_base_addr_i", 0),
            index=_effective_load_addr(
                load_index=load_issued if decoupled_mode else load_index,
                program_slot_count=program_slot_count,
            ),
            stride=slot_stride,
        )
        if decoupled_mode and load_issued > compute_issued and compute_issued < int(
            step.get("compute_iterations_i", 0)
        ):
            snapshot["compute_en_o"] = 1
        if decoupled_mode and compute_issued > store_issued and store_issued < program_store_burst_count:
            snapshot["store_results_en_o"] = 1
            snapshot["result_write_addr_o"] = _descriptor_addr(
                base_addr=step.get("result_base_addr_i", 0),
                index=store_issued,
                stride=_sanitize_stride(step.get("store_stride_i", 1)),
            )
            snapshot["store_burst_index_o"] = store_issued
    elif state == SCHEDULER_COMPUTE:
        if not decoupled_mode or (
            load_issued > compute_issued
            and compute_issued < int(step.get("compute_iterations_i", 0))
        ):
            snapshot["compute_en_o"] = 1
        if decoupled_mode and compute_issued > store_issued and store_issued < program_store_burst_count:
            snapshot["store_results_en_o"] = 1
            snapshot["result_write_addr_o"] = _descriptor_addr(
                base_addr=step.get("result_base_addr_i", 0),
                index=store_issued,
                stride=_sanitize_stride(step.get("store_stride_i", 1)),
            )
            snapshot["store_burst_index_o"] = store_issued
    elif state == SCHEDULER_STORE:
        snapshot["store_results_en_o"] = 1
        snapshot["result_write_addr_o"] = _descriptor_addr(
            base_addr=step.get("result_base_addr_i", 0),
            index=store_issued if decoupled_mode else store_index,
            stride=_sanitize_stride(step.get("store_stride_i", 1)),
        )
        snapshot["store_burst_index_o"] = store_issued if decoupled_mode else store_index
    elif state == SCHEDULER_FLUSH:
        snapshot["flush_pipeline_o"] = 1
    elif state == SCHEDULER_CLEAR:
        snapshot["clear_acc_o"] = 1

    return snapshot


def _pad_vector(values: object, width: int) -> List[int]:
    raw_values = [int(value) for value in values]
    return raw_values[:width] + [0 for _ in range(max(0, width - len(raw_values)))]


def _sanitize_slot_count(raw_value: object) -> int:
    value = int(raw_value)
    if value <= 0:
        return 1
    return min(2, value)


def _sanitize_tile_count(raw_value: object) -> int:
    value = int(raw_value)
    if value <= 0:
        return 1
    return min(4, value)


def _sanitize_tile_enable_mask(raw_value: object, tile_count: int) -> List[int]:
    if raw_value is None:
        return [1 for _ in range(tile_count)]
    if isinstance(raw_value, list):
        values = [1 if int(value) else 0 for value in raw_value]
    else:
        values = [1 if int(raw_value) else 0]
    padded = values[:tile_count] + [0 for _ in range(max(0, tile_count - len(values)))]
    return padded


def _sanitize_ready_mask(raw_value: object, tile_count: int) -> List[int]:
    if raw_value is None:
        return [1 for _ in range(tile_count)]
    return _sanitize_tile_enable_mask(raw_value, tile_count)


def _sanitize_accept(raw_value: object) -> int:
    return 1 if int(raw_value) else 0


def _sanitize_compute_iterations(raw_value: object) -> int:
    value = int(raw_value)
    if value <= 0:
        return 0
    return min(15, value)


def _sanitize_load_iterations(raw_value: object) -> int:
    value = int(raw_value)
    if value <= 0:
        return 1
    return min(2, value)


def _sanitize_bank_index(raw_value: object, bank_count: int) -> int:
    if bank_count <= 1:
        return 0
    return int(raw_value) % bank_count


def _sanitize_stride(raw_value: object) -> int:
    value = int(raw_value)
    if value <= 0:
        return 1
    return min(7, value)


def _sanitize_store_burst_count(raw_value: object, rows: int) -> int:
    value = int(raw_value)
    if value <= 0:
        return 1
    return min(max(1, rows), value)


def _descriptor_addr(base_addr: object, index: int, stride: int) -> int:
    return int(base_addr) + (int(index) * int(stride))


def _slot_bank_select(slot_addr: int, bank_count: int = 2) -> int:
    return _sanitize_bank_index(slot_addr, bank_count)


def _slot_local_addr(slot_addr: int, bank_count: int = 2) -> int:
    if bank_count <= 1:
        return int(slot_addr)
    return int(slot_addr) // bank_count


def _select_activation_slot(step: Dict[str, object], slot_index: int, width: int) -> List[int]:
    if slot_index <= 0:
        return _pad_vector(step["activation_slot0_i"], width)
    return _pad_vector(step["activation_slot1_i"], width)


def _select_weight_slot(step: Dict[str, object], slot_index: int, width: int) -> List[int]:
    if slot_index <= 0:
        return _pad_vector(step["weight_slot0_i"], width)
    return _pad_vector(step["weight_slot1_i"], width)


def _effective_load_addr(load_index: int, program_slot_count: int) -> int:
    if program_slot_count <= 1:
        return 0
    return min(load_index, program_slot_count - 1)


def _store_segment_payload(
    psums: List[int],
    tile_enable_mask: List[int],
    rows: int,
    cols: int,
    burst_index: int,
) -> Tuple[List[int], List[int]]:
    payload = [0 for _ in psums]
    valid_mask = [0 for _ in psums]
    if rows <= 0 or cols <= 0:
        return payload, valid_mask

    selected_row = max(0, min(int(burst_index), rows - 1))
    tile_stride = rows * cols
    for tile_index, tile_enabled in enumerate(tile_enable_mask):
        if not tile_enabled:
            continue
        tile_base = tile_index * tile_stride
        row_base = tile_base + (selected_row * cols)
        for col_index in range(cols):
            lane_index = row_base + col_index
            if lane_index >= len(psums):
                continue
            payload[lane_index] = int(psums[lane_index])
            valid_mask[lane_index] = 1
    return payload, valid_mask
