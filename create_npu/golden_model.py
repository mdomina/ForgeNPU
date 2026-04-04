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
        activations_west = [int(value) for value in step["activations_west_i"]]
        weights_north = [int(value) for value in step["weights_north_i"]]

        if flush_pipeline:
            activation_regs = [[0 for _ in range(cols)] for _ in range(rows)]
            weight_regs = [[0 for _ in range(cols)] for _ in range(rows)]
        elif load_inputs_en:
            next_activations = [[0 for _ in range(cols)] for _ in range(rows)]
            next_weights = [[0 for _ in range(cols)] for _ in range(rows)]

            for row in range(rows):
                for col in range(cols):
                    if col == 0:
                        next_activations[row][col] = activations_west[row]
                    else:
                        next_activations[row][col] = activation_regs[row][col - 1]

            for row in range(rows):
                for col in range(cols):
                    if row == 0:
                        next_weights[row][col] = weights_north[col]
                    else:
                        next_weights[row][col] = weight_regs[row - 1][col]

            activation_regs = next_activations
            weight_regs = next_weights

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
            for row in range(rows):
                for col in range(cols - 1, -1, -1):
                    if col == 0:
                        next_activation_regs[row][col] = activations_west[row]
                    else:
                        next_activation_regs[row][col] = prev_activation_regs[row][col - 1]

            for row in range(rows - 1, -1, -1):
                for col in range(cols):
                    if row == 0:
                        next_weight_regs[row][col] = weights_north[col]
                    else:
                        next_weight_regs[row][col] = prev_weight_regs[row - 1][col]

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
            )
        )

    return snapshots


def top_npu_reference(
    steps: List[Dict[str, object]],
    rows: int = 2,
    cols: int = 2,
    depth: int = 4,
    tile_count: int = 1,
) -> List[Dict[str, object]]:
    scheduler_snapshots = scheduler_reference(steps=steps, rows=rows, cols=cols)
    sanitized_tile_count = _sanitize_tile_count(tile_count)
    tile_enable_masks = [
        _sanitize_tile_enable_mask(step.get("tile_enable_i"), sanitized_tile_count)
        for step in steps
    ]
    idle_scheduler_snapshot = {
        "dma_valid_o": 0,
        "dma_write_weights_o": 0,
        "dma_addr_o": 0,
        "dma_payload_o": [0 for _ in range(max(rows, cols))],
        "load_vector_en_o": 0,
        "activation_read_addr_o": 0,
        "weight_read_addr_o": 0,
        "compute_en_o": 0,
        "flush_pipeline_o": 0,
        "clear_acc_o": 0,
        "store_burst_index_o": 0,
    }

    tile_snapshots_by_index: List[List[Dict[str, object]]] = []
    for tile_index in range(sanitized_tile_count):
        tile_steps = []
        for step_index, tile_enable_mask in enumerate(tile_enable_masks):
            scheduler_snapshot = (
                scheduler_snapshots[step_index - 1]
                if step_index > 0
                else idle_scheduler_snapshot
            )
            tile_enabled = bool(tile_enable_mask[tile_index])
            dma_slot_addr = int(scheduler_snapshot["dma_addr_o"])
            activation_slot_addr = int(scheduler_snapshot["activation_read_addr_o"])
            weight_slot_addr = int(scheduler_snapshot["weight_read_addr_o"])
            tile_steps.append(
                {
                    "dma_valid_i": scheduler_snapshot["dma_valid_o"] if tile_enabled else 0,
                    "dma_write_weights_i": scheduler_snapshot["dma_write_weights_o"],
                    "dma_addr_i": _slot_local_addr(dma_slot_addr),
                    "dma_payload_i": scheduler_snapshot["dma_payload_o"],
                    "activation_write_bank_i": _slot_bank_select(dma_slot_addr),
                    "weight_write_bank_i": _slot_bank_select(dma_slot_addr),
                    "load_vector_en_i": scheduler_snapshot["load_vector_en_o"] if tile_enabled else 0,
                    "activation_read_bank_i": _slot_bank_select(activation_slot_addr),
                    "activation_read_addr_i": _slot_local_addr(activation_slot_addr),
                    "weight_read_bank_i": _slot_bank_select(weight_slot_addr),
                    "weight_read_addr_i": _slot_local_addr(weight_slot_addr),
                    "compute_en_i": scheduler_snapshot["compute_en_o"] if tile_enabled else 0,
                    "flush_pipeline_i": scheduler_snapshot["flush_pipeline_o"] if tile_enabled else 0,
                    "clear_acc_i": scheduler_snapshot["clear_acc_o"] if tile_enabled else 0,
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

    snapshots: List[Dict[str, object]] = []
    for step_index, scheduler_snapshot in enumerate(scheduler_snapshots):
        psums: List[int] = []
        valids: List[int] = []
        for tile_index in range(sanitized_tile_count):
            tile_snapshot = tile_snapshots_by_index[tile_index][step_index]
            psums.extend([int(value) for value in tile_snapshot["psums_o"]])
            valids.extend([int(value) for value in tile_snapshot["valids_o"]])
        store_payload, store_valid_mask = _store_segment_payload(
            psums=psums,
            tile_enable_mask=tile_enable_masks[step_index],
            rows=rows,
            cols=cols,
            burst_index=int(scheduler_snapshot.get("store_burst_index_o", 0)),
        )

        snapshots.append(
            {
                "scheduler_state_o": scheduler_snapshot["state_o"],
                "busy_o": scheduler_snapshot["busy_o"],
                "done_o": scheduler_snapshot["done_o"],
                "result_write_valid_o": scheduler_snapshot["store_results_en_o"],
                "result_write_addr_o": scheduler_snapshot["result_write_addr_o"],
                "result_write_payload_o": (
                    store_payload if int(scheduler_snapshot["store_results_en_o"]) else [0 for _ in psums]
                ),
                "result_write_valid_mask_o": (
                    store_valid_mask if int(scheduler_snapshot["store_results_en_o"]) else [0 for _ in valids]
                ),
                "psums_o": psums,
                "valids_o": valids,
            }
        )

    return snapshots


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

    for case in payload.get("scheduler", []):
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
                    f"{expected['clear_acc_o']}), got "
                    f"({observed['state_o']}, {observed['busy_o']}, {observed['done_o']}, "
                    f"{observed['dma_valid_o']}, {observed['dma_write_weights_o']}, "
                    f"{observed['dma_addr_o']}, {observed['dma_payload_o']}, "
                    f"{observed['load_vector_en_o']}, {observed['activation_read_addr_o']}, "
                    f"{observed['weight_read_addr_o']}, {observed['store_results_en_o']}, "
                    f"{observed['result_write_addr_o']}, {observed['store_burst_index_o']}, "
                    f"{observed['compute_en_o']}, {observed['flush_pipeline_o']}, "
                    f"{observed['clear_acc_o']})"
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

    for case in payload.get("top_npu", []):
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
        + sum(len(case["steps"]) for case in payload.get("scheduler", []))
        + sum(len(case["steps"]) for case in payload.get("tile_compute_unit", []))
        + sum(len(case["steps"]) for case in payload.get("top_npu", []))
    )
    return True, f"Golden model Python valido su {total_cases} casi."


def _flatten_matrix(matrix: List[List[int]]) -> List[int]:
    return [value for row in matrix for value in row]


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
) -> Tuple[int, int, int, int, int, int, int, int, bool]:
    store_burst_count = _sanitize_store_burst_count(
        step.get("store_burst_count_i", rows),
        rows=rows,
    )
    if state == SCHEDULER_IDLE:
        if bool(step["start_i"]):
            return (
                SCHEDULER_DMA_ACT,
                0,
                0,
                0,
                0,
                _sanitize_slot_count(step.get("slot_count_i", 2)),
                _sanitize_load_iterations(step.get("load_iterations_i", 2)),
                _sanitize_compute_iterations(step.get("compute_iterations_i", 2)),
                bool(step.get("clear_on_done_i", 1)),
            )
        return (
            SCHEDULER_IDLE,
            slot_index,
            load_index,
            compute_index,
            store_index,
            program_slot_count,
            program_load_iterations,
            program_compute_iterations,
            program_clear_on_done,
        )
    if state == SCHEDULER_DMA_ACT:
        return (
            SCHEDULER_DMA_WGT,
            slot_index,
            load_index,
            compute_index,
            store_index,
            program_slot_count,
            program_load_iterations,
            program_compute_iterations,
            program_clear_on_done,
        )
    if state == SCHEDULER_DMA_WGT:
        if slot_index + 1 < program_slot_count:
            return (
                SCHEDULER_DMA_ACT,
                slot_index + 1,
                load_index,
                compute_index,
                store_index,
                program_slot_count,
                program_load_iterations,
                program_compute_iterations,
                program_clear_on_done,
            )
        return (
            SCHEDULER_LOAD,
            slot_index,
            0,
            compute_index,
            0,
            program_slot_count,
            program_load_iterations,
            program_compute_iterations,
            program_clear_on_done,
        )
    if state == SCHEDULER_LOAD:
        if load_index + 1 < program_load_iterations:
            return (
                SCHEDULER_LOAD,
                slot_index,
                load_index + 1,
                compute_index,
                store_index,
                program_slot_count,
                program_load_iterations,
                program_compute_iterations,
                program_clear_on_done,
            )
        if program_compute_iterations > 0:
            return (
                SCHEDULER_COMPUTE,
                slot_index,
                load_index,
                0,
                0,
                program_slot_count,
                program_load_iterations,
                program_compute_iterations,
                program_clear_on_done,
            )
        if program_clear_on_done:
            return (
                SCHEDULER_CLEAR,
                slot_index,
                load_index,
                compute_index,
                0,
                program_slot_count,
                program_load_iterations,
                program_compute_iterations,
                program_clear_on_done,
            )
        return (
            SCHEDULER_DONE,
            slot_index,
            load_index,
            compute_index,
            0,
            program_slot_count,
            program_load_iterations,
            program_compute_iterations,
            program_clear_on_done,
        )
    if state == SCHEDULER_COMPUTE:
        if compute_index + 1 < program_compute_iterations:
            return (
                SCHEDULER_COMPUTE,
                slot_index,
                load_index,
                compute_index + 1,
                0,
                program_slot_count,
                program_load_iterations,
                program_compute_iterations,
                program_clear_on_done,
            )
        return (
            SCHEDULER_STORE,
            slot_index,
            load_index,
            compute_index,
            0,
            program_slot_count,
            program_load_iterations,
            program_compute_iterations,
            program_clear_on_done,
        )
    if state == SCHEDULER_STORE:
        if store_index + 1 < store_burst_count:
            return (
                SCHEDULER_STORE,
                slot_index,
                load_index,
                compute_index,
                store_index + 1,
                program_slot_count,
                program_load_iterations,
                program_compute_iterations,
                program_clear_on_done,
            )
        return (
            SCHEDULER_FLUSH,
            slot_index,
            load_index,
            compute_index,
            0,
            program_slot_count,
            program_load_iterations,
            program_compute_iterations,
            program_clear_on_done,
        )
    if state == SCHEDULER_FLUSH:
        if program_clear_on_done:
            return (
                SCHEDULER_CLEAR,
                slot_index,
                load_index,
                compute_index,
                0,
                program_slot_count,
                program_load_iterations,
                program_compute_iterations,
                program_clear_on_done,
            )
        return (
            SCHEDULER_DONE,
            slot_index,
            load_index,
            compute_index,
            0,
            program_slot_count,
            program_load_iterations,
            program_compute_iterations,
            program_clear_on_done,
        )
    if state == SCHEDULER_CLEAR:
        return (
            SCHEDULER_DONE,
            slot_index,
            load_index,
            compute_index,
            0,
            program_slot_count,
            program_load_iterations,
            program_compute_iterations,
            program_clear_on_done,
        )
    if state == SCHEDULER_DONE:
        return (
            SCHEDULER_IDLE,
            slot_index,
            load_index,
            compute_index,
            0,
            program_slot_count,
            program_load_iterations,
            program_compute_iterations,
            program_clear_on_done,
        )
    return (
        SCHEDULER_IDLE,
        slot_index,
        load_index,
        compute_index,
        0,
        program_slot_count,
        program_load_iterations,
        program_compute_iterations,
        program_clear_on_done,
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
    }

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
                load_index=load_index,
                program_slot_count=program_slot_count,
            ),
            stride=slot_stride,
        )
        snapshot["weight_read_addr_o"] = _descriptor_addr(
            base_addr=step.get("weight_base_addr_i", 0),
            index=_effective_load_addr(
                load_index=load_index,
                program_slot_count=program_slot_count,
            ),
            stride=slot_stride,
        )
    elif state == SCHEDULER_COMPUTE:
        snapshot["compute_en_o"] = 1
    elif state == SCHEDULER_STORE:
        snapshot["store_results_en_o"] = 1
        snapshot["result_write_addr_o"] = _descriptor_addr(
            base_addr=step.get("result_base_addr_i", 0),
            index=store_index,
            stride=_sanitize_stride(step.get("store_stride_i", 1)),
        )
        snapshot["store_burst_index_o"] = store_index
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
