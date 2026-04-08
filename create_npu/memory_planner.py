"""SRAM memory planner for ForgeNPU compiled programs.

Computes tensor allocations with liveness-based reuse, reporting peak SRAM
usage both without reuse (naive) and with reuse (overlap of non-live tensors).
Inspired by the memory planning patterns in tiny-NPU:
  https://github.com/harishsg993010/tiny-NPU

Algorithm
---------
1. Filter the per-operator tensor descriptors (skip the 3 seed-level aliases).
2. Build a liveness interval for each tensor: [birth_step, death_step].
   - Inputs (activation/weight): live only during their operator's step.
   - Outputs: live from their operator's step until the last consumer's step
     (if consumer_ops is populated) or step+1 (sequential heuristic).
3. No-reuse baseline: every tensor occupies a unique, non-overlapping region.
4. With-reuse: greedy first-fit allocator over the free-list; tensors whose
   intervals do not overlap can share SRAM.
5. Compute per-step live bytes and peak SRAM for both strategies.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Seed-level summary descriptors — excluded from per-operator planning.
_SEED_LEVEL_NAMES = {"activation_in", "weight_in", "result_out"}


@dataclass
class MemoryInterval:
    """Allocation record for one tensor in the memory plan."""

    tensor_name: str
    role: str
    operator_name: str
    birth_step: int           # first step where tensor is needed (inclusive)
    death_step: int           # last step where tensor is needed (inclusive)
    size_bytes: int
    base_addr_no_reuse: int   # address if all tensors are independently allocated
    base_addr_with_reuse: int # address after liveness-based reuse

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryPlan:
    """Complete SRAM memory plan for a compiled program."""

    operator_count: int
    tensor_count: int
    peak_sram_bytes_no_reuse: int
    peak_sram_bytes_with_reuse: int
    reuse_savings_bytes: int
    reuse_savings_pct: float
    per_step_live_bytes: List[int]   # bytes live at each execution step
    allocations: List[Dict[str, Any]]  # serialised MemoryInterval list

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_memory(
    tensor_descriptors: List[Dict[str, Any]],
    operator_descriptors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the memory planner and return a serialisable MemoryPlan dict.

    Parameters
    ----------
    tensor_descriptors:
        The ``tensor_descriptors`` list from ``CompiledProgram``.
    operator_descriptors:
        The ``operator_descriptors`` list from ``CompiledProgram``.
    """
    if not operator_descriptors:
        return MemoryPlan(
            operator_count=0,
            tensor_count=0,
            peak_sram_bytes_no_reuse=0,
            peak_sram_bytes_with_reuse=0,
            reuse_savings_bytes=0,
            reuse_savings_pct=0.0,
            per_step_live_bytes=[],
            allocations=[],
        ).to_dict()

    op_step_map: Dict[str, int] = {
        str(op.get("name", op.get("op_type", f"op{i}"))): i
        for i, op in enumerate(operator_descriptors)
    }
    n_steps = len(operator_descriptors)

    # Keep only per-operator descriptors
    per_op = [
        d for d in tensor_descriptors
        if d.get("name") not in _SEED_LEVEL_NAMES
    ]

    if not per_op:
        return MemoryPlan(
            operator_count=n_steps,
            tensor_count=0,
            peak_sram_bytes_no_reuse=0,
            peak_sram_bytes_with_reuse=0,
            reuse_savings_bytes=0,
            reuse_savings_pct=0.0,
            per_step_live_bytes=[0] * n_steps,
            allocations=[],
        ).to_dict()

    intervals = _build_intervals(per_op, op_step_map, n_steps)
    intervals = _assign_no_reuse_addrs(intervals)
    intervals = _greedy_allocate(intervals, n_steps)

    peak_no_reuse = sum(iv.size_bytes for iv in intervals)
    per_step = _per_step_live_bytes(intervals, n_steps)
    peak_with_reuse = max(per_step) if per_step else 0
    savings = peak_no_reuse - peak_with_reuse
    savings_pct = round(100.0 * savings / max(1, peak_no_reuse), 2)

    return MemoryPlan(
        operator_count=n_steps,
        tensor_count=len(intervals),
        peak_sram_bytes_no_reuse=peak_no_reuse,
        peak_sram_bytes_with_reuse=peak_with_reuse,
        reuse_savings_bytes=savings,
        reuse_savings_pct=savings_pct,
        per_step_live_bytes=per_step,
        allocations=[iv.to_dict() for iv in intervals],
    ).to_dict()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_intervals(
    per_op: List[Dict[str, Any]],
    op_step_map: Dict[str, int],
    n_steps: int,
) -> List["MemoryInterval"]:
    intervals: List[MemoryInterval] = []
    for desc in per_op:
        op_name = str(desc.get("operator_name", ""))
        step = op_step_map.get(op_name, 0)
        role = str(desc.get("role", ""))
        size = int(desc.get("size_bytes", 0))

        if role in ("activation_input", "weight_input"):
            birth = step
            death = step
        else:  # output
            consumer_ops: List[str] = desc.get("consumer_ops") or []
            if consumer_ops:
                death = max(op_step_map.get(c, step) for c in consumer_ops)
            else:
                # Sequential heuristic: result needed one step beyond its producer
                death = min(step + 1, n_steps - 1)
            birth = step

        intervals.append(
            MemoryInterval(
                tensor_name=str(desc.get("name", "")),
                role=role,
                operator_name=op_name,
                birth_step=birth,
                death_step=death,
                size_bytes=size,
                base_addr_no_reuse=0,
                base_addr_with_reuse=0,
            )
        )
    return intervals


def _assign_no_reuse_addrs(intervals: List[MemoryInterval]) -> List[MemoryInterval]:
    """Assign sequential, non-overlapping addresses for the no-reuse baseline."""
    addr = 0
    for iv in intervals:
        iv.base_addr_no_reuse = addr
        addr += iv.size_bytes
    return intervals


def _greedy_allocate(
    intervals: List[MemoryInterval],
    n_steps: int,
) -> List[MemoryInterval]:
    """Greedy first-fit SRAM allocator with a free-list for liveness reuse.

    Processes steps in order; at each step, first frees tensors whose
    ``death_step`` was the previous step, then allocates tensors whose
    ``birth_step`` equals the current step.
    """
    # addr -> size of currently free (released) regions
    free_regions: List[Tuple[int, int]] = []
    high_water = 0

    # Index for quick lookup
    born_at: Dict[int, List[MemoryInterval]] = {}
    for iv in intervals:
        born_at.setdefault(iv.birth_step, []).append(iv)

    died_at: Dict[int, List[MemoryInterval]] = {}
    for iv in intervals:
        died_at.setdefault(iv.death_step, []).append(iv)

    for step in range(n_steps):
        # Release tensors that finished at the previous step
        for iv in died_at.get(step - 1, []):
            _free_region(free_regions, iv.base_addr_with_reuse, iv.size_bytes)

        # Allocate tensors born at this step (stable sort: larger first → better packing)
        for iv in sorted(born_at.get(step, []), key=lambda x: -x.size_bytes):
            addr = _find_free(free_regions, iv.size_bytes)
            if addr is None:
                addr = high_water
                high_water += iv.size_bytes
            iv.base_addr_with_reuse = addr

    # Release remaining live tensors to measure final high-water mark
    # (nothing to do for the plan itself)
    return intervals


def _free_region(
    free_regions: List[Tuple[int, int]],
    addr: int,
    size: int,
) -> None:
    """Add a freed region and merge adjacent free blocks."""
    free_regions.append((addr, size))
    free_regions.sort(key=lambda r: r[0])
    # Merge adjacent/overlapping blocks
    merged: List[Tuple[int, int]] = []
    for start, length in free_regions:
        if merged and merged[-1][0] + merged[-1][1] >= start:
            prev_start, prev_len = merged[-1]
            merged[-1] = (prev_start, max(prev_len, start + length - prev_start))
        else:
            merged.append((start, length))
    free_regions[:] = merged


def _find_free(
    free_regions: List[Tuple[int, int]],
    size: int,
) -> Optional[int]:
    """First-fit: return the start address of the first region large enough."""
    for i, (addr, region_size) in enumerate(free_regions):
        if region_size >= size:
            remainder = region_size - size
            if remainder > 0:
                free_regions[i] = (addr + size, remainder)
            else:
                free_regions.pop(i)
            return addr
    return None


def _per_step_live_bytes(
    intervals: List[MemoryInterval],
    n_steps: int,
) -> List[int]:
    """Return total bytes live at each execution step (with reuse addressing)."""
    result = []
    for step in range(n_steps):
        live = sum(
            iv.size_bytes
            for iv in intervals
            if iv.birth_step <= step <= iv.death_step
        )
        result.append(live)
    return result
