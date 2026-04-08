"""Gemmini-like reference architecture profiles and candidate comparison.

Reference configurations inspired by:
  https://github.com/ucb-bar/gemmini

Used as an external baseline to measure when a ForgeNPU candidate converges
or diverges from the Gemmini architecture family.  No Gemmini source code is
imported — only published architectural parameters are reproduced here.
"""

from typing import Any, Dict, List, Optional

from create_npu.models import ArchitectureCandidate, RequirementSpec


# ---------------------------------------------------------------------------
# Reference configurations
# ---------------------------------------------------------------------------

GEMMINI_REFERENCE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gemmini_small": {
        "name": "gemmini_small",
        "description": "Gemmini 16x16 small config (INT8, weight-stationary, 256 KB scratchpad)",
        "tile_rows": 16,
        "tile_cols": 16,
        "tile_count": 1,
        "dataflow": "weight_stationary",
        "local_sram_kb": 256,
        "accumulator_kb": 64,
        "bus_width_bits": 128,
        "frequency_mhz": 1000.0,
        "estimated_tops": 0.512,
        "architecture_family": "tiled_systolic_array",
        "benchmark_shapes": [
            {
                "name": "resnet50_conv1",
                "op_type": "conv2d",
                "M": 3136,
                "K": 147,
                "N": 64,
                "macs": 3136 * 147 * 64,
            },
            {
                "name": "mobilenet_dw",
                "op_type": "conv2d",
                "M": 784,
                "K": 9,
                "N": 32,
                "macs": 784 * 9 * 32,
            },
            {
                "name": "gemm_256",
                "op_type": "gemm",
                "M": 256,
                "K": 256,
                "N": 256,
                "macs": 256 * 256 * 256,
            },
        ],
    },
    "gemmini_medium": {
        "name": "gemmini_medium",
        "description": "Gemmini 16x16 medium config (INT8, weight-stationary, 512 KB scratchpad)",
        "tile_rows": 16,
        "tile_cols": 16,
        "tile_count": 1,
        "dataflow": "weight_stationary",
        "local_sram_kb": 512,
        "accumulator_kb": 128,
        "bus_width_bits": 256,
        "frequency_mhz": 1000.0,
        "estimated_tops": 0.512,
        "architecture_family": "tiled_systolic_array",
        "benchmark_shapes": [
            {
                "name": "resnet50_fc",
                "op_type": "gemm",
                "M": 1,
                "K": 2048,
                "N": 1000,
                "macs": 1 * 2048 * 1000,
            },
            {
                "name": "bert_attn_128",
                "op_type": "gemm",
                "M": 128,
                "K": 768,
                "N": 768,
                "macs": 128 * 768 * 768,
            },
            {
                "name": "gemm_512",
                "op_type": "gemm",
                "M": 512,
                "K": 512,
                "N": 512,
                "macs": 512 * 512 * 512,
            },
        ],
    },
    "gemmini_large": {
        "name": "gemmini_large",
        "description": "Gemmini 32x32 large config (INT8, weight-stationary, 4 MB scratchpad)",
        "tile_rows": 32,
        "tile_cols": 32,
        "tile_count": 1,
        "dataflow": "weight_stationary",
        "local_sram_kb": 4096,
        "accumulator_kb": 512,
        "bus_width_bits": 512,
        "frequency_mhz": 1000.0,
        "estimated_tops": 2.048,
        "architecture_family": "tiled_systolic_array",
        "benchmark_shapes": [
            {
                "name": "llm_fc1",
                "op_type": "gemm",
                "M": 512,
                "K": 4096,
                "N": 4096,
                "macs": 512 * 4096 * 4096,
            },
            {
                "name": "resnet50_layer3",
                "op_type": "gemm",
                "M": 784,
                "K": 512,
                "N": 1024,
                "macs": 784 * 512 * 1024,
            },
            {
                "name": "gemm_1024",
                "op_type": "gemm",
                "M": 1024,
                "K": 1024,
                "N": 1024,
                "macs": 1024 * 1024 * 1024,
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Reference selection
# ---------------------------------------------------------------------------

def select_nearest_gemmini_reference(architecture: ArchitectureCandidate) -> str:
    """Pick the closest Gemmini reference config based on candidate estimated_tops."""
    tops = float(architecture.estimated_tops)
    if tops >= 1.5:
        return "gemmini_large"
    if tops >= 0.3:
        return "gemmini_medium"
    return "gemmini_small"


# ---------------------------------------------------------------------------
# Delta computation helpers
# ---------------------------------------------------------------------------

def _dataflow_from_family(family: str) -> str:
    return {
        "tiled_systolic_array": "weight_stationary",
        "tiled_systolic_transformer": "weight_stationary",
        "weight_stationary_array": "weight_stationary",
        "output_stationary_array": "output_stationary",
        "sparse_pe_mesh": "sparse_streaming",
    }.get(family, "weight_stationary")


def _convergence_score(
    candidate_dataflow: str,
    candidate_family: str,
    reference: Dict[str, Any],
) -> float:
    """Return a 0.0-1.0 similarity score against the given reference."""
    score = 0.0
    # Dataflow match: 40 points
    if candidate_dataflow == reference["dataflow"]:
        score += 0.40
    elif candidate_dataflow in ("weight_stationary", "systolic"):
        score += 0.25
    # Architecture family: 40 points
    if candidate_family == reference["architecture_family"]:
        score += 0.40
    elif candidate_family in ("tiled_systolic_array", "tiled_systolic_transformer"):
        score += 0.30
    # Dense/non-sparse orientation: 20 points
    if candidate_dataflow not in ("sparse_streaming",):
        score += 0.20
    return round(min(1.0, score), 2)


def _throughput_comparison(
    architecture: ArchitectureCandidate,
    reference: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Estimate speedup of the candidate vs the reference for each benchmark shape."""
    candidate_tops = float(architecture.estimated_tops)
    reference_tops = float(reference["estimated_tops"])
    comparisons = []
    for shape in reference["benchmark_shapes"]:
        speedup = round(candidate_tops / max(1e-9, reference_tops), 2)
        comparisons.append(
            {
                "benchmark": shape["name"],
                "op_type": shape.get("op_type", "gemm"),
                "macs": int(shape["macs"]),
                "candidate_estimated_tops": candidate_tops,
                "reference_estimated_tops": reference_tops,
                "estimated_speedup": speedup,
            }
        )
    return comparisons


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_gemmini_delta(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
    compiled_program: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute a delta between the candidate and the nearest Gemmini reference.

    Returns a dictionary suitable for embedding in the execution report summary
    under the key ``gemmini_reference_delta``.
    """
    reference_name = select_nearest_gemmini_reference(architecture)
    reference = GEMMINI_REFERENCE_CONFIGS[reference_name]

    # Prefer the dataflow actually compiled into the program
    candidate_dataflow = _dataflow_from_family(architecture.family)
    if compiled_program:
        compiled_df = (compiled_program.get("mapping_plan") or {}).get("dataflow", "")
        if compiled_df and compiled_df not in ("sparse", ""):
            candidate_dataflow = compiled_df

    candidate_total_sram_kb = (
        int(architecture.local_sram_kb_per_tile) * int(architecture.tile_count)
    )
    reference_total_sram_kb = int(reference["local_sram_kb"])
    candidate_pe_count = int(architecture.pe_count)
    reference_pe_count = (
        int(reference["tile_rows"]) * int(reference["tile_cols"]) * int(reference["tile_count"])
    )

    tops_ratio = round(
        float(architecture.estimated_tops) / max(1e-9, float(reference["estimated_tops"])), 2
    )
    sram_ratio = round(candidate_total_sram_kb / max(1, reference_total_sram_kb), 2)
    pe_ratio = round(candidate_pe_count / max(1, reference_pe_count), 2)

    dataflow_match = candidate_dataflow == reference["dataflow"]
    family_match = architecture.family == reference["architecture_family"]

    conv_score = _convergence_score(candidate_dataflow, architecture.family, reference)
    if conv_score >= 0.70:
        convergence = "converges"
    elif conv_score >= 0.40:
        convergence = "partial"
    else:
        convergence = "diverges"

    # Human-readable rationale
    reasons: List[str] = []
    if dataflow_match:
        reasons.append(
            f"Dataflow '{candidate_dataflow}' corrisponde al riferimento Gemmini"
            f" ({reference['dataflow']})."
        )
    else:
        reasons.append(
            f"Dataflow '{candidate_dataflow}' non corrisponde a '{reference['dataflow']}'"
            " del riferimento Gemmini."
        )
    if family_match:
        reasons.append(
            f"Famiglia architetturale '{architecture.family}' identica al riferimento."
        )
    else:
        reasons.append(
            f"Famiglia '{architecture.family}' diversa da"
            f" '{reference['architecture_family']}' del riferimento."
        )
    reasons.append(
        f"Throughput {architecture.estimated_tops:.2f} TOPS candidato vs"
        f" {reference['estimated_tops']:.3f} TOPS riferimento (ratio {tops_ratio}x)."
    )
    reasons.append(
        f"SRAM totale {candidate_total_sram_kb} KB candidato vs"
        f" {reference_total_sram_kb} KB riferimento (ratio {sram_ratio}x)."
    )

    # Requirement vs mapping delta
    requirement_dataflow = getattr(spec, "preferred_dataflow", "auto")
    mapping_chosen = candidate_dataflow
    req_vs_mapping: List[str] = []
    if requirement_dataflow == "auto":
        req_vs_mapping.append(
            f"Requirement dataflow 'auto' -> mapping scelto '{mapping_chosen}'."
        )
    elif requirement_dataflow == mapping_chosen:
        req_vs_mapping.append(
            f"Requirement dataflow '{requirement_dataflow}' soddisfatto dal mapping."
        )
    else:
        req_vs_mapping.append(
            f"Requirement dataflow '{requirement_dataflow}' non corrisponde al mapping"
            f" scelto '{mapping_chosen}'."
        )

    return {
        "reference_name": reference_name,
        "reference_config": {
            "name": reference["name"],
            "description": reference["description"],
            "tile_rows": reference["tile_rows"],
            "tile_cols": reference["tile_cols"],
            "dataflow": reference["dataflow"],
            "local_sram_kb": reference["local_sram_kb"],
            "accumulator_kb": reference["accumulator_kb"],
            "frequency_mhz": reference["frequency_mhz"],
            "estimated_tops": reference["estimated_tops"],
            "architecture_family": reference["architecture_family"],
        },
        "candidate_vs_reference": {
            "pe_count": {
                "candidate": candidate_pe_count,
                "reference": reference_pe_count,
                "ratio": pe_ratio,
            },
            "total_sram_kb": {
                "candidate": candidate_total_sram_kb,
                "reference": reference_total_sram_kb,
                "ratio": sram_ratio,
            },
            "estimated_tops": {
                "candidate": float(architecture.estimated_tops),
                "reference": float(reference["estimated_tops"]),
                "ratio": tops_ratio,
            },
            "dataflow_match": dataflow_match,
            "architecture_family_match": family_match,
        },
        "convergence": convergence,
        "convergence_score": conv_score,
        "convergence_reasons": reasons,
        "requirement_vs_mapping": req_vs_mapping,
        "throughput_comparison": _throughput_comparison(
            architecture=architecture,
            reference=reference,
        ),
    }
