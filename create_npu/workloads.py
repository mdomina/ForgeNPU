from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class WorkloadProfile:
    workload_type: str
    family: str
    detection_patterns: Tuple[str, ...]
    compatible_families: Tuple[str, ...]
    local_sram_kb_delta: int = 0
    global_buffer_mb_delta: int = 0
    bus_width_bits_delta: int = 0
    score_bonus: float = 10.0
    module_hints: Tuple[str, ...] = ()
    rationale: str = ""
    report_summary: str = ""


WORKLOAD_PROFILES: Dict[str, WorkloadProfile] = {
    "transformer": WorkloadProfile(
        workload_type="transformer",
        family="tiled_systolic_transformer",
        detection_patterns=(
            r"\btransformer\b",
            r"\battention\b",
            r"\bllm\b",
            r"\bgpt\b",
        ),
        compatible_families=(
            "tiled_systolic_transformer",
            "tiled_systolic_array",
        ),
        local_sram_kb_delta=256,
        global_buffer_mb_delta=1,
        bus_width_bits_delta=256,
        score_bonus=12.0,
        rationale=(
            "Privilegio una mesh systolic con buffering piu' ampio per sequenze, "
            "reuse dei pesi e traffico di attivazioni stile attention/MLP."
        ),
        report_summary=(
            "Profilo transformer: mesh systolic con buffering piu' ampio per "
            "sequenze e forte reuse dei tensori."
        ),
    ),
    "convolution": WorkloadProfile(
        workload_type="convolution",
        family="weight_stationary_array",
        detection_patterns=(
            r"\bconvolution\b",
            r"\bconv2d\b",
            r"\bconv\b",
            r"\bcnn\b",
            r"\bresnet\b",
            r"\bmobilenet\b",
            r"\bconvoluz(?:ione|ionale)\b",
        ),
        compatible_families=("weight_stationary_array",),
        local_sram_kb_delta=128,
        global_buffer_mb_delta=1,
        bus_width_bits_delta=256,
        score_bonus=12.0,
        module_hints=("line_buffer",),
        rationale=(
            "Favorisco dataflow weight-stationary e buffering dei kernel per "
            "ridurre il traffico dei pesi nel path di convoluzione."
        ),
        report_summary=(
            "Profilo convolution: dataflow weight-stationary con buffering dei "
            "kernel e banda esterna rinforzata."
        ),
    ),
    "sparse_linear_algebra": WorkloadProfile(
        workload_type="sparse_linear_algebra",
        family="sparse_pe_mesh",
        detection_patterns=(
            r"\bsparse\s+matmul\b",
            r"\bsparse\s+gemm\b",
            r"\bspmm\b",
            r"\bspmv\b",
            r"\bsparse\b",
            r"\bsparsity\b",
        ),
        compatible_families=("sparse_pe_mesh",),
        local_sram_kb_delta=64,
        global_buffer_mb_delta=1,
        score_bonus=12.0,
        module_hints=("sparsity_decoder",),
        rationale=(
            "Introduco una mesh orientata alla sparsity con buffering dei "
            "metadata e dispatch piu' irregolare dei non-zero."
        ),
        report_summary=(
            "Profilo sparse_linear_algebra: mesh dedicata alla sparsity con "
            "supporto ai metadata dei non-zero."
        ),
    ),
    "dense_gemm": WorkloadProfile(
        workload_type="dense_gemm",
        family="tiled_systolic_array",
        detection_patterns=(
            r"\bdense\s+gemm\b",
            r"\bgemm\b",
            r"\bmatmul\b",
            r"\bmatrix\s+multiply\b",
            r"\bdense\b",
        ),
        compatible_families=(
            "tiled_systolic_array",
            "tiled_systolic_transformer",
        ),
        score_bonus=10.0,
        rationale=(
            "Uso una mesh systolic densa come baseline per prodotti matrice-matrice."
        ),
        report_summary=(
            "Profilo dense_gemm: mesh systolic densa come baseline per matmul."
        ),
    ),
}

WORKLOAD_DETECTION_ORDER = (
    "transformer",
    "convolution",
    "sparse_linear_algebra",
    "dense_gemm",
)

DATAFLOW_TO_FAMILY = {
    "systolic": "tiled_systolic_array",
    "weight_stationary": "weight_stationary_array",
    "output_stationary": "output_stationary_array",
    "sparse": "sparse_pe_mesh",
}

FAMILY_TO_DATAFLOW = {
    "tiled_systolic_transformer": "systolic",
    "tiled_systolic_array": "systolic",
    "weight_stationary_array": "weight_stationary",
    "output_stationary_array": "output_stationary",
    "sparse_pe_mesh": "sparse",
}


def get_workload_profile(workload_type: str) -> WorkloadProfile:
    return WORKLOAD_PROFILES.get(workload_type, WORKLOAD_PROFILES["dense_gemm"])


def resolve_family_for_spec(
    workload_type: str,
    preferred_dataflow: str = "auto",
) -> str:
    if preferred_dataflow == "systolic":
        if workload_type == "transformer":
            return "tiled_systolic_transformer"
        return "tiled_systolic_array"
    if preferred_dataflow in DATAFLOW_TO_FAMILY:
        return DATAFLOW_TO_FAMILY[preferred_dataflow]
    return get_workload_profile(workload_type).family


def compatible_families_for_spec(
    workload_type: str,
    preferred_dataflow: str = "auto",
) -> Tuple[str, ...]:
    if preferred_dataflow != "auto":
        return (resolve_family_for_spec(workload_type, preferred_dataflow),)
    return get_workload_profile(workload_type).compatible_families


def resolve_dataflow_for_family(family: str) -> str:
    return FAMILY_TO_DATAFLOW.get(family, "systolic")


def resolve_dataflow_for_spec(
    workload_type: str,
    preferred_dataflow: str = "auto",
) -> str:
    if preferred_dataflow != "auto":
        return preferred_dataflow
    return resolve_dataflow_for_family(resolve_family_for_spec(workload_type, preferred_dataflow))
