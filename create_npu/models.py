from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RequirementSpec:
    original_text: str
    numeric_precision: str = "INT8"
    throughput_value: Optional[float] = None
    throughput_unit: Optional[str] = None
    power_budget_watts: Optional[float] = None
    latency_budget_ms: Optional[float] = None
    area_budget_mm2: Optional[float] = None
    workload_type: str = "dense_gemm"
    batch_min: int = 1
    batch_max: int = 1
    interfaces: List[str] = field(default_factory=list)
    target_technology: Optional[str] = None
    target_frequency_mhz: Optional[float] = None
    assumptions: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArchitectureCandidate:
    candidate_id: str
    family: str
    tile_rows: int
    tile_cols: int
    tile_count: int
    pe_rows: int
    pe_cols: int
    pe_count: int
    local_sram_kb_per_tile: int
    global_buffer_mb: int
    bus_width_bits: int
    target_frequency_mhz: float
    estimated_tops: float
    estimated_power_watts: float
    estimated_area_mm2: float
    modules: List[str] = field(default_factory=list)
    rationale: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeneratedDesignBundle:
    rtl_files: List[str]
    testbench_files: List[str]
    primary_module: str
    supporting_files: List[str] = field(default_factory=list)
    reference_cases_path: Optional[str] = None
    candidate_id: str = "baseline"
    generator_backend: str = "heuristic"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolResult:
    name: str
    available: bool
    passed: Optional[bool]
    return_code: Optional[int]
    summary: str
    log_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    spec: RequirementSpec
    architecture: ArchitectureCandidate
    generated: GeneratedDesignBundle
    tool_results: List[ToolResult]
    score: float
    output_dir: str
    report: Dict[str, Any] = field(default_factory=dict)
    candidate_results: List[Dict[str, Any]] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "architecture": self.architecture.to_dict(),
            "generated": self.generated.to_dict(),
            "tool_results": [result.to_dict() for result in self.tool_results],
            "score": self.score,
            "output_dir": self.output_dir,
            "report": self.report,
            "candidate_results": self.candidate_results,
            "environment": self.environment,
        }
