import json
from pathlib import Path
from typing import Dict, List, Optional

from create_npu.environment import probe_llm_backend
from create_npu.models import ArchitectureCandidate, RequirementSpec


def prepare_backend_context(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
    output_dir: Path,
    requested_backend: str,
    llm_model: Optional[str] = None,
) -> Dict[str, object]:
    if requested_backend != "llm":
        return {
            "effective_backend": "heuristic",
            "notes": [],
            "supporting_files": [],
        }

    llm_status = probe_llm_backend(requested_backend="llm", llm_model=llm_model)
    prompt_payload = {
        "candidate_id": architecture.candidate_id,
        "model": llm_model or "unset",
        "spec": spec.to_dict(),
        "architecture": architecture.to_dict(),
        "prompt": _build_llm_prompt(spec=spec, architecture=architecture),
        "status": llm_status,
    }
    request_path = output_dir / "llm_request.json"
    request_path.write_text(
        json.dumps(prompt_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    notes: List[str] = [
        "Richiesta backend LLM registrata in `llm_request.json` per futura generazione guidata."
    ]
    if llm_status["effective_backend"] != "llm":
        notes.append(str(llm_status["reason"]))

    return {
        "effective_backend": str(llm_status["effective_backend"]),
        "notes": notes,
        "supporting_files": [str(request_path)],
    }


def _build_llm_prompt(spec: RequirementSpec, architecture: ArchitectureCandidate) -> str:
    return f"""Sei un generatore RTL per acceleratori NPU.

Requirement utente:
{spec.original_text}

Spec strutturata:
- precisione: {spec.numeric_precision}
- throughput: {spec.throughput_value} {spec.throughput_unit}
- workload: {spec.workload_type}
- batch: {spec.batch_min}-{spec.batch_max}
- interfacce: {", ".join(spec.interfaces)}

Architettura candidata:
- candidate_id: {architecture.candidate_id}
- family: {architecture.family}
- pe mesh: {architecture.pe_rows}x{architecture.pe_cols}
- tile: {architecture.tile_rows}x{architecture.tile_cols}
- sram per tile: {architecture.local_sram_kb_per_tile} KB
- bus width: {architecture.bus_width_bits} bit
- target frequency: {architecture.target_frequency_mhz} MHz

Richiesta:
1. proponi varianti di `processing_element` e `systolic_tile`
2. evidenzia rischi di verifica
3. restituisci output strutturato e conciso
"""
