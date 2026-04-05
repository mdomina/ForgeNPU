import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from create_npu.environment import probe_llm_backend
from create_npu.models import ArchitectureCandidate, RequirementSpec
from create_npu.rtl_generator import (
    _processing_element_template,
    _resolve_width,
    _seed_tile_shape,
    _systolic_tile_template,
)

_ALLOWED_OVERRIDE_FILES = {
    "processing_element.sv": "processing_element",
    "systolic_tile.sv": "systolic_tile",
}


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
            "rtl_overrides": {},
            "compare_against_seed": False,
        }

    llm_status = probe_llm_backend(requested_backend="llm", llm_model=llm_model)
    prompt_payload = _build_llm_request_payload(
        spec=spec,
        architecture=architecture,
        llm_status=llm_status,
    )
    request_path = output_dir / "llm_request.json"
    request_path.write_text(
        json.dumps(prompt_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    notes: List[str] = [
        "Richiesta backend LLM registrata in `llm_request.json` con prompt e schema strutturati."
    ]
    supporting_files: List[str] = [str(request_path)]

    if llm_status["effective_backend"] != "llm":
        notes.append(str(llm_status["reason"]))
        return {
            "effective_backend": "heuristic",
            "notes": notes,
            "supporting_files": supporting_files,
            "rtl_overrides": {},
            "compare_against_seed": False,
        }

    try:
        parsed_payload, response_payload = _run_live_llm_generation(prompt_payload=prompt_payload)
    except Exception as exc:
        error_path = output_dir / "llm_error.json"
        error_payload = {
            "candidate_id": architecture.candidate_id,
            "model": llm_status.get("model"),
            "error": str(exc),
        }
        error_path.write_text(
            json.dumps(error_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        notes.append(
            "Chiamata live LLM fallita; fallback a generatore euristico. "
            f"Errore: {exc}"
        )
        supporting_files.append(str(error_path))
        return {
            "effective_backend": "heuristic",
            "notes": notes,
            "supporting_files": supporting_files,
            "rtl_overrides": {},
            "compare_against_seed": False,
        }

    response_path = output_dir / "llm_response.json"
    parsed_path = output_dir / "llm_structured_output.json"
    response_path.write_text(
        json.dumps(response_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parsed_path.write_text(
        json.dumps(parsed_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    supporting_files.extend([str(response_path), str(parsed_path)])

    rtl_overrides, validation_notes = _extract_rtl_overrides(parsed_payload)
    notes.extend(validation_notes)
    if not rtl_overrides:
        notes.append(
            "Output LLM valido ma senza override RTL applicabili; fallback a generatore euristico."
        )
        return {
            "effective_backend": "heuristic",
            "notes": notes,
            "supporting_files": supporting_files,
            "rtl_overrides": {},
            "compare_against_seed": False,
        }

    notes.append(
        "Backend LLM live attivo con override RTL strutturati su "
        + ", ".join(sorted(rtl_overrides))
        + "."
    )
    return {
        "effective_backend": "llm",
        "notes": notes,
        "supporting_files": supporting_files,
        "rtl_overrides": rtl_overrides,
        "compare_against_seed": True,
    }


def _build_llm_request_payload(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
    llm_status: Dict[str, Any],
) -> Dict[str, Any]:
    seed_modules = _build_seed_module_context(spec=spec, architecture=architecture)
    resolved_model = str(llm_status.get("model") or os.getenv("CREATE_NPU_LLM_MODEL") or "gpt-4o-mini")
    system_prompt, user_prompt = _build_llm_prompts(
        spec=spec,
        architecture=architecture,
        seed_modules=seed_modules,
    )
    return {
        "candidate_id": architecture.candidate_id,
        "model": resolved_model,
        "spec": spec.to_dict(),
        "architecture": architecture.to_dict(),
        "seed_modules": seed_modules,
        "response_schema": _llm_output_schema(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "status": llm_status,
    }


def _build_llm_prompts(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
    seed_modules: Dict[str, str],
) -> Tuple[str, str]:
    system_prompt = """Sei un generatore RTL per acceleratori NPU.
Devi restituire solo JSON conforme allo schema fornito.
Puoi proporre override soltanto per `processing_element.sv` e `systolic_tile.sv`.
Mantieni nomi dei moduli, parametri e porte compatibili con il seed corrente.
Non usare markdown, testo libero o campi extra."""
    user_prompt = f"""Requirement utente:
{spec.original_text}

Spec strutturata:
{json.dumps(spec.to_dict(), indent=2, sort_keys=True)}

Architettura candidata:
{json.dumps(architecture.to_dict(), indent=2, sort_keys=True)}

Vincoli di verifica:
- l'output deve compilare con i testbench seed esistenti;
- non cambiare le interfacce pubbliche dei moduli;
- conserva la semantica base MAC/accumulo del datapath;
- se non vedi un miglioramento sensato, restituisci `rtl_overrides: []`.

Moduli seed disponibili per la variazione:

### processing_element.sv
{seed_modules["processing_element.sv"]}

### systolic_tile.sv
{seed_modules["systolic_tile.sv"]}
"""
    return system_prompt, user_prompt


def _build_seed_module_context(
    spec: RequirementSpec,
    architecture: ArchitectureCandidate,
) -> Dict[str, str]:
    operand_width, _ = _resolve_width(spec.numeric_precision)
    acc_width = max(32, operand_width * 4)
    seed_rows, seed_cols = _seed_tile_shape(
        architecture.tile_rows,
        architecture.tile_cols,
    )
    return {
        "processing_element.sv": _processing_element_template(
            operand_width=operand_width,
            acc_width=acc_width,
        ),
        "systolic_tile.sv": _systolic_tile_template(
            operand_width=operand_width,
            acc_width=acc_width,
            seed_rows=seed_rows,
            seed_cols=seed_cols,
        ),
    }


def _llm_output_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "summary",
            "expected_benefits",
            "verification_risks",
            "rtl_overrides",
        ],
        "properties": {
            "summary": {
                "type": "string",
            },
            "expected_benefits": {
                "type": "array",
                "items": {"type": "string"},
            },
            "verification_risks": {
                "type": "array",
                "items": {"type": "string"},
            },
            "rtl_overrides": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "file_name",
                        "module_name",
                        "source",
                        "rationale",
                    ],
                    "properties": {
                        "file_name": {
                            "type": "string",
                            "enum": sorted(_ALLOWED_OVERRIDE_FILES),
                        },
                        "module_name": {
                            "type": "string",
                            "enum": sorted(_ALLOWED_OVERRIDE_FILES.values()),
                        },
                        "source": {
                            "type": "string",
                        },
                        "rationale": {
                            "type": "string",
                        },
                    },
                },
            },
        },
    }


def _run_live_llm_generation(prompt_payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    openai_module = __import__("openai")
    client_kwargs = {
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    timeout_seconds = float(os.getenv("CREATE_NPU_LLM_TIMEOUT_SECONDS", "60"))
    client_kwargs["timeout"] = timeout_seconds
    client = openai_module.OpenAI(**client_kwargs)
    response = client.responses.create(
        model=str(prompt_payload["model"]),
        input=[
            {"role": "system", "content": str(prompt_payload["system_prompt"])},
            {"role": "user", "content": str(prompt_payload["user_prompt"])},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "rtl_variant_bundle",
                "strict": True,
                "schema": prompt_payload["response_schema"],
            }
        },
    )
    response_payload = _serialize_sdk_response(response)
    output_text = _extract_response_text(response_payload, response)
    if not output_text:
        raise ValueError("Risposta LLM vuota o non serializzabile in testo JSON.")
    return json.loads(output_text), response_payload


def _serialize_sdk_response(response: Any) -> Dict[str, Any]:
    if hasattr(response, "model_dump"):
        payload = response.model_dump(mode="json")
        if isinstance(payload, dict):
            return payload
    if hasattr(response, "to_dict"):
        payload = response.to_dict()
        if isinstance(payload, dict):
            return payload
    if isinstance(response, dict):
        return response
    return {
        "output_text": getattr(response, "output_text", ""),
    }


def _extract_response_text(response_payload: Dict[str, Any], response: Any) -> str:
    output_text = response_payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    raw_output_text = getattr(response, "output_text", None)
    if isinstance(raw_output_text, str) and raw_output_text.strip():
        return raw_output_text.strip()
    return _find_text_value(response_payload.get("output", []))


def _find_text_value(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ("text", "output_text"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for value in payload.values():
            extracted = _find_text_value(value)
            if extracted:
                return extracted
        return ""
    if isinstance(payload, list):
        for item in payload:
            extracted = _find_text_value(item)
            if extracted:
                return extracted
    return ""


def _extract_rtl_overrides(parsed_payload: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    rtl_overrides: Dict[str, str] = {}
    notes: List[str] = []
    summary = parsed_payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        notes.append("Sintesi LLM: " + summary.strip())
    for item in parsed_payload.get("expected_benefits", []):
        if isinstance(item, str) and item.strip():
            notes.append("Beneficio atteso LLM: " + item.strip())
    for item in parsed_payload.get("verification_risks", []):
        if isinstance(item, str) and item.strip():
            notes.append("Rischio verifica LLM: " + item.strip())

    for override in parsed_payload.get("rtl_overrides", []):
        if not isinstance(override, dict):
            continue
        file_name = str(override.get("file_name", ""))
        module_name = str(override.get("module_name", ""))
        source = override.get("source")
        rationale = str(override.get("rationale", "")).strip()
        expected_module_name = _ALLOWED_OVERRIDE_FILES.get(file_name)
        if expected_module_name is None:
            notes.append(f"Override LLM ignorato: file non consentito `{file_name}`.")
            continue
        if module_name != expected_module_name:
            notes.append(
                "Override LLM ignorato: associazione file/modulo incoerente per "
                f"`{file_name}`."
            )
            continue
        if not isinstance(source, str) or not source.strip():
            notes.append(f"Override LLM ignorato: sorgente vuota per `{file_name}`.")
            continue
        if f"module {module_name}" not in source or "endmodule" not in source:
            notes.append(
                f"Override LLM ignorato: modulo `{module_name}` non riconoscibile in `{file_name}`."
            )
            continue
        rtl_overrides[file_name] = source
        if rationale:
            notes.append(f"Override LLM `{file_name}`: {rationale}")
    return rtl_overrides, notes
