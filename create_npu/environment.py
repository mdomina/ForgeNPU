import importlib.util
import os
import shutil
from typing import Any, Dict, List, Optional


EDA_TOOLCHAIN = [
    {
        "name": "iverilog",
        "kind": "simulation",
        "install_hint": "brew install icarus-verilog",
    },
    {
        "name": "verilator",
        "kind": "lint",
        "install_hint": "brew install verilator",
    },
    {
        "name": "yosys",
        "kind": "synthesis",
        "install_hint": "brew install yosys",
    },
]


def probe_toolchain() -> List[Dict[str, Any]]:
    report = []
    for tool in EDA_TOOLCHAIN:
        resolved_path = shutil.which(tool["name"])
        report.append(
            {
                "name": tool["name"],
                "kind": tool["kind"],
                "available": bool(resolved_path),
                "path": resolved_path,
                "install_hint": tool["install_hint"],
            }
        )
    return report


def probe_llm_backend(requested_backend: str, llm_model: Optional[str]) -> Dict[str, Any]:
    if requested_backend != "llm":
        return {
            "requested_backend": requested_backend,
            "effective_backend": "heuristic",
            "available": True,
            "reason": "Backend euristico locale attivo.",
            "model": None,
            "live_generation_enabled": False,
        }

    openai_spec = importlib.util.find_spec("openai")
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    base_url = os.getenv("OPENAI_BASE_URL")
    live_generation_enabled = os.getenv("CREATE_NPU_ENABLE_LIVE_LLM") == "1"

    if openai_spec is None:
        return {
            "requested_backend": "llm",
            "effective_backend": "heuristic",
            "available": False,
            "reason": "Package `openai` non installato; fallback a generatore euristico.",
            "model": llm_model,
            "live_generation_enabled": live_generation_enabled,
        }

    if not api_key_present:
        return {
            "requested_backend": "llm",
            "effective_backend": "heuristic",
            "available": False,
            "reason": "Variabile `OPENAI_API_KEY` assente; fallback a generatore euristico.",
            "model": llm_model,
            "live_generation_enabled": live_generation_enabled,
        }

    if not live_generation_enabled:
        return {
            "requested_backend": "llm",
            "effective_backend": "heuristic",
            "available": True,
            "reason": (
                "Configurazione LLM rilevata ma chiamate live disabilitate. "
                "Imposta `CREATE_NPU_ENABLE_LIVE_LLM=1` per attivarle."
            ),
            "model": llm_model,
            "base_url": base_url,
            "live_generation_enabled": False,
        }

    return {
        "requested_backend": "llm",
        "effective_backend": "llm",
        "available": True,
        "reason": "Backend LLM pronto per chiamate live.",
        "model": llm_model,
        "base_url": base_url,
        "live_generation_enabled": True,
    }


def collect_environment_snapshot(
    requested_backend: str = "heuristic", llm_model: Optional[str] = None
) -> Dict[str, Any]:
    return {
        "toolchain": probe_toolchain(),
        "llm": probe_llm_backend(requested_backend=requested_backend, llm_model=llm_model),
    }
