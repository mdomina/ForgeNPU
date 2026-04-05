import re
from typing import List, Optional, Tuple

from create_npu.models import RequirementSpec
from create_npu.workloads import WORKLOAD_DETECTION_ORDER, get_workload_profile


NUMBER_PATTERN = r"(\d+(?:[.,]\d+)?)"


class RequirementParser:
    """Heuristic parser for the MVP bootstrap."""

    def parse(self, text: str) -> RequirementSpec:
        spec = RequirementSpec(original_text=text.strip())
        lower_text = text.lower()

        precision = self._extract_precision(lower_text)
        if precision:
            spec.numeric_precision = precision
        else:
            spec.assumptions.append("Precisione non specificata: assumo INT8 per l'MVP.")

        throughput = self._extract_throughput(lower_text)
        if throughput:
            spec.throughput_value, spec.throughput_unit = throughput
        else:
            spec.ambiguities.append("Target di throughput non trovato nel requisito.")

        spec.power_budget_watts = self._extract_number(lower_text, NUMBER_PATTERN + r"\s*w\b")
        spec.latency_budget_ms = self._extract_number(
            lower_text, NUMBER_PATTERN + r"\s*ms\b"
        )
        spec.area_budget_mm2 = self._extract_number(
            lower_text, NUMBER_PATTERN + r"\s*mm2\b"
        )
        spec.available_memory_mb = self._extract_memory_mb(lower_text)
        spec.memory_bandwidth_gb_per_s = self._extract_memory_bandwidth_gb_per_s(lower_text)

        workload = self._extract_workload(lower_text)
        if workload:
            spec.workload_type = workload
        else:
            spec.assumptions.append(
                "Workload non esplicitato: assumo dense_gemm come default iniziale."
            )

        execution_mode = self._extract_execution_mode(lower_text)
        if execution_mode:
            spec.execution_mode = execution_mode
        else:
            spec.assumptions.append(
                "Modalita' di esecuzione non specificata: assumo inference."
            )

        optimization_priority = self._extract_optimization_priority(lower_text)
        if optimization_priority:
            spec.optimization_priority = optimization_priority
        else:
            spec.assumptions.append(
                "Priorita' di ottimizzazione non esplicitata: assumo balanced."
            )

        offchip_memory_type = self._extract_offchip_memory_type(lower_text)
        if offchip_memory_type:
            spec.offchip_memory_type = offchip_memory_type
        else:
            spec.assumptions.append(
                "Memoria off-chip non specificata: assumo generic_dram."
            )

        preferred_dataflow = self._extract_preferred_dataflow(lower_text)
        if preferred_dataflow:
            spec.preferred_dataflow = preferred_dataflow
        else:
            spec.assumptions.append(
                "Dataflow non specificato: uso selezione automatica workload-aware."
            )

        spec.sparsity_support = self._extract_sparsity_support(
            lower_text=lower_text,
            workload_type=spec.workload_type,
        )
        spec.sequence_length = self._extract_sequence_length(lower_text)
        spec.kernel_size = self._extract_kernel_size(lower_text)

        batch_min, batch_max = self._extract_batch(lower_text)
        spec.batch_min = batch_min
        spec.batch_max = batch_max
        if "batch" not in lower_text:
            spec.assumptions.append("Batch non specificato: assumo batch 1.")

        interfaces = self._extract_interfaces(lower_text)
        if interfaces:
            spec.interfaces = interfaces
        else:
            spec.interfaces = ["AXI4", "DMA", "scratchpad_sram"]
            spec.assumptions.append(
                "Interfacce non dichiarate: assumo AXI4, DMA e scratchpad SRAM."
            )

        spec.target_technology = self._extract_technology(lower_text)
        if not spec.target_technology:
            spec.assumptions.append(
                "Nodo tecnologico non specificato: uso una baseline generica 5nm ASIC."
            )

        spec.target_frequency_mhz = self._extract_frequency_mhz(lower_text)
        if spec.target_frequency_mhz is None:
            spec.assumptions.append(
                "Frequenza target non specificata: assumo 1000 MHz per la stima iniziale."
            )

        if spec.power_budget_watts is None:
            spec.ambiguities.append("Budget di potenza assente o non parsabile.")
        if spec.target_technology is None:
            spec.ambiguities.append("Nodo tecnologico assente.")
        if spec.target_frequency_mhz is None:
            spec.ambiguities.append("Frequenza target assente.")

        if spec.throughput_unit == "TFLOPS" and spec.numeric_precision.startswith("INT"):
            spec.assumptions.append(
                "Interpreto TFLOPS su INT8 come throughput di operazioni equivalente a TOPS."
            )

        return spec

    def _extract_precision(self, lower_text: str) -> Optional[str]:
        for token in ("int8", "int16", "fp16", "bf16"):
            if token in lower_text:
                return token.upper()
        return None

    def _extract_execution_mode(self, lower_text: str) -> Optional[str]:
        if any(token in lower_text for token in ("training", "addestramento", "finetuning", "fine-tuning")):
            return "training"
        if any(token in lower_text for token in ("inference", "inferenza", "serving")):
            return "inference"
        return None

    def _extract_optimization_priority(self, lower_text: str) -> Optional[str]:
        priority_patterns = [
            ("latency", (r"\blatency\b", r"\bbassa latenza\b", r"\breal[- ]time\b", r"\btempo reale\b")),
            ("efficiency", (r"\befficien", r"\blow power\b", r"\bbasso consumo\b", r"\benergy\b")),
            ("area", (r"\barea[- ]first\b", r"\bcompact\b", r"\bsmall die\b", r"\barea minima\b")),
            ("throughput", (r"\bthroughput\b", r"\bmassim[oa] throughput\b", r"\bmax performance\b")),
            ("balanced", (r"\bbalanced\b", r"\bbilanciat", r"\bcompromesso\b")),
        ]
        for priority, patterns in priority_patterns:
            for pattern in patterns:
                if re.search(pattern, lower_text):
                    return priority
        return None

    def _extract_offchip_memory_type(self, lower_text: str) -> Optional[str]:
        if "hbm" in lower_text:
            return "HBM"
        if "lpddr" in lower_text:
            return "LPDDR"
        if "gddr" in lower_text:
            return "GDDR"
        if "ddr" in lower_text or "dram" in lower_text:
            return "DDR"
        if "host memory" in lower_text or "pcie host" in lower_text:
            return "host_memory"
        return None

    def _extract_preferred_dataflow(self, lower_text: str) -> Optional[str]:
        dataflow_patterns = [
            ("weight_stationary", (r"weight[- ]stationary", r"stationary dei pesi")),
            ("output_stationary", (r"output[- ]stationary", r"stationary delle uscite")),
            ("sparse", (r"sparse dataflow",)),
            ("systolic", (r"\bsystolic\b", r"array sistolic")),
        ]
        for dataflow, patterns in dataflow_patterns:
            for pattern in patterns:
                if re.search(pattern, lower_text):
                    return dataflow
        return None

    def _extract_sparsity_support(self, lower_text: str, workload_type: str) -> str:
        if re.search(r"\b2\s*:\s*4\b", lower_text) or "structured sparsity" in lower_text:
            return "structured"
        if "unstructured sparsity" in lower_text:
            return "unstructured"
        if "sparse" in lower_text or "sparsity" in lower_text or workload_type == "sparse_linear_algebra":
            return "unstructured"
        return "dense"

    def _extract_sequence_length(self, lower_text: str) -> Optional[int]:
        patterns = [
            r"(?:sequence length|seq(?:uence)?(?: len(?:gth)?)?|context(?: window)?)\s*(?:di|da|=)?\s*(\d+)",
            r"(\d+)\s*tokens?\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, lower_text)
            if match:
                return int(match.group(1))
        return None

    def _extract_kernel_size(self, lower_text: str) -> Optional[int]:
        patterns = [
            r"(?:kernel|filtro)\s*(\d+)\s*x\s*(\d+)",
            r"(\d+)\s*x\s*(\d+)\s*(?:kernel|filtro)",
        ]
        for pattern in patterns:
            match = re.search(pattern, lower_text)
            if match and match.group(1) == match.group(2):
                return int(match.group(1))
        single_match = re.search(r"(?:kernel|filtro)\s*(\d+)\b", lower_text)
        if single_match:
            return int(single_match.group(1))
        return None

    def _extract_throughput(self, lower_text: str) -> Optional[Tuple[float, str]]:
        match = re.search(NUMBER_PATTERN + r"\s*(tops|tflops)\b", lower_text)
        if not match:
            return None
        return self._to_float(match.group(1)), match.group(2).upper()

    def _extract_number(self, lower_text: str, pattern: str) -> Optional[float]:
        match = re.search(pattern, lower_text)
        if not match:
            return None
        return self._to_float(match.group(1))

    def _extract_workload(self, lower_text: str) -> Optional[str]:
        for workload in WORKLOAD_DETECTION_ORDER:
            profile = get_workload_profile(workload)
            for pattern in profile.detection_patterns:
                if re.search(pattern, lower_text):
                    return workload
        return None

    def _extract_batch(self, lower_text: str) -> Tuple[int, int]:
        range_match = re.search(r"batch\s*(\d+)\s*[-–]\s*(\d+)", lower_text)
        if range_match:
            return int(range_match.group(1)), int(range_match.group(2))

        single_match = re.search(r"batch\s*(\d+)", lower_text)
        if single_match:
            value = int(single_match.group(1))
            return value, value

        return 1, 1

    def _extract_interfaces(self, lower_text: str) -> List[str]:
        interfaces = []
        if "axi" in lower_text:
            interfaces.append("AXI4")
        if "dma" in lower_text:
            interfaces.append("DMA")
        if "sram" in lower_text:
            interfaces.append("SRAM")
        if "pcie" in lower_text:
            interfaces.append("PCIe")
        return interfaces

    def _extract_technology(self, lower_text: str) -> Optional[str]:
        nm_match = re.search(r"(\d+)\s*nm\b", lower_text)
        if nm_match:
            return nm_match.group(1) + "nm"
        if "fpga" in lower_text:
            return "fpga"
        return None

    def _extract_frequency_mhz(self, lower_text: str) -> Optional[float]:
        match = re.search(NUMBER_PATTERN + r"\s*(ghz|mhz)\b", lower_text)
        if not match:
            return None
        value = self._to_float(match.group(1))
        unit = match.group(2)
        if unit == "ghz":
            return value * 1000.0
        return value

    def _extract_memory_mb(self, lower_text: str) -> Optional[float]:
        patterns = [
            NUMBER_PATTERN
            + r"\s*(tb|gb|mb|kb)\b(?:\s+di)?\s*(?:memoria|sram|dram|hbm|cache|buffer)\b",
            r"(?:memoria|sram|dram|hbm|cache|buffer)\s*(?:da|di)?\s*"
            + NUMBER_PATTERN
            + r"\s*(tb|gb|mb|kb)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, lower_text)
            if match:
                return self._memory_unit_to_mb(
                    value=self._to_float(match.group(1)),
                    unit=match.group(2),
                )
        return None

    def _extract_memory_bandwidth_gb_per_s(self, lower_text: str) -> Optional[float]:
        match = re.search(
            NUMBER_PATTERN + r"\s*(tb/s|gb/s|mb/s|tbps|gbps|mbps)\b",
            lower_text,
        )
        if not match:
            return None

        value = self._to_float(match.group(1))
        unit = match.group(2)
        decimal_scale = {
            "tb/s": 1000.0,
            "gb/s": 1.0,
            "mb/s": 0.001,
        }
        if unit in decimal_scale:
            return round(value * decimal_scale[unit], 6)

        bit_rate_scale = {
            "tbps": 125.0,
            "gbps": 0.125,
            "mbps": 0.000125,
        }
        return round(value * bit_rate_scale[unit], 6)

    def _memory_unit_to_mb(self, value: float, unit: str) -> float:
        scale = {
            "tb": 1024.0 * 1024.0,
            "gb": 1024.0,
            "mb": 1.0,
            "kb": 1.0 / 1024.0,
        }
        return round(value * scale[unit], 6)

    def _to_float(self, token: str) -> float:
        return float(token.replace(",", "."))
