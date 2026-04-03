import re
from typing import List, Optional, Tuple

from create_npu.models import RequirementSpec


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

        workload = self._extract_workload(lower_text)
        if workload:
            spec.workload_type = workload
        else:
            spec.assumptions.append(
                "Workload non esplicitato: assumo dense_gemm come default iniziale."
            )

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
        ordered_keywords = [
            ("transformer", "transformer"),
            ("gemm", "dense_gemm"),
            ("dense", "dense_gemm"),
            ("conv", "convolution"),
            ("cnn", "convolution"),
            ("sparse", "sparse_linear_algebra"),
        ]
        for keyword, workload in ordered_keywords:
            if keyword in lower_text:
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

    def _to_float(self, token: str) -> float:
        return float(token.replace(",", "."))

