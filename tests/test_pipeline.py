import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from create_npu.architect import plan_architecture
from create_npu.benchmark import run_regression_benchmark
from create_npu.harness import VerificationHarness
from create_npu.models import (
    ArchitectureCandidate,
    GeneratedDesignBundle,
    PipelineResult,
    RequirementSpec,
    ToolResult,
)
from create_npu.pipeline import CreateNPUPipeline
from create_npu.requirement_parser import RequirementParser


class RequirementParserTest(unittest.TestCase):
    def test_parse_transformer_requirement(self) -> None:
        parser = RequirementParser()
        spec = parser.parse(
            "Voglio una NPU INT8 da 1000 TFLOPS per transformer con 250 W e batch 1-16."
        )

        self.assertEqual(spec.numeric_precision, "INT8")
        self.assertEqual(spec.throughput_value, 1000.0)
        self.assertEqual(spec.throughput_unit, "TFLOPS")
        self.assertEqual(spec.power_budget_watts, 250.0)
        self.assertEqual(spec.batch_min, 1)
        self.assertEqual(spec.batch_max, 16)
        self.assertEqual(spec.workload_type, "transformer")

    def test_parse_memory_and_bandwidth_requirement(self) -> None:
        parser = RequirementParser()
        spec = parser.parse(
            "Voglio una NPU INT8 da 2 TOPS per transformer con 4 MB di SRAM, "
            "500 GB/s e 1 GHz."
        )

        self.assertEqual(spec.available_memory_mb, 4.0)
        self.assertEqual(spec.memory_bandwidth_gb_per_s, 500.0)
        self.assertEqual(spec.target_frequency_mhz, 1000.0)


class ArchitecturePlanningTest(unittest.TestCase):
    def test_plan_architecture_respects_memory_and_bandwidth_hints(self) -> None:
        architecture = plan_architecture(
            RequirementSpec(
                original_text="NPU INT8 2 TOPS transformer con 4 MB e 500 GB/s.",
                numeric_precision="INT8",
                throughput_value=2.0,
                throughput_unit="TOPS",
                available_memory_mb=4.0,
                memory_bandwidth_gb_per_s=500.0,
                workload_type="transformer",
                batch_min=1,
                batch_max=4,
                target_frequency_mhz=1000.0,
            )
        )

        total_local_memory_mb = architecture.global_buffer_mb + (
            architecture.local_sram_kb_per_tile * architecture.tile_count
        ) / 1024.0
        sustained_bandwidth_gb_per_s = (
            architecture.bus_width_bits * architecture.target_frequency_mhz
        ) / 8000.0

        self.assertLessEqual(total_local_memory_mb, 4.0)
        self.assertGreaterEqual(sustained_bandwidth_gb_per_s, 500.0)


class PipelineTest(unittest.TestCase):
    def test_pipeline_generates_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CreateNPUPipeline(base_output_dir=Path(temp_dir))
            result = pipeline.run(
                "Voglio una NPU INT8 da 50 TOPS con supporto transformer e batch 1-4.",
                num_candidates=3,
            )

            result_path = Path(result.output_dir) / "result.json"
            self.assertTrue(result_path.exists())
            self.assertTrue((Path(result.output_dir) / "candidates.json").exists())
            self.assertTrue(
                (Path(result.output_dir) / "candidates" / "balanced" / "rtl" / "mac_unit.sv").exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "rtl"
                    / "processing_element.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "verification_vectors.json"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "rtl"
                    / "systolic_tile.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "tb"
                    / "systolic_tile_tb.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "rtl"
                    / "scratchpad_controller.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "rtl"
                    / "dma_engine.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "rtl"
                    / "tile_compute_unit.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "rtl"
                    / "scheduler.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "rtl"
                    / "top_npu.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "tb"
                    / "scratchpad_controller_tb.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "tb"
                    / "dma_engine_tb.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "tb"
                    / "tile_compute_unit_tb.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "tb"
                    / "scheduler_tb.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "tb"
                    / "top_npu_tb.sv"
                ).exists()
            )

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["generated"]["primary_module"], "top_npu")
            self.assertEqual(len(payload["candidate_results"]), 3)
            self.assertIn("environment", payload)
            self.assertIn("report", payload)
            python_reference = next(
                result for result in payload["tool_results"] if result["name"] == "python_reference"
            )
            self.assertTrue(python_reference["available"])
            self.assertTrue(python_reference["passed"])
            self.assertIn("casi", python_reference["summary"])
            self.assertGreaterEqual(payload["score"], 0.0)

            report = payload["report"]
            self.assertTrue(Path(report["path"]).exists())
            self.assertEqual(report["summary"]["top_level_case_count"], 3)
            self.assertEqual(
                report["summary"]["scheduler_state_sequence"],
                [
                    "DMA_ACT",
                    "DMA_WGT",
                    "DMA_ACT",
                    "DMA_WGT",
                    "LOAD",
                    "LOAD",
                    "COMPUTE",
                    "COMPUTE",
                    "CLEAR",
                    "DONE",
                    "IDLE",
                    "DMA_ACT",
                    "DMA_WGT",
                    "LOAD",
                    "LOAD",
                    "COMPUTE",
                    "DONE",
                    "IDLE",
                    "DMA_ACT",
                    "DMA_WGT",
                    "LOAD",
                    "LOAD",
                    "COMPUTE",
                    "DONE",
                    "IDLE",
                ],
            )
            self.assertEqual(report["summary"]["total_cycles"], 25)
            self.assertEqual(report["summary"]["busy_cycles"], 19)
            self.assertEqual(report["summary"]["done_cycles"], 3)
            self.assertEqual(report["summary"]["idle_cycles"], 3)
            self.assertEqual(report["summary"]["memory_path"]["dma_cycles"], 8)
            self.assertEqual(report["summary"]["memory_path"]["dma_activation_cycles"], 4)
            self.assertEqual(report["summary"]["memory_path"]["dma_weight_cycles"], 4)
            self.assertEqual(report["summary"]["memory_path"]["load_cycles"], 6)
            self.assertEqual(report["summary"]["memory_path"]["max_scratchpad_depth"], 4)
            self.assertEqual(report["summary"]["memory_path"]["activation_slots_touched"], 2)
            self.assertEqual(report["summary"]["memory_path"]["weight_slots_touched"], 2)
            self.assertEqual(report["summary"]["memory_path"]["peak_activation_slots_live"], 2)
            self.assertEqual(report["summary"]["memory_path"]["peak_weight_slots_live"], 2)
            self.assertEqual(report["summary"]["memory_path"]["working_set_utilization"], 0.5)
            self.assertEqual(report["summary"]["memory_path"]["total_dma_bits_transferred"], 128)
            self.assertEqual(
                report["summary"]["memory_path"]["average_dma_bits_per_dma_cycle"],
                16.0,
            )
            self.assertEqual(report["summary"]["memory_path"]["peak_dma_bits_per_cycle"], 16)
            self.assertEqual(
                report["summary"]["memory_path"]["effective_external_bandwidth_gb_per_s"],
                0.64,
            )
            self.assertEqual(
                report["summary"]["memory_path"]["peak_external_bandwidth_gb_per_s"],
                2.0,
            )
            self.assertEqual(
                report["summary"]["memory_path"]["theoretical_bus_bandwidth_gb_per_s"],
                128.0,
            )
            self.assertEqual(
                report["summary"]["memory_path"]["bus_bandwidth_utilization"],
                0.005,
            )
            self.assertEqual(
                report["summary"]["memory_path"]["peak_bus_bandwidth_utilization"],
                0.015625,
            )
            self.assertEqual(report["summary"]["compute_path"]["compute_cycles"], 4)
            self.assertEqual(report["summary"]["compute_path"]["clear_cycles"], 1)
            self.assertEqual(report["summary"]["compute_path"]["estimated_mac_operations"], 20)
            self.assertTrue(report["summary"]["top_npu_throughput"]["available"])
            self.assertEqual(
                report["summary"]["top_npu_throughput"]["estimation_model"],
                "peak_scaled_by_compute_duty_cycle",
            )
            self.assertEqual(report["summary"]["top_npu_throughput"]["total_cycles"], 25)
            self.assertEqual(report["summary"]["top_npu_throughput"]["compute_cycles"], 4)
            self.assertEqual(
                report["summary"]["top_npu_throughput"]["scheduler_overhead_cycles"],
                21,
            )
            self.assertEqual(
                report["summary"]["top_npu_throughput"]["architecture_pe_count"],
                25600,
            )
            self.assertEqual(
                report["summary"]["top_npu_throughput"]["theoretical_peak_tops"],
                51.2,
            )
            self.assertEqual(
                report["summary"]["top_npu_throughput"]["compute_duty_cycle"],
                0.16,
            )
            self.assertEqual(
                report["summary"]["top_npu_throughput"]["compute_duty_cycle_while_busy"],
                0.210526,
            )
            self.assertEqual(
                report["summary"]["top_npu_throughput"]["estimated_effective_ops_per_cycle"],
                8192.0,
            )
            self.assertEqual(
                report["summary"]["top_npu_throughput"]["estimated_effective_tops"],
                8.192,
            )

            report_payload = json.loads(Path(report["path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(report_payload["cases"]), 3)
            self.assertEqual(report_payload["cases"][1]["name"], "single_slot_single_compute_top")
            self.assertEqual(report_payload["cases"][2]["name"], "dual_tile_broadcast_compute_top")
            self.assertEqual(report_payload["cases"][1]["program"]["slot_count_i"], 1)
            self.assertEqual(report_payload["cases"][1]["program"]["load_iterations_i"], 2)
            self.assertEqual(report_payload["cases"][1]["program"]["compute_iterations_i"], 1)
            self.assertEqual(report_payload["cases"][1]["program"]["clear_on_done_i"], 0)
            self.assertEqual(report_payload["cases"][2]["tile_count"], 2)
            self.assertEqual(report_payload["cases"][2]["program"]["tile_enable_i"], [1, 1])
            self.assertEqual(
                report_payload["cases"][2]["top_npu_throughput"]["seed_peak_macs_per_cycle"],
                8,
            )
            self.assertEqual(
                report_payload["cases"][2]["summary"]["compute_path"]["estimated_mac_operations"],
                8,
            )
            self.assertEqual(
                report_payload["cases"][0]["summary"]["memory_path"]["working_set_utilization"],
                0.5,
            )
            self.assertEqual(
                report_payload["cases"][1]["summary"]["memory_path"]["working_set_utilization"],
                0.25,
            )
            self.assertEqual(
                report_payload["cases"][0]["top_npu_throughput"]["estimated_effective_tops"],
                9.309091,
            )
            self.assertEqual(
                report_payload["cases"][1]["top_npu_throughput"]["estimated_effective_tops"],
                7.314286,
            )
            self.assertEqual(
                report_payload["cases"][2]["top_npu_throughput"]["estimated_effective_tops"],
                7.314286,
            )

            balanced_candidate = next(
                candidate
                for candidate in payload["candidate_results"]
                if candidate["candidate_id"] == "balanced"
            )
            self.assertIn("report", balanced_candidate)
            self.assertTrue(Path(balanced_candidate["report"]["path"]).exists())
            self.assertIn(
                "execution_report.json",
                " ".join(balanced_candidate["generated"]["supporting_files"]),
            )

    def test_llm_backend_falls_back_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CreateNPUPipeline(base_output_dir=Path(temp_dir))
            result = pipeline.run(
                "Voglio una NPU INT8 da 10 TOPS per dense GEMM.",
                num_candidates=1,
                generator_backend="llm",
                llm_model="gpt-test",
            )

            self.assertEqual(result.generated.generator_backend, "heuristic")
            self.assertEqual(result.environment["llm"]["requested_backend"], "llm")
            self.assertIn("llm_request.json", " ".join(result.generated.supporting_files))


class HarnessTest(unittest.TestCase):
    def test_verilator_runs_each_testbench_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "candidate"
            output_dir.mkdir(parents=True, exist_ok=True)
            harness = VerificationHarness(output_dir)
            bundle = GeneratedDesignBundle(
                rtl_files=["rtl/mac_unit.sv", "rtl/top_npu.sv"],
                testbench_files=["tb/mac_unit_tb.sv", "tb/top_npu_tb.sv"],
                primary_module="top_npu",
            )
            commands = []

            def fake_run(command, cwd, capture_output, text, check):
                commands.append((command, cwd))
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with patch("create_npu.harness.shutil.which", return_value="/usr/bin/verilator"):
                with patch("create_npu.harness.subprocess.run", side_effect=fake_run):
                    result = harness._run_verilator_lint(bundle)

            self.assertTrue(result.available)
            self.assertTrue(result.passed)
            self.assertEqual(result.summary, "Verilator valido su 2 testbench.")
            self.assertEqual(len(commands), 2)

            lint_mac = commands[0][0]
            lint_top = commands[1][0]
            self.assertIn("--timing", lint_mac)
            self.assertIn("--top-module", lint_mac)
            self.assertIn("mac_unit_tb", lint_mac)
            self.assertIn(str(Path("tb/mac_unit_tb.sv").resolve()), lint_mac)
            self.assertNotIn(str(Path("tb/top_npu_tb.sv").resolve()), lint_mac)
            self.assertIn("top_npu_tb", lint_top)
            self.assertIn(str(Path("tb/top_npu_tb.sv").resolve()), lint_top)
            self.assertNotIn(str(Path("tb/mac_unit_tb.sv").resolve()), lint_top)

            aggregate_log = output_dir / "logs" / "verilator_lint.log"
            self.assertTrue(aggregate_log.exists())
            aggregate_payload = aggregate_log.read_text(encoding="utf-8")
            self.assertIn("[mac_unit_tb]", aggregate_payload)
            self.assertIn("[top_npu_tb]", aggregate_payload)

    def test_verilator_reports_failing_testbench(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "candidate"
            output_dir.mkdir(parents=True, exist_ok=True)
            harness = VerificationHarness(output_dir)
            bundle = GeneratedDesignBundle(
                rtl_files=["rtl/mac_unit.sv", "rtl/top_npu.sv"],
                testbench_files=["tb/mac_unit_tb.sv", "tb/top_npu_tb.sv"],
                primary_module="top_npu",
            )

            def fake_run(command, cwd, capture_output, text, check):
                if "--top-module" in command and command[command.index("--top-module") + 1] == "top_npu_tb":
                    return subprocess.CompletedProcess(command, 1, stdout="", stderr="lint error\n")
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with patch("create_npu.harness.shutil.which", return_value="/usr/bin/verilator"):
                with patch("create_npu.harness.subprocess.run", side_effect=fake_run):
                    result = harness._run_verilator_lint(bundle)

            self.assertTrue(result.available)
            self.assertFalse(result.passed)
            self.assertEqual(result.return_code, 1)
            self.assertIn("top_npu_tb", result.summary)
            self.assertTrue(Path(result.log_path).exists())

    def test_iverilog_runs_each_testbench_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "candidate"
            output_dir.mkdir(parents=True, exist_ok=True)
            harness = VerificationHarness(output_dir)
            bundle = GeneratedDesignBundle(
                rtl_files=["rtl/mac_unit.sv", "rtl/top_npu.sv"],
                testbench_files=["tb/mac_unit_tb.sv", "tb/top_npu_tb.sv"],
                primary_module="top_npu",
            )
            commands = []

            def fake_run(command, cwd, capture_output, text, check):
                commands.append((command, cwd))
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with patch("create_npu.harness.shutil.which", return_value="/usr/bin/iverilog"):
                with patch("create_npu.harness.subprocess.run", side_effect=fake_run):
                    result = harness._run_iverilog_sim(bundle)

            self.assertTrue(result.available)
            self.assertTrue(result.passed)
            self.assertEqual(result.summary, "Icarus valido su 2 testbench.")
            self.assertEqual(len(commands), 4)

            compile_mac = commands[0][0]
            compile_top = commands[2][0]
            self.assertIn("-s", compile_mac)
            self.assertIn("mac_unit_tb", compile_mac)
            self.assertIn(str(Path("tb/mac_unit_tb.sv").resolve()), compile_mac)
            self.assertNotIn(str(Path("tb/top_npu_tb.sv").resolve()), compile_mac)
            self.assertIn("top_npu_tb", compile_top)
            self.assertIn(str(Path("tb/top_npu_tb.sv").resolve()), compile_top)
            self.assertNotIn(str(Path("tb/mac_unit_tb.sv").resolve()), compile_top)
            self.assertTrue(str(commands[1][0][0]).endswith("mac_unit_tb.out"))
            self.assertTrue(str(commands[3][0][0]).endswith("top_npu_tb.out"))

            aggregate_log = output_dir / "logs" / "iverilog_sim.log"
            self.assertTrue(aggregate_log.exists())
            aggregate_payload = aggregate_log.read_text(encoding="utf-8")
            self.assertIn("[mac_unit_tb]", aggregate_payload)
            self.assertIn("[top_npu_tb]", aggregate_payload)

    def test_iverilog_reports_failing_testbench(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "candidate"
            output_dir.mkdir(parents=True, exist_ok=True)
            harness = VerificationHarness(output_dir)
            bundle = GeneratedDesignBundle(
                rtl_files=["rtl/mac_unit.sv", "rtl/top_npu.sv"],
                testbench_files=["tb/mac_unit_tb.sv", "tb/top_npu_tb.sv"],
                primary_module="top_npu",
            )

            def fake_run(command, cwd, capture_output, text, check):
                if "-s" in command and "top_npu_tb" in command:
                    return subprocess.CompletedProcess(command, 1, stdout="", stderr="compile error\n")
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with patch("create_npu.harness.shutil.which", return_value="/usr/bin/iverilog"):
                with patch("create_npu.harness.subprocess.run", side_effect=fake_run):
                    result = harness._run_iverilog_sim(bundle)

            self.assertTrue(result.available)
            self.assertFalse(result.passed)
            self.assertEqual(result.return_code, 1)
            self.assertIn("top_npu_tb", result.summary)
            self.assertTrue(Path(result.log_path).exists())


class BenchmarkTest(unittest.TestCase):
    def test_regression_benchmark_passes_with_strict_toolchain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            def fake_run(*, requirement_text, output_dir, num_candidates, generator_backend, llm_model):
                case_id = Path(output_dir).name
                output_dir.mkdir(parents=True, exist_ok=True)
                if case_id == "llm_fallback_capture":
                    llm_request = output_dir / "llm_request.json"
                    llm_request.write_text("{}", encoding="utf-8")
                    return _make_fake_pipeline_result(
                        output_dir=output_dir,
                        requirement_text=requirement_text,
                        candidate_id="balanced",
                        generator_backend="heuristic",
                        requested_backend="llm",
                        supporting_files=[str(llm_request)],
                        estimated_effective_tops=1.6384,
                    )
                return _make_fake_pipeline_result(
                    output_dir=output_dir,
                    requirement_text=requirement_text,
                    candidate_id="balanced",
                    generator_backend="heuristic",
                    requested_backend="heuristic",
                    supporting_files=[],
                    estimated_effective_tops=8.192,
                )

            with patch("create_npu.benchmark.CreateNPUPipeline") as pipeline_cls:
                pipeline_cls.return_value.run.side_effect = fake_run
                payload = run_regression_benchmark(
                    output_dir=temp_path / "benchmark",
                    require_full_toolchain=True,
                    llm_model="gpt-test",
                )

            self.assertTrue(payload["passed"])
            self.assertEqual(len(payload["cases"]), 2)
            self.assertTrue(Path(payload["summary_path"]).exists())

    def test_regression_benchmark_reports_missing_toolchain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            def fake_run(*, requirement_text, output_dir, num_candidates, generator_backend, llm_model):
                output_dir.mkdir(parents=True, exist_ok=True)
                return _make_fake_pipeline_result(
                    output_dir=output_dir,
                    requirement_text=requirement_text,
                    candidate_id="balanced",
                    generator_backend="heuristic",
                    requested_backend="heuristic" if generator_backend == "heuristic" else "llm",
                    supporting_files=[],
                    estimated_effective_tops=8.192 if generator_backend == "heuristic" else 1.6384,
                    missing_tool_name="iverilog_sim",
                )

            with patch("create_npu.benchmark.CreateNPUPipeline") as pipeline_cls:
                pipeline_cls.return_value.run.side_effect = fake_run
                payload = run_regression_benchmark(
                    output_dir=temp_path / "benchmark",
                    require_full_toolchain=True,
                    llm_model="gpt-test",
                )

            self.assertFalse(payload["passed"])
            toolchain_case = next(case for case in payload["cases"] if case["case_id"] == "toolchain_transformer_smoke")
            self.assertIn("Tool non disponibile: iverilog_sim.", toolchain_case["failures"])

def _make_fake_pipeline_result(
    output_dir: Path,
    requirement_text: str,
    candidate_id: str,
    generator_backend: str,
    requested_backend: str,
    supporting_files: list,
    estimated_effective_tops: float,
    missing_tool_name: str = "",
) -> PipelineResult:
    architecture = ArchitectureCandidate(
        candidate_id=candidate_id,
        family="tiled_systolic_transformer",
        tile_rows=32,
        tile_cols=32,
        tile_count=25,
        pe_rows=160,
        pe_cols=160,
        pe_count=25600,
        local_sram_kb_per_tile=512,
        global_buffer_mb=6,
        bus_width_bits=1024,
        target_frequency_mhz=1000.0,
        estimated_tops=51.2 if requested_backend == "heuristic" else 10.24,
        estimated_power_watts=25.07,
        estimated_area_mm2=4.72,
    )
    generated = GeneratedDesignBundle(
        rtl_files=[],
        testbench_files=[],
        primary_module="top_npu",
        candidate_id=candidate_id,
        generator_backend=generator_backend,
        supporting_files=supporting_files,
    )
    tool_results = []
    for tool_name in ("python_reference", "verilator_lint", "iverilog_sim", "yosys_synth"):
        available = tool_name != missing_tool_name
        tool_results.append(
            ToolResult(
                name=tool_name,
                available=available,
                passed=True if available else None,
                return_code=0 if available else None,
                summary="OK" if available else "missing",
            )
        )

    return PipelineResult(
        spec=RequirementSpec(
            original_text=requirement_text,
            numeric_precision="INT8",
            throughput_value=50.0 if requested_backend == "heuristic" else 10.0,
            throughput_unit="TOPS",
            workload_type="transformer" if requested_backend == "heuristic" else "dense_gemm",
            batch_min=1,
            batch_max=4 if requested_backend == "heuristic" else 1,
            interfaces=["AXI4", "DMA", "scratchpad_sram"],
            assumptions=["baseline"],
            ambiguities=["power", "node", "frequency"],
        ),
        architecture=architecture,
        generated=generated,
        tool_results=tool_results,
        score=100.0,
        output_dir=str(output_dir),
        report={
            "available": True,
            "summary": {
                "top_level_case_count": 3,
                "total_cycles": 25,
                "busy_cycles": 19,
                "done_cycles": 3,
                "idle_cycles": 3,
                "memory_path": {
                    "dma_cycles": 8,
                    "load_cycles": 6,
                    "working_set_utilization": 0.5,
                    "total_dma_bits_transferred": 128,
                    "peak_external_bandwidth_gb_per_s": 2.0,
                },
                "compute_path": {
                    "compute_cycles": 4,
                    "clear_cycles": 1,
                    "estimated_mac_operations": 20,
                },
                "top_npu_throughput": {
                    "estimated_effective_tops": estimated_effective_tops,
                    "theoretical_peak_tops": 51.2,
                },
            },
        },
        environment={
            "llm": {
                "requested_backend": requested_backend,
            }
        },
    )


if __name__ == "__main__":
    unittest.main()
