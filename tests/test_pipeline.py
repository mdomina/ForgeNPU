import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from create_npu.harness import VerificationHarness
from create_npu.models import GeneratedDesignBundle
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


if __name__ == "__main__":
    unittest.main()
