import tempfile
import unittest
from pathlib import Path

from create_npu.architect import plan_architecture
from create_npu.compiler import compile_seed_program
from create_npu.harness import VerificationHarness
from create_npu.models import RequirementSpec
from create_npu.reporting import generate_execution_report
from create_npu.rtl_generator import emit_seed_rtl
from create_npu.simulation_wrapper import build_simulation_wrapper_report


class SimulationWrapperTest(unittest.TestCase):
    def test_transformer_wrapper_report_covers_deterministic_and_fuzz_cases(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 20 TOPS transformer batch 1-4.",
            numeric_precision="INT8",
            throughput_value=20.0,
            throughput_unit="TOPS",
            workload_type="transformer",
            batch_min=1,
            batch_max=4,
            sequence_length=1024,
            target_frequency_mhz=1000.0,
        )
        architecture = plan_architecture(spec)
        compiled_program = compile_seed_program(spec=spec, architecture=architecture)

        report = build_simulation_wrapper_report(
            compiled_program=compiled_program.to_dict(),
            architecture=architecture,
            spec=spec,
        )

        self.assertTrue(report["available"])
        self.assertTrue(report["passed"])
        self.assertEqual(report["summary"]["case_count"], 6)
        self.assertEqual(report["summary"]["deterministic_case_count"], 2)
        self.assertEqual(report["summary"]["fuzz_case_count"], 4)
        self.assertEqual(report["summary"]["detected_shape_mismatch_count"], 1)
        self.assertEqual(report["summary"]["detected_sram_overflow_count"], 1)
        self.assertEqual(report["summary"]["detected_reuse_hazard_count"], 1)
        self.assertEqual(report["summary"]["detected_dma_compute_overlap_count"], 1)
        self.assertEqual(report["summary"]["actual_program_dma_compute_overlap_cycles"], 2)

    def test_dense_gemm_actual_program_overlap_is_zero_but_fuzz_overlap_is_detected(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 4 TOPS dense GEMM batch 1.",
            numeric_precision="INT8",
            throughput_value=4.0,
            throughput_unit="TOPS",
            workload_type="dense_gemm",
            batch_min=1,
            batch_max=1,
            target_frequency_mhz=1000.0,
        )
        architecture = plan_architecture(spec)
        compiled_program = compile_seed_program(spec=spec, architecture=architecture)

        report = build_simulation_wrapper_report(
            compiled_program=compiled_program.to_dict(),
            architecture=architecture,
            spec=spec,
        )

        self.assertEqual(report["summary"]["actual_program_dma_compute_overlap_cycles"], 0)
        self.assertEqual(report["summary"]["detected_dma_compute_overlap_count"], 1)

    def test_emit_seed_rtl_persists_wrapper_report_and_harness_passes(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 8 TOPS convolution kernel 3x3 batch 1-2.",
            numeric_precision="INT8",
            throughput_value=8.0,
            throughput_unit="TOPS",
            workload_type="convolution",
            kernel_size=3,
            batch_min=1,
            batch_max=2,
            target_frequency_mhz=900.0,
        )
        architecture = plan_architecture(spec)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            bundle = emit_seed_rtl(
                spec=spec,
                architecture=architecture,
                output_dir=output_dir,
                candidate_id="balanced",
            )

            wrapper_report_path = output_dir / "simulation_wrapper_report.json"
            self.assertTrue(wrapper_report_path.exists())
            self.assertIn(str(wrapper_report_path), bundle.supporting_files)

            report = generate_execution_report(
                bundle=bundle,
                output_dir=output_dir,
                architecture=architecture,
                spec=spec,
            )
            self.assertIn("simulation_wrapper", report["summary"])
            self.assertTrue(report["summary"]["simulation_wrapper"]["passed"])

            tool_results = VerificationHarness(output_dir).run(bundle)
            simulation_wrapper_result = next(
                result for result in tool_results if result.name == "simulation_wrapper"
            )
            self.assertTrue(simulation_wrapper_result.available)
            self.assertTrue(simulation_wrapper_result.passed)


if __name__ == "__main__":
    unittest.main()
