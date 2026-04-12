import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import patch

from create_npu.architect import generate_candidate_architectures, plan_architecture
from create_npu.benchmark import run_regression_benchmark
from create_npu.compiler import TensorDescriptor, compile_seed_program, compiled_program_seed_vectors
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
from create_npu.rtl_generator import (
    _processing_element_template,
    _resolve_width,
    _seed_tile_shape,
    _systolic_tile_template,
    emit_seed_rtl,
)
from create_npu.gemmini_reference import (
    GEMMINI_REFERENCE_CONFIGS,
    compute_gemmini_delta,
    select_nearest_gemmini_reference,
)
from create_npu.scorer import score_design


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

    def test_parse_additional_workloads(self) -> None:
        parser = RequirementParser()
        convolution_spec = parser.parse(
            "Serve una NPU INT8 da 8 TOPS per CNN conv2d con 900 MHz."
        )
        sparse_spec = parser.parse(
            "Serve una NPU INT8 da 1 TOPS per sparse matmul SpMM con batch 4."
        )

        self.assertEqual(convolution_spec.workload_type, "convolution")
        self.assertEqual(sparse_spec.workload_type, "sparse_linear_algebra")
        self.assertNotIn(
            "Workload non esplicitato: assumo dense_gemm come default iniziale.",
            convolution_spec.assumptions,
        )
        self.assertNotIn(
            "Workload non esplicitato: assumo dense_gemm come default iniziale.",
            sparse_spec.assumptions,
        )

    def test_parse_structured_requirement_fields(self) -> None:
        parser = RequirementParser()
        spec = parser.parse(
            "Voglio una NPU INT8 da 20 TOPS per transformer inference a bassa latenza "
            "con HBM, systolic dataflow, sparsity 2:4 e sequence length 4096."
        )

        self.assertEqual(spec.execution_mode, "inference")
        self.assertEqual(spec.optimization_priority, "latency")
        self.assertEqual(spec.offchip_memory_type, "HBM")
        self.assertEqual(spec.preferred_dataflow, "systolic")
        self.assertEqual(spec.sparsity_support, "structured")
        self.assertEqual(spec.sequence_length, 4096)


class CompilerTest(unittest.TestCase):
    def test_compile_seed_program_for_transformer(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 20 TOPS transformer con sequence length 4096 e batch 1-8.",
            numeric_precision="INT8",
            throughput_value=20.0,
            throughput_unit="TOPS",
            workload_type="transformer",
            sequence_length=4096,
            batch_min=1,
            batch_max=8,
            target_frequency_mhz=1000.0,
        )
        architecture = plan_architecture(spec)
        program = compile_seed_program(spec=spec, architecture=architecture)

        self.assertEqual(program.tiling_strategy, "sequence_blocking")
        self.assertEqual(program.slot_count, 2)
        self.assertEqual(program.load_iterations, 2)
        self.assertEqual(program.compute_iterations, 4)
        self.assertEqual(program.store_burst_count, 2)
        self.assertEqual(program.clear_on_done, True)
        self.assertGreaterEqual(program.active_tile_count, 1)
        self.assertEqual(program.problem_shape["sequence_length"], 4096)
        self.assertGreaterEqual(program.problem_shape["hidden_dim"], 256)
        self.assertEqual(len(program.operator_descriptors), 4)
        self.assertEqual(program.operator_descriptors[0]["name"], "qkv_projection")
        self.assertEqual(program.operator_descriptors[1]["op_type"], "batched_gemm")
        self.assertEqual(
            program.mapping_plan["loop_order"],
            ["batch", "sequence_block", "head", "k_block"],
        )
        self.assertGreater(program.estimated_mac_operations, 0)

    def test_compile_seed_program_for_sparse_workload(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 4 TOPS sparse.",
            numeric_precision="INT8",
            throughput_value=4.0,
            throughput_unit="TOPS",
            workload_type="sparse_linear_algebra",
            batch_min=1,
            batch_max=1,
            target_frequency_mhz=900.0,
        )
        architecture = plan_architecture(spec)
        program = compile_seed_program(spec=spec, architecture=architecture)
        vectors = compiled_program_seed_vectors(program)

        self.assertEqual(program.tiling_strategy, "sparse_stream_compaction")
        self.assertEqual(program.slot_count, 1)
        self.assertEqual(program.load_iterations, 1)
        self.assertEqual(program.compute_iterations, 1)
        self.assertEqual(program.operator_descriptors[0]["op_type"], "spmm")
        self.assertGreater(program.problem_shape["nnz"], 0)
        self.assertEqual(vectors["store_burst_count_i"], 1)
        self.assertEqual(vectors["clear_on_done_i"], 1)

    def test_compile_seed_program_for_convolution_shape(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 8 TOPS convolution kernel 5x5 batch 1-4.",
            numeric_precision="INT8",
            throughput_value=8.0,
            throughput_unit="TOPS",
            workload_type="convolution",
            kernel_size=5,
            batch_min=1,
            batch_max=4,
            target_frequency_mhz=900.0,
        )
        architecture = plan_architecture(spec)
        program = compile_seed_program(spec=spec, architecture=architecture)

        self.assertEqual(program.problem_shape["kernel_height"], 5)
        self.assertEqual(program.problem_shape["kernel_width"], 5)
        self.assertEqual(program.operator_descriptors[0]["op_type"], "conv2d")
        self.assertEqual(program.mapping_plan["dataflow"], "weight_stationary")
        self.assertGreater(program.operator_descriptors[0]["output_channels"], 0)

    def test_compile_seed_program_for_output_stationary_dataflow(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 4 TOPS dense GEMM output-stationary.",
            numeric_precision="INT8",
            throughput_value=4.0,
            throughput_unit="TOPS",
            workload_type="dense_gemm",
            preferred_dataflow="output_stationary",
            target_frequency_mhz=900.0,
        )
        architecture = plan_architecture(spec)
        program = compile_seed_program(spec=spec, architecture=architecture)
        vectors = compiled_program_seed_vectors(program)

        self.assertEqual(architecture.family, "output_stationary_array")
        self.assertEqual(program.mapping_plan["dataflow"], "output_stationary")
        self.assertEqual(program.mapping_plan["output_stationary_enabled"], 1)
        self.assertEqual(program.mapping_plan["preload_enabled"], 1)
        self.assertEqual(program.mapping_plan["transpose_inputs"], 1)
        self.assertEqual(program.tiling_strategy, "output_stationary_blocking")
        self.assertEqual(vectors["output_stationary_i"], 1)
        self.assertEqual(vectors["preload_en_i"], 1)
        self.assertEqual(vectors["transpose_inputs_i"], 1)

    def test_compile_tiled_loop_nest_for_gemm(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 10 TOPS dense GEMM batch 1-8.",
            numeric_precision="INT8",
            throughput_value=10.0,
            throughput_unit="TOPS",
            workload_type="dense_gemm",
            batch_min=1,
            batch_max=8,
            target_frequency_mhz=1000.0,
        )
        architecture = plan_architecture(spec)
        program = compile_seed_program(spec=spec, architecture=architecture)

        nest = program.tiled_loop_nest
        self.assertEqual(nest["op_type"], "gemm")
        self.assertTrue(nest["double_buffering_enabled"])
        self.assertEqual(nest["prefetch_distance"], 1)
        self.assertGreaterEqual(nest["activation_tiles_in_flight"], 2)
        self.assertGreaterEqual(nest["weight_tiles_in_flight"], 2)
        self.assertGreater(nest["total_tile_iterations"], 0)
        self.assertGreater(nest["estimated_compute_cycles"], 0)
        self.assertGreater(nest["estimated_memory_cycles"], 0)
        self.assertGreater(nest["estimated_overlap_cycles"], 0)
        self.assertGreater(nest["cluster_occupancy_percent"], 0)

        loops = nest["loops"]
        self.assertGreaterEqual(len(loops), 4)
        dim_names = [loop["dim"] for loop in loops]
        self.assertIn("batch", dim_names)
        self.assertIn("m", dim_names)
        self.assertIn("n", dim_names)
        self.assertIn("k", dim_names)
        for loop in loops:
            self.assertGreater(loop["full_size"], 0)
            self.assertGreater(loop["tile_size"], 0)
            self.assertGreater(loop["iterations"], 0)

    def test_compile_tiled_loop_nest_for_convolution(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 8 TOPS convolution kernel 3x3 batch 1-4.",
            numeric_precision="INT8",
            throughput_value=8.0,
            throughput_unit="TOPS",
            workload_type="convolution",
            kernel_size=3,
            batch_min=1,
            batch_max=4,
            target_frequency_mhz=900.0,
        )
        architecture = plan_architecture(spec)
        program = compile_seed_program(spec=spec, architecture=architecture)

        nest = program.tiled_loop_nest
        self.assertEqual(nest["op_type"], "conv2d")
        self.assertTrue(nest["double_buffering_enabled"])
        self.assertEqual(nest["prefetch_distance"], 1)
        self.assertGreater(nest["total_tile_iterations"], 0)
        self.assertGreater(nest["estimated_overlap_cycles"], 0)

        loops = nest["loops"]
        self.assertGreaterEqual(len(loops), 5)
        dim_names = [loop["dim"] for loop in loops]
        self.assertIn("batch", dim_names)
        self.assertIn("spatial", dim_names)
        self.assertIn("oc", dim_names)
        self.assertIn("ic", dim_names)
        self.assertIn("kernel", dim_names)
        kernel_loop = next(l for l in loops if l["dim"] == "kernel")
        self.assertEqual(kernel_loop["full_size"], 9)
        self.assertEqual(kernel_loop["iterations"], 9)

    def test_compile_cluster_occupancy(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 10 TOPS dense GEMM batch 1-8.",
            numeric_precision="INT8",
            throughput_value=10.0,
            throughput_unit="TOPS",
            workload_type="dense_gemm",
            batch_min=1,
            batch_max=8,
            target_frequency_mhz=1000.0,
        )
        architecture = plan_architecture(spec)
        program = compile_seed_program(spec=spec, architecture=architecture)

        occ = program.cluster_occupancy
        self.assertGreater(occ["total_pe"], 0)
        self.assertGreater(occ["active_pe"], 0)
        self.assertGreater(occ["spatial_utilization_percent"], 0)
        self.assertGreater(occ["compute_bound_occupancy_percent"], 0)
        self.assertTrue(occ["double_buffering_enabled"])
        self.assertGreater(occ["estimated_compute_cycles"], 0)
        self.assertGreater(occ["estimated_overlap_cycles"], 0)
        self.assertGreater(occ["effective_cycles"], 0)
        self.assertGreater(occ["memory_compute_ratio"], 0)

    def test_compile_gemm_accumulation_plan(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 10 TOPS dense GEMM batch 1-8.",
            numeric_precision="INT8",
            throughput_value=10.0,
            throughput_unit="TOPS",
            workload_type="dense_gemm",
            batch_min=1,
            batch_max=8,
            target_frequency_mhz=1000.0,
        )
        architecture = plan_architecture(spec)
        program = compile_seed_program(spec=spec, architecture=architecture)

        accumulation = program.gemm_accumulation
        self.assertTrue(accumulation["enabled"])
        self.assertEqual(accumulation["controller_module"], "gemm_ctrl")
        self.assertEqual(accumulation["accumulator_storage"], "accumulator_buffer")
        self.assertGreater(accumulation["reduction_dim"], 0)
        self.assertGreater(accumulation["k_tile_count"], 1)
        self.assertTrue(accumulation["partial_sum_persistence"])
        self.assertTrue(accumulation["deferred_writeback"])
        self.assertEqual(accumulation["writeback_policy"], "explicit_final_k_tile")

    def test_compile_sparse_has_tiled_loop_nest(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 4 TOPS sparse.",
            numeric_precision="INT8",
            throughput_value=4.0,
            throughput_unit="TOPS",
            workload_type="sparse_linear_algebra",
            target_frequency_mhz=900.0,
        )
        architecture = plan_architecture(spec)
        program = compile_seed_program(spec=spec, architecture=architecture)

        nest = program.tiled_loop_nest
        self.assertEqual(nest["op_type"], "gemm")
        self.assertFalse(nest["double_buffering_enabled"])
        self.assertEqual(nest["prefetch_distance"], 0)
        self.assertEqual(nest["activation_tiles_in_flight"], 1)
        self.assertGreater(nest["total_tile_iterations"], 0)
        self.assertEqual(nest["estimated_overlap_cycles"], 0)


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

    def test_plan_architecture_specializes_additional_workloads(self) -> None:
        dense_architecture = plan_architecture(
            RequirementSpec(
                original_text="NPU INT8 8 TOPS dense GEMM.",
                numeric_precision="INT8",
                throughput_value=8.0,
                throughput_unit="TOPS",
                workload_type="dense_gemm",
                target_frequency_mhz=900.0,
            )
        )
        convolution_architecture = plan_architecture(
            RequirementSpec(
                original_text="NPU INT8 8 TOPS per CNN.",
                numeric_precision="INT8",
                throughput_value=8.0,
                throughput_unit="TOPS",
                workload_type="convolution",
                target_frequency_mhz=900.0,
            )
        )
        sparse_architecture = plan_architecture(
            RequirementSpec(
                original_text="NPU INT8 8 TOPS sparse.",
                numeric_precision="INT8",
                throughput_value=8.0,
                throughput_unit="TOPS",
                workload_type="sparse_linear_algebra",
                target_frequency_mhz=900.0,
            )
        )

        self.assertEqual(convolution_architecture.family, "weight_stationary_array")
        self.assertEqual(sparse_architecture.family, "sparse_pe_mesh")
        self.assertIn("line_buffer", convolution_architecture.modules)
        self.assertIn("sparsity_decoder", sparse_architecture.modules)
        self.assertGreater(
            convolution_architecture.local_sram_kb_per_tile,
            dense_architecture.local_sram_kb_per_tile,
        )

    def test_plan_architecture_uses_structured_requirement_fields(self) -> None:
        baseline_architecture = plan_architecture(
            RequirementSpec(
                original_text="NPU INT8 20 TOPS transformer.",
                numeric_precision="INT8",
                throughput_value=20.0,
                throughput_unit="TOPS",
                workload_type="transformer",
                target_frequency_mhz=1000.0,
            )
        )
        structured_architecture = plan_architecture(
            RequirementSpec(
                original_text="NPU INT8 20 TOPS transformer training low latency with HBM.",
                numeric_precision="INT8",
                throughput_value=20.0,
                throughput_unit="TOPS",
                workload_type="transformer",
                execution_mode="training",
                optimization_priority="latency",
                offchip_memory_type="HBM",
                preferred_dataflow="systolic",
                sparsity_support="structured",
                sequence_length=4096,
                target_frequency_mhz=1000.0,
            )
        )

        self.assertEqual(structured_architecture.family, "tiled_systolic_transformer")
        self.assertGreater(
            structured_architecture.local_sram_kb_per_tile,
            baseline_architecture.local_sram_kb_per_tile,
        )
        self.assertGreater(
            structured_architecture.global_buffer_mb,
            baseline_architecture.global_buffer_mb,
        )
        self.assertGreater(
            structured_architecture.bus_width_bits,
            baseline_architecture.bus_width_bits,
        )
        self.assertGreater(
            structured_architecture.target_frequency_mhz,
            baseline_architecture.target_frequency_mhz,
        )

    def test_generate_candidate_architectures_supports_best_of_n(self) -> None:
        architectures = generate_candidate_architectures(
            RequirementSpec(
                original_text="NPU INT8 20 TOPS transformer.",
                numeric_precision="INT8",
                throughput_value=20.0,
                throughput_unit="TOPS",
                workload_type="transformer",
                target_frequency_mhz=1000.0,
            ),
            max_candidates=5,
        )

        self.assertEqual(
            [architecture.candidate_id for architecture in architectures],
            ["balanced", "throughput_max", "efficiency", "balanced_b1", "throughput_max_b1"],
        )
        self.assertGreater(
            architectures[3].local_sram_kb_per_tile,
            architectures[0].local_sram_kb_per_tile,
        )
        self.assertGreater(
            architectures[4].bus_width_bits,
            architectures[1].bus_width_bits,
        )


class TensorDescriptorTest(unittest.TestCase):
    def _compile_gemm(self, tops: float = 4.0, batch: int = 1) -> object:
        spec = RequirementSpec(
            original_text=f"NPU INT8 {tops} TOPS dense GEMM batch {batch}.",
            numeric_precision="INT8",
            throughput_value=tops,
            throughput_unit="TOPS",
            workload_type="dense_gemm",
            batch_min=1,
            batch_max=batch,
            target_frequency_mhz=1000.0,
        )
        return compile_seed_program(spec=spec, architecture=plan_architecture(spec))

    def test_tensor_descriptors_present(self) -> None:
        program = self._compile_gemm()
        self.assertIn("tensor_descriptors", program.to_dict())
        self.assertIsInstance(program.tensor_descriptors, list)
        self.assertGreater(len(program.tensor_descriptors), 0)

    def test_seed_level_roles(self) -> None:
        program = self._compile_gemm()
        # First 3 descriptors are always the seed-level activation/weight/output
        seed = program.tensor_descriptors[:3]
        roles = [d["role"] for d in seed]
        self.assertEqual(roles, ["activation_input", "weight_input", "output"])

    def test_dtype_int8_for_inputs_int32_for_output(self) -> None:
        program = self._compile_gemm()
        seed = program.tensor_descriptors[:3]
        self.assertEqual(seed[0]["dtype"], "INT8")
        self.assertEqual(seed[1]["dtype"], "INT8")
        self.assertEqual(seed[2]["dtype"], "INT32")

    def test_shape_and_size_bytes_consistent(self) -> None:
        program = self._compile_gemm()
        for desc in program.tensor_descriptors:
            shape = desc["shape"]
            size = desc["size_bytes"]
            itemsize = 1 if desc["dtype"] == "INT8" else (2 if desc["dtype"] in ("INT16", "FP16") else 4)
            expected = 1
            for d in shape:
                expected *= d
            expected *= itemsize
            self.assertEqual(size, expected, msg=f"Mismatch in {desc['name']}: {size} != {expected}")

    def test_base_addr_matches_compiled_program(self) -> None:
        program = self._compile_gemm()
        seed = program.tensor_descriptors[:3]
        self.assertEqual(seed[0]["base_addr"], program.activation_base_addr)
        self.assertEqual(seed[1]["base_addr"], program.weight_base_addr)
        self.assertEqual(seed[2]["base_addr"], program.result_base_addr)

    def test_output_has_producer_op(self) -> None:
        program = self._compile_gemm()
        outputs = [d for d in program.tensor_descriptors if d["role"] == "output"]
        for out in outputs:
            self.assertIsNotNone(out["producer_op"])

    def test_input_tensors_have_consumer_ops(self) -> None:
        program = self._compile_gemm()
        inputs = [
            d for d in program.tensor_descriptors
            if d["role"] in ("activation_input", "weight_input")
        ]
        for inp in inputs:
            self.assertGreater(len(inp["consumer_ops"]), 0)

    def test_per_operator_descriptors_for_transformer(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 20 TOPS transformer seq 512 batch 1.",
            numeric_precision="INT8",
            throughput_value=20.0,
            throughput_unit="TOPS",
            workload_type="transformer",
            sequence_length=512,
            batch_min=1,
            batch_max=1,
            target_frequency_mhz=1000.0,
        )
        program = compile_seed_program(spec=spec, architecture=plan_architecture(spec))
        # transformer has 4 operators → 3 seed + 4*3 per-op = 15 descriptors
        self.assertEqual(len(program.tensor_descriptors), 3 + 4 * 3)
        op_names = {d["operator_name"] for d in program.tensor_descriptors}
        self.assertIn("qkv_projection", op_names)
        self.assertIn("attention_scores", op_names)

    def test_convolution_uses_im2col_layout(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 8 TOPS conv2d kernel 3x3.",
            numeric_precision="INT8",
            throughput_value=8.0,
            throughput_unit="TOPS",
            workload_type="convolution",
            kernel_size=3,
            target_frequency_mhz=1000.0,
        )
        program = compile_seed_program(spec=spec, architecture=plan_architecture(spec))
        # conv2d has 1 operator → 3 seed + 1*3 per-op = 6 descriptors
        self.assertEqual(len(program.tensor_descriptors), 6)
        conv_act = next(
            d for d in program.tensor_descriptors
            if d["role"] == "activation_input" and "conv2d_main" in d["operator_name"]
        )
        self.assertEqual(conv_act["layout"], "im2col_row_major")

    def test_tensor_descriptor_serialises_to_dict(self) -> None:
        desc = TensorDescriptor(
            name="test_act",
            role="activation_input",
            operator_name="gemm_main",
            dtype="INT8",
            shape=[32, 64],
            size_bytes=2048,
            base_addr=0,
            stride_bytes=64,
            layout="row_major",
        )
        d = desc.to_dict()
        self.assertEqual(d["name"], "test_act")
        self.assertEqual(d["shape"], [32, 64])
        self.assertEqual(d["size_bytes"], 2048)
        self.assertEqual(d["producer_op"], None)
        self.assertEqual(d["consumer_ops"], [])


class MemoryPlannerTest(unittest.TestCase):
    def _compile_gemm(self, tops: float = 4.0) -> object:
        spec = RequirementSpec(
            original_text=f"NPU INT8 {tops} TOPS dense GEMM batch 1.",
            numeric_precision="INT8",
            throughput_value=tops,
            throughput_unit="TOPS",
            workload_type="dense_gemm",
            batch_min=1,
            batch_max=1,
            target_frequency_mhz=1000.0,
        )
        return compile_seed_program(spec=spec, architecture=plan_architecture(spec))

    def test_memory_plan_present_in_compiled_program(self) -> None:
        program = self._compile_gemm()
        self.assertIn("memory_plan", program.to_dict())
        self.assertIsInstance(program.memory_plan, dict)

    def test_memory_plan_has_required_fields(self) -> None:
        program = self._compile_gemm()
        mp = program.memory_plan
        for field in ("operator_count", "tensor_count", "peak_sram_bytes_no_reuse",
                      "peak_sram_bytes_with_reuse", "reuse_savings_bytes",
                      "reuse_savings_pct", "per_step_live_bytes", "allocations"):
            self.assertIn(field, mp, msg=f"Missing field: {field}")

    def test_memory_plan_single_op_gemm_has_positive_peak(self) -> None:
        program = self._compile_gemm()
        mp = program.memory_plan
        self.assertGreater(mp["peak_sram_bytes_no_reuse"], 0)
        self.assertGreater(mp["peak_sram_bytes_with_reuse"], 0)

    def test_memory_plan_single_op_peak_equals_sum_of_tensors(self) -> None:
        # With a single operator, all per-op tensors are co-live → no savings
        program = self._compile_gemm()
        mp = program.memory_plan
        total = sum(alloc["size_bytes"] for alloc in mp["allocations"])
        self.assertEqual(mp["peak_sram_bytes_no_reuse"], total)

    def test_memory_plan_allocations_cover_all_per_op_tensors(self) -> None:
        program = self._compile_gemm()
        mp = program.memory_plan
        # 1 gemm_main operator → 3 per-operator descriptors
        self.assertEqual(mp["tensor_count"], 3)
        self.assertEqual(len(mp["allocations"]), 3)

    def test_memory_plan_per_step_live_bytes_length_matches_operator_count(self) -> None:
        program = self._compile_gemm()
        mp = program.memory_plan
        self.assertEqual(len(mp["per_step_live_bytes"]), mp["operator_count"])

    def test_memory_plan_transformer_multi_op_no_reuse_geq_with_reuse(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 20 TOPS transformer seq 512 batch 1.",
            numeric_precision="INT8",
            throughput_value=20.0,
            throughput_unit="TOPS",
            workload_type="transformer",
            sequence_length=512,
            batch_min=1,
            batch_max=1,
            target_frequency_mhz=1000.0,
        )
        program = compile_seed_program(spec=spec, architecture=plan_architecture(spec))
        mp = program.memory_plan
        self.assertGreaterEqual(
            mp["peak_sram_bytes_no_reuse"],
            mp["peak_sram_bytes_with_reuse"],
        )

    def test_memory_plan_reuse_savings_consistent(self) -> None:
        program = self._compile_gemm()
        mp = program.memory_plan
        expected_savings = mp["peak_sram_bytes_no_reuse"] - mp["peak_sram_bytes_with_reuse"]
        self.assertEqual(mp["reuse_savings_bytes"], expected_savings)
        expected_pct = round(100.0 * expected_savings / max(1, mp["peak_sram_bytes_no_reuse"]), 2)
        self.assertAlmostEqual(mp["reuse_savings_pct"], expected_pct, places=5)


class LoweredProgramTest(unittest.TestCase):
    def _compile(self, workload_type: str, tops: float = 4.0, **kwargs) -> object:
        spec = RequirementSpec(
            original_text=f"NPU INT8 {tops} TOPS {workload_type} batch 1.",
            numeric_precision="INT8",
            throughput_value=tops,
            throughput_unit="TOPS",
            workload_type=workload_type,
            batch_min=1,
            batch_max=kwargs.get("batch_max", 1),
            target_frequency_mhz=1000.0,
            kernel_size=kwargs.get("kernel_size"),
            sequence_length=kwargs.get("sequence_length"),
        )
        return compile_seed_program(spec=spec, architecture=plan_architecture(spec))

    def test_lowered_program_present(self) -> None:
        program = self._compile("dense_gemm")
        self.assertIn("lowered_program", program.to_dict())
        self.assertIsInstance(program.lowered_program, list)
        self.assertGreater(len(program.lowered_program), 0)

    def test_dense_gemm_has_gemm_and_relu(self) -> None:
        program = self._compile("dense_gemm")
        ops = program.lowered_program
        self.assertEqual(ops[0]["op_type"], "gemm")
        self.assertEqual(ops[1]["op_type"], "relu")

    def test_dense_gemm_gemm_primitive_is_systolic(self) -> None:
        program = self._compile("dense_gemm")
        self.assertEqual(program.lowered_program[0]["hardware_primitive"], "systolic_gemm")

    def test_dense_gemm_gemm_is_fusable(self) -> None:
        program = self._compile("dense_gemm")
        self.assertTrue(program.lowered_program[0]["fusable_with_next"])

    def test_convolution_has_conv_relu_pool(self) -> None:
        program = self._compile("convolution", tops=8.0, kernel_size=3)
        op_types = [op["op_type"] for op in program.lowered_program]
        self.assertIn("conv2d", op_types)
        self.assertIn("relu", op_types)
        self.assertIn("pool", op_types)
        self.assertEqual(op_types[0], "conv2d")

    def test_convolution_pool_is_max(self) -> None:
        program = self._compile("convolution", tops=8.0, kernel_size=3)
        pool_op = next(op for op in program.lowered_program if op["op_type"] == "pool")
        self.assertEqual(pool_op["hardware_primitive"], "vector_pool")
        self.assertEqual(pool_op["params"]["pool_type"], "max")

    def test_convolution_conv_uses_im2col(self) -> None:
        program = self._compile("convolution", tops=8.0, kernel_size=3)
        conv_op = next(op for op in program.lowered_program if op["op_type"] == "conv2d")
        self.assertEqual(conv_op["hardware_primitive"], "im2col_conv")
        self.assertTrue(conv_op["params"]["via_im2col"])

    def test_transformer_has_slice_reduce_concat(self) -> None:
        program = self._compile("transformer", tops=20.0, sequence_length=512)
        op_types = [op["op_type"] for op in program.lowered_program]
        self.assertIn("slice", op_types)
        self.assertIn("reduce", op_types)
        self.assertIn("concat", op_types)

    def test_transformer_reduce_is_softmax(self) -> None:
        program = self._compile("transformer", tops=20.0, sequence_length=512)
        reduce_op = next(op for op in program.lowered_program if op["op_type"] == "reduce")
        self.assertEqual(reduce_op["hardware_primitive"], "vector_reduce")
        self.assertEqual(reduce_op["params"]["op"], "softmax")

    def test_sparse_has_spmm_and_relu(self) -> None:
        program = self._compile("sparse_linear_algebra", tops=4.0)
        ops = program.lowered_program
        self.assertEqual(ops[0]["op_type"], "spmm")
        self.assertEqual(ops[0]["hardware_primitive"], "sparse_gemm")
        self.assertEqual(ops[1]["op_type"], "relu")

    def test_all_required_op_types_covered(self) -> None:
        required = {"gemm", "conv2d", "relu", "reduce", "concat", "slice", "pool"}
        found: set = set()
        for workload, kwargs in [
            ("dense_gemm", {}),
            ("convolution", {"tops": 8.0, "kernel_size": 3}),
            ("transformer", {"tops": 20.0, "sequence_length": 512}),
        ]:
            prog = self._compile(workload, **kwargs)
            found.update(op["op_type"] for op in prog.lowered_program)
        self.assertTrue(required.issubset(found), f"Missing op types: {required - found}")

    def test_lowered_op_serialises_to_dict(self) -> None:
        from create_npu.compiler import LoweredOp
        op = LoweredOp(
            name="test_op",
            op_type="relu",
            hardware_primitive="vector_elementwise",
            inputs=["in_tensor"],
            outputs=["out_tensor"],
            fusable_with_next=False,
            params={"shape": [16, 16]},
        )
        d = op.to_dict()
        self.assertEqual(d["name"], "test_op")
        self.assertEqual(d["op_type"], "relu")
        self.assertEqual(d["inputs"], ["in_tensor"])
        self.assertFalse(d["fusable_with_next"])


class DispatchScheduleTest(unittest.TestCase):
    def _compile(self, workload_type: str, tops: float = 4.0, **kwargs) -> object:
        spec = RequirementSpec(
            original_text=f"NPU INT8 {tops} TOPS {workload_type} batch 1.",
            numeric_precision="INT8",
            throughput_value=tops,
            throughput_unit="TOPS",
            workload_type=workload_type,
            batch_min=1,
            batch_max=kwargs.get("batch_max", 1),
            target_frequency_mhz=1000.0,
            kernel_size=kwargs.get("kernel_size"),
            sequence_length=kwargs.get("sequence_length"),
        )
        return compile_seed_program(spec=spec, architecture=plan_architecture(spec))

    def test_dispatch_schedule_present(self) -> None:
        program = self._compile("dense_gemm")
        self.assertIn("dispatch_schedule", program.to_dict())
        self.assertIsInstance(program.dispatch_schedule, dict)

    def test_dispatch_schedule_has_required_fields(self) -> None:
        program = self._compile("dense_gemm")
        ds = program.dispatch_schedule
        for key in ("entries", "total_issue_cycles", "total_completion_cycles",
                    "barrier_count", "stall_cycles", "raw_hazard_count",
                    "engine_utilization"):
            self.assertIn(key, ds, msg=f"Missing key: {key}")

    def test_dense_gemm_first_entry_is_compute(self) -> None:
        program = self._compile("dense_gemm")
        entries = program.dispatch_schedule["entries"]
        self.assertEqual(entries[0]["engine"], "compute")
        self.assertFalse(entries[0]["barrier_before"])

    def test_dense_gemm_relu_has_raw_dep_and_barrier(self) -> None:
        program = self._compile("dense_gemm")
        entries = program.dispatch_schedule["entries"]
        relu_entry = next(e for e in entries if e["op_name"] == "relu_post_gemm")
        self.assertIn("gemm_main", relu_entry["depends_on"])
        self.assertTrue(relu_entry["barrier_before"])

    def test_issue_cycle_ordering(self) -> None:
        program = self._compile("dense_gemm")
        entries = program.dispatch_schedule["entries"]
        # Each entry must issue at or after the previous
        for i in range(1, len(entries)):
            self.assertGreaterEqual(entries[i]["issue_cycle"], entries[i - 1]["issue_cycle"])

    def test_completion_cycle_geq_issue_plus_latency(self) -> None:
        program = self._compile("dense_gemm")
        for entry in program.dispatch_schedule["entries"]:
            self.assertGreaterEqual(entry["completion_cycle"], entry["issue_cycle"])

    def test_raw_hazard_count_geq_barrier_count(self) -> None:
        # Every barrier implies a RAW hazard; inverse not necessarily true
        program = self._compile("dense_gemm")
        ds = program.dispatch_schedule
        self.assertGreaterEqual(ds["raw_hazard_count"], ds["barrier_count"])

    def test_engine_utilization_between_0_and_1(self) -> None:
        program = self._compile("dense_gemm")
        for eng, util in program.dispatch_schedule["engine_utilization"].items():
            self.assertGreaterEqual(util, 0.0, msg=f"{eng} utilization < 0")
            self.assertLessEqual(util, 1.0, msg=f"{eng} utilization > 1")

    def test_transformer_has_multiple_entries(self) -> None:
        program = self._compile("transformer", tops=20.0, sequence_length=512)
        entries = program.dispatch_schedule["entries"]
        self.assertGreater(len(entries), 5)

    def test_dispatch_entry_serialises_to_dict(self) -> None:
        from create_npu.compiler import DispatchEntry
        entry = DispatchEntry(
            op_name="test_gemm",
            engine="compute",
            depends_on=[],
            barrier_before=False,
            issue_cycle=0,
            completion_cycle=4,
            stall_cycles=0,
        )
        d = entry.to_dict()
        self.assertEqual(d["op_name"], "test_gemm")
        self.assertEqual(d["engine"], "compute")
        self.assertEqual(d["completion_cycle"], 4)
        self.assertFalse(d["barrier_before"])


class GemminiReferenceTest(unittest.TestCase):
    def _make_architecture(
        self,
        estimated_tops: float = 10.0,
        family: str = "tiled_systolic_array",
        tile_rows: int = 16,
        tile_cols: int = 16,
        tile_count: int = 4,
        pe_count: int = 1024,
        local_sram_kb_per_tile: int = 256,
    ) -> ArchitectureCandidate:
        return ArchitectureCandidate(
            candidate_id="balanced",
            family=family,
            tile_rows=tile_rows,
            tile_cols=tile_cols,
            tile_count=tile_count,
            pe_rows=tile_rows,
            pe_cols=tile_cols * tile_count,
            pe_count=pe_count,
            local_sram_kb_per_tile=local_sram_kb_per_tile,
            global_buffer_mb=4,
            bus_width_bits=512,
            target_frequency_mhz=1000.0,
            estimated_tops=estimated_tops,
            estimated_power_watts=20.0,
            estimated_area_mm2=1.0,
        )

    def test_reference_configs_are_complete(self) -> None:
        required_keys = {
            "name", "description", "tile_rows", "tile_cols", "tile_count",
            "dataflow", "local_sram_kb", "accumulator_kb", "bus_width_bits",
            "frequency_mhz", "estimated_tops", "architecture_family", "benchmark_shapes",
        }
        for config_name, config in GEMMINI_REFERENCE_CONFIGS.items():
            for key in required_keys:
                self.assertIn(key, config, f"Chiave '{key}' mancante in {config_name}")
            self.assertGreater(len(config["benchmark_shapes"]), 0, config_name)
            for shape in config["benchmark_shapes"]:
                self.assertIn("macs", shape, config_name)
                self.assertGreater(shape["macs"], 0, config_name)

    def test_select_nearest_reference_small(self) -> None:
        arch = self._make_architecture(estimated_tops=0.1)
        self.assertEqual(select_nearest_gemmini_reference(arch), "gemmini_small")

    def test_select_nearest_reference_medium(self) -> None:
        arch = self._make_architecture(estimated_tops=0.5)
        self.assertEqual(select_nearest_gemmini_reference(arch), "gemmini_medium")

    def test_select_nearest_reference_large(self) -> None:
        arch = self._make_architecture(estimated_tops=5.0)
        self.assertEqual(select_nearest_gemmini_reference(arch), "gemmini_large")

    def test_delta_structure_for_systolic_gemm(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 10 TOPS dense GEMM.",
            numeric_precision="INT8",
            throughput_value=10.0,
            throughput_unit="TOPS",
            workload_type="dense_gemm",
        )
        arch = self._make_architecture(estimated_tops=10.0)
        delta = compute_gemmini_delta(spec=spec, architecture=arch)

        self.assertIn("reference_name", delta)
        self.assertIn("reference_config", delta)
        self.assertIn("candidate_vs_reference", delta)
        self.assertIn("convergence", delta)
        self.assertIn("convergence_score", delta)
        self.assertIn("convergence_reasons", delta)
        self.assertIn("requirement_vs_mapping", delta)
        self.assertIn("throughput_comparison", delta)

        ref = delta["reference_config"]
        self.assertIn("dataflow", ref)
        self.assertIn("local_sram_kb", ref)
        self.assertIn("estimated_tops", ref)

        cmp = delta["candidate_vs_reference"]
        self.assertIn("pe_count", cmp)
        self.assertIn("total_sram_kb", cmp)
        self.assertIn("estimated_tops", cmp)
        self.assertIn("dataflow_match", cmp)
        self.assertIn("architecture_family_match", cmp)

    def test_systolic_gemm_converges(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 10 TOPS dense GEMM.",
            numeric_precision="INT8",
            throughput_value=10.0,
            throughput_unit="TOPS",
            workload_type="dense_gemm",
        )
        arch = self._make_architecture(
            estimated_tops=10.0, family="tiled_systolic_array"
        )
        delta = compute_gemmini_delta(spec=spec, architecture=arch)
        self.assertEqual(delta["convergence"], "converges")
        self.assertTrue(delta["candidate_vs_reference"]["dataflow_match"])
        self.assertTrue(delta["candidate_vs_reference"]["architecture_family_match"])

    def test_sparse_workload_diverges(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 4 TOPS sparse.",
            numeric_precision="INT8",
            throughput_value=4.0,
            throughput_unit="TOPS",
            workload_type="sparse_linear_algebra",
            sparsity_support="structured",
        )
        arch = self._make_architecture(
            estimated_tops=4.0, family="sparse_pe_mesh"
        )
        delta = compute_gemmini_delta(spec=spec, architecture=arch)
        self.assertEqual(delta["convergence"], "diverges")
        self.assertFalse(delta["candidate_vs_reference"]["dataflow_match"])
        self.assertFalse(delta["candidate_vs_reference"]["architecture_family_match"])

    def test_throughput_comparison_has_all_benchmarks(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 0.5 TOPS dense GEMM.",
            numeric_precision="INT8",
            throughput_value=0.5,
            throughput_unit="TOPS",
            workload_type="dense_gemm",
        )
        arch = self._make_architecture(estimated_tops=0.5)
        delta = compute_gemmini_delta(spec=spec, architecture=arch)
        ref_name = delta["reference_name"]
        ref_shapes = GEMMINI_REFERENCE_CONFIGS[ref_name]["benchmark_shapes"]
        self.assertEqual(
            len(delta["throughput_comparison"]), len(ref_shapes)
        )
        for entry in delta["throughput_comparison"]:
            self.assertIn("benchmark", entry)
            self.assertIn("macs", entry)
            self.assertIn("estimated_speedup", entry)
            self.assertGreater(entry["macs"], 0)

    def test_compiled_program_dataflow_overrides_family_dataflow(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 4 TOPS output-stationary.",
            numeric_precision="INT8",
            throughput_value=4.0,
            throughput_unit="TOPS",
            workload_type="dense_gemm",
            preferred_dataflow="output_stationary",
        )
        arch = self._make_architecture(
            estimated_tops=4.0, family="output_stationary_array"
        )
        compiled = {"mapping_plan": {"dataflow": "output_stationary"}}
        delta = compute_gemmini_delta(spec=spec, architecture=arch, compiled_program=compiled)
        self.assertFalse(delta["candidate_vs_reference"]["dataflow_match"])
        self.assertIn("output_stationary", delta["convergence_reasons"][0])

    def test_delta_in_execution_report_summary(self) -> None:
        spec = RequirementSpec(
            original_text="NPU INT8 10 TOPS dense GEMM.",
            numeric_precision="INT8",
            throughput_value=10.0,
            throughput_unit="TOPS",
            workload_type="dense_gemm",
        )
        architecture = plan_architecture(spec)
        with tempfile.TemporaryDirectory() as tmp:
            from create_npu.rtl_generator import emit_seed_rtl
            from create_npu.reporting import generate_execution_report
            bundle = emit_seed_rtl(spec=spec, architecture=architecture, output_dir=Path(tmp))
            report = generate_execution_report(
                bundle=bundle, output_dir=Path(tmp), architecture=architecture, spec=spec
            )
        self.assertTrue(report["available"])
        summary = report["summary"]
        self.assertIn("gemmini_reference_delta", summary)
        delta = summary["gemmini_reference_delta"]
        self.assertIn("convergence", delta)
        self.assertIn("reference_config", delta)
        self.assertIn("throughput_comparison", delta)


class ScoringTest(unittest.TestCase):
    def test_score_design_rewards_matching_workload_family(self) -> None:
        convolution_spec = RequirementSpec(
            original_text="NPU INT8 8 TOPS per CNN.",
            numeric_precision="INT8",
            throughput_value=8.0,
            throughput_unit="TOPS",
            workload_type="convolution",
            target_frequency_mhz=900.0,
        )
        matching_architecture = ArchitectureCandidate(
            candidate_id="matching",
            family="weight_stationary_array",
            tile_rows=16,
            tile_cols=16,
            tile_count=4,
            pe_rows=32,
            pe_cols=32,
            pe_count=1024,
            local_sram_kb_per_tile=384,
            global_buffer_mb=5,
            bus_width_bits=768,
            target_frequency_mhz=900.0,
            estimated_tops=8.0,
            estimated_power_watts=40.0,
            estimated_area_mm2=5.0,
        )
        mismatched_architecture = ArchitectureCandidate(
            candidate_id="mismatched",
            family="tiled_systolic_array",
            tile_rows=16,
            tile_cols=16,
            tile_count=4,
            pe_rows=32,
            pe_cols=32,
            pe_count=1024,
            local_sram_kb_per_tile=384,
            global_buffer_mb=5,
            bus_width_bits=768,
            target_frequency_mhz=900.0,
            estimated_tops=8.0,
            estimated_power_watts=40.0,
            estimated_area_mm2=5.0,
        )

        self.assertGreater(
            score_design(convolution_spec, matching_architecture, []),
            score_design(convolution_spec, mismatched_architecture, []),
        )

    def test_score_design_rewards_structured_preference_alignment(self) -> None:
        throughput_spec = RequirementSpec(
            original_text="NPU INT8 32 TOPS throughput-oriented con HBM.",
            numeric_precision="INT8",
            throughput_value=32.0,
            throughput_unit="TOPS",
            workload_type="dense_gemm",
            optimization_priority="throughput",
            offchip_memory_type="HBM",
            target_frequency_mhz=1000.0,
        )
        throughput_architecture = ArchitectureCandidate(
            candidate_id="throughput_max",
            family="tiled_systolic_array",
            tile_rows=16,
            tile_cols=16,
            tile_count=8,
            pe_rows=64,
            pe_cols=64,
            pe_count=4096,
            local_sram_kb_per_tile=512,
            global_buffer_mb=8,
            bus_width_bits=1536,
            target_frequency_mhz=1150.0,
            estimated_tops=32.0,
            estimated_power_watts=60.0,
            estimated_area_mm2=8.0,
        )
        balanced_architecture = ArchitectureCandidate(
            candidate_id="balanced",
            family="tiled_systolic_array",
            tile_rows=16,
            tile_cols=16,
            tile_count=8,
            pe_rows=64,
            pe_cols=64,
            pe_count=4096,
            local_sram_kb_per_tile=512,
            global_buffer_mb=8,
            bus_width_bits=1024,
            target_frequency_mhz=1000.0,
            estimated_tops=32.0,
            estimated_power_watts=60.0,
            estimated_area_mm2=8.0,
        )

        self.assertGreater(
            score_design(throughput_spec, throughput_architecture, []),
            score_design(throughput_spec, balanced_architecture, []),
        )


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
                    / "compiled_program.json"
                ).exists()
            )
            compiled_program = json.loads(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "compiled_program.json"
                ).read_text(encoding="utf-8")
            )
            self.assertIn("problem_shape", compiled_program)
            self.assertIn("operator_descriptors", compiled_program)
            self.assertIn("mapping_plan", compiled_program)
            self.assertGreaterEqual(len(compiled_program["operator_descriptors"]), 1)
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "rtl"
                    / "systolic_tile.sv"
                ).exists()
            )
            expected_rows = int(result.architecture.tile_rows)
            expected_cols = int(result.architecture.tile_cols)
            expected_tile_count = int(result.architecture.tile_count)
            top_npu_rtl = (
                Path(result.output_dir)
                / "candidates"
                / "balanced"
                / "rtl"
                / "top_npu.sv"
            ).read_text(encoding="utf-8")
            systolic_tile_rtl = (
                Path(result.output_dir)
                / "candidates"
                / "balanced"
                / "rtl"
                / "systolic_tile.sv"
            ).read_text(encoding="utf-8")
            self.assertIn(f"parameter int ROWS = {expected_rows}", top_npu_rtl)
            self.assertIn(f"parameter int COLS = {expected_cols}", top_npu_rtl)
            self.assertIn(f"parameter int TILE_COUNT = {expected_tile_count}", top_npu_rtl)
            self.assertIn(f"parameter int ROWS = {expected_rows}", systolic_tile_rtl)
            self.assertIn(f"parameter int COLS = {expected_cols}", systolic_tile_rtl)
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
                    / "accumulator_buffer.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "rtl"
                    / "gemm_ctrl.sv"
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
                    / "cluster_control.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "rtl"
                    / "cluster_interconnect.sv"
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
            cluster_control_rtl = (
                Path(result.output_dir)
                / "candidates"
                / "balanced"
                / "rtl"
                / "cluster_control.sv"
            ).read_text(encoding="utf-8")
            cluster_interconnect_rtl = (
                Path(result.output_dir)
                / "candidates"
                / "balanced"
                / "rtl"
                / "cluster_interconnect.sv"
            ).read_text(encoding="utf-8")
            tile_compute_rtl = (
                Path(result.output_dir)
                / "candidates"
                / "balanced"
                / "rtl"
                / "tile_compute_unit.sv"
            ).read_text(encoding="utf-8")
            self.assertIn("module cluster_control", cluster_control_rtl)
            self.assertIn(
                f"parameter int TILE_COUNT = {expected_tile_count}",
                cluster_control_rtl,
            )
            self.assertIn("module cluster_interconnect", cluster_interconnect_rtl)
            self.assertIn("cluster_interconnect #(", top_npu_rtl)
            self.assertIn("cluster_control #(", top_npu_rtl)
            self.assertIn("accumulator_buffer #(", tile_compute_rtl)
            gemm_ctrl_rtl = (
                Path(result.output_dir)
                / "candidates"
                / "balanced"
                / "rtl"
                / "gemm_ctrl.sv"
            ).read_text(encoding="utf-8")
            self.assertIn("module gemm_ctrl", gemm_ctrl_rtl)
            self.assertIn("accumulator_buffer #(", gemm_ctrl_rtl)
            self.assertIn("tile_store_payloads", top_npu_rtl)
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
                    / "accumulator_buffer_tb.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "tb"
                    / "gemm_ctrl_tb.sv"
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
                    / "cluster_control_tb.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "tb"
                    / "cluster_interconnect_tb.sv"
                ).exists()
            )
            self.assertTrue(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "tb"
                    / "systolic_tile_rect_tb.sv"
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
            verification_vectors = json.loads(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "verification_vectors.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(len(verification_vectors["top_npu_stress"]), 3)
            self.assertEqual(len(verification_vectors["accumulator_buffer"]), 1)
            self.assertEqual(len(verification_vectors["gemm_ctrl"]), 1)
            self.assertIn("randomized", verification_vectors["top_npu_stress"][0]["stress_tags"])
            self.assertEqual(len(verification_vectors["scheduler_stress"]), 1)
            self.assertEqual(len(verification_vectors["cluster_control_stress"]), 1)
            self.assertEqual(len(verification_vectors["cluster_interconnect_stress"]), 1)
            self.assertIn("multi_slot", verification_vectors["scheduler_stress"][0]["stress_tags"])
            self.assertIn("control_routing", verification_vectors["cluster_control_stress"][0]["stress_tags"])
            self.assertIn("store_fanout", verification_vectors["cluster_interconnect_stress"][0]["stress_tags"])

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
            reference_coverage = next(
                result for result in payload["tool_results"] if result["name"] == "reference_coverage"
            )
            self.assertTrue(reference_coverage["available"])
            self.assertTrue(reference_coverage["passed"])
            self.assertIn("Coverage score", reference_coverage["summary"])
            self.assertIn(
                "coverage_report.json",
                " ".join(payload["generated"]["supporting_files"]),
            )
            self.assertGreaterEqual(payload["score"], 0.0)
            self.assertIn(
                f"{expected_rows}x{expected_cols}",
                " ".join(payload["generated"]["notes"]),
            )
            self.assertNotIn("ridotta", " ".join(payload["generated"]["notes"]))
            self.assertIn(
                f"tile count architetturale {expected_tile_count}",
                " ".join(payload["generated"]["notes"]),
            )
            self.assertIn("cluster_interconnect", " ".join(payload["generated"]["notes"]))
            self.assertIn("accumulator_buffer", " ".join(payload["generated"]["notes"]))
            self.assertIn("gemm_ctrl", " ".join(payload["generated"]["notes"]))
            self.assertIn("gemm_accumulation", compiled_program)
            self.assertEqual(
                compiled_program["gemm_accumulation"]["controller_module"],
                "gemm_ctrl",
            )

            report = payload["report"]
            self.assertTrue(Path(report["path"]).exists())
            self.assertEqual(report["summary"]["top_level_case_count"], 4)
            self.assertEqual(report["summary"]["requirement_profile"]["execution_mode"], "inference")
            self.assertEqual(
                report["summary"]["requirement_profile"]["optimization_priority"],
                "balanced",
            )
            self.assertEqual(report["summary"]["workload_profile"]["workload_type"], "transformer")
            self.assertEqual(
                report["summary"]["workload_profile"]["selected_architecture_family"],
                "tiled_systolic_transformer",
            )
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
                    "STORE",
                    "STORE",
                    "FLUSH",
                    "CLEAR",
                    "DONE",
                    "IDLE",
                    "DMA_ACT",
                    "DMA_WGT",
                    "LOAD",
                    "LOAD",
                    "COMPUTE",
                    "STORE",
                    "STORE",
                    "FLUSH",
                    "DONE",
                    "IDLE",
                    "DMA_ACT",
                    "DMA_WGT",
                    "LOAD",
                    "LOAD",
                    "COMPUTE",
                    "STORE",
                    "STORE",
                    "FLUSH",
                    "DONE",
                    "IDLE",
                    "DMA_ACT",
                    "DMA_ACT",
                    "DMA_WGT",
                    "LOAD",
                    "LOAD",
                    "COMPUTE",
                    "STORE",
                    "STORE",
                    "FLUSH",
                    "DONE",
                    "IDLE",
                ],
            )
            self.assertEqual(report["summary"]["total_cycles"], 45)
            self.assertEqual(report["summary"]["busy_cycles"], 37)
            self.assertEqual(report["summary"]["done_cycles"], 4)
            self.assertEqual(report["summary"]["idle_cycles"], 4)
            self.assertEqual(report["summary"]["control_path"]["dma_broadcast_cycles"], 11)
            self.assertEqual(report["summary"]["control_path"]["load_broadcast_cycles"], 8)
            self.assertEqual(report["summary"]["control_path"]["compute_broadcast_cycles"], 5)
            self.assertEqual(report["summary"]["control_path"]["flush_broadcast_cycles"], 4)
            self.assertEqual(report["summary"]["control_path"]["clear_broadcast_cycles"], 1)
            self.assertEqual(report["summary"]["control_path"]["store_broadcast_cycles"], 8)
            self.assertEqual(report["summary"]["control_path"]["peak_active_tiles"], 2)
            self.assertEqual(
                report["summary"]["control_path"]["average_active_tiles_per_active_cycle"],
                1.216216,
            )
            self.assertEqual(
                report["summary"]["control_path"]["average_active_tiles_per_compute_cycle"],
                1.2,
            )
            self.assertEqual(report["summary"]["interconnect_path"]["dma_fanout_cycles"], 10)
            self.assertEqual(report["summary"]["interconnect_path"]["load_fanout_cycles"], 7)
            self.assertEqual(report["summary"]["interconnect_path"]["compute_fanout_cycles"], 5)
            self.assertEqual(report["summary"]["interconnect_path"]["dma_accept_cycles"], 10)
            self.assertEqual(report["summary"]["interconnect_path"]["load_accept_cycles"], 7)
            self.assertEqual(report["summary"]["interconnect_path"]["store_accept_cycles"], 7)
            self.assertEqual(report["summary"]["interconnect_path"]["dma_backpressure_cycles"], 1)
            self.assertEqual(report["summary"]["interconnect_path"]["load_backpressure_cycles"], 1)
            self.assertEqual(report["summary"]["interconnect_path"]["store_backpressure_cycles"], 1)
            self.assertEqual(report["summary"]["interconnect_path"]["store_lane_write_cycles"], 18)
            self.assertEqual(report["summary"]["interconnect_path"]["peak_store_lanes_per_cycle"], 4)
            self.assertEqual(report["summary"]["memory_path"]["dma_cycles"], 11)
            self.assertEqual(report["summary"]["memory_path"]["dma_activation_cycles"], 6)
            self.assertEqual(report["summary"]["memory_path"]["dma_weight_cycles"], 5)
            self.assertEqual(report["summary"]["memory_path"]["load_cycles"], 8)
            self.assertEqual(report["summary"]["memory_path"]["store_cycles"], 7)
            self.assertEqual(report["summary"]["memory_path"]["max_scratchpad_depth"], 4)
            self.assertEqual(report["summary"]["memory_path"]["activation_slots_touched"], 2)
            self.assertEqual(report["summary"]["memory_path"]["weight_slots_touched"], 2)
            self.assertEqual(report["summary"]["memory_path"]["peak_activation_slots_live"], 2)
            self.assertEqual(report["summary"]["memory_path"]["peak_weight_slots_live"], 2)
            self.assertEqual(report["summary"]["memory_path"]["working_set_utilization"], 0.5)
            self.assertEqual(report["summary"]["memory_path"]["total_dma_bits_transferred"], 176)
            self.assertEqual(report["summary"]["memory_path"]["result_elements_stored"], 18)
            self.assertEqual(report["summary"]["memory_path"]["total_store_bits_transferred"], 576)
            self.assertEqual(report["summary"]["memory_path"]["total_memory_bits_transferred"], 752)
            self.assertEqual(
                report["summary"]["memory_path"]["average_dma_bits_per_dma_cycle"],
                16.0,
            )
            self.assertEqual(report["summary"]["memory_path"]["peak_dma_bits_per_cycle"], 16)
            self.assertEqual(
                report["summary"]["memory_path"]["average_store_bits_per_store_cycle"],
                82.285714,
            )
            self.assertEqual(report["summary"]["memory_path"]["peak_store_bits_per_cycle"], 128)
            self.assertEqual(
                report["summary"]["memory_path"]["effective_external_bandwidth_gb_per_s"],
                2.088889,
            )
            self.assertEqual(
                report["summary"]["memory_path"]["peak_external_bandwidth_gb_per_s"],
                16.0,
            )
            self.assertEqual(
                report["summary"]["memory_path"]["theoretical_bus_bandwidth_gb_per_s"],
                160.0,
            )
            self.assertEqual(
                report["summary"]["memory_path"]["bus_bandwidth_utilization"],
                0.013056,
            )
            self.assertEqual(
                report["summary"]["memory_path"]["peak_bus_bandwidth_utilization"],
                0.1,
            )
            self.assertEqual(report["summary"]["compute_path"]["compute_cycles"], 5)
            self.assertEqual(report["summary"]["compute_path"]["flush_cycles"], 4)
            self.assertEqual(report["summary"]["compute_path"]["clear_cycles"], 1)
            self.assertEqual(report["summary"]["compute_path"]["estimated_mac_operations"], 24)
            self.assertIn("performance_counters", report["summary"])
            self.assertEqual(
                report["summary"]["performance_counters"]["compute"]["active_cycles"],
                5,
            )
            self.assertEqual(
                report["summary"]["performance_counters"]["dma"]["active_cycles"],
                11,
            )
            self.assertEqual(
                report["summary"]["performance_counters"]["stall"]["total_backpressure_cycles"],
                3,
            )
            self.assertEqual(
                report["summary"]["performance_counters"]["stall"]["dispatch_barrier_count"],
                report["summary"]["compiled_program"]["dispatch_schedule"]["barrier_count"],
            )
            self.assertEqual(
                report["summary"]["performance_counters"]["overlap"]["memory_compute_overlap_cycles"],
                report["summary"]["overlap_metrics"]["memory_compute_overlap_cycles"],
            )
            self.assertGreater(
                report["summary"]["performance_counters"]["cluster_occupancy"][
                    "runtime_tile_utilization_while_compute_percent"
                ],
                0.0,
            )
            self.assertGreaterEqual(
                report["summary"]["performance_counters"]["cluster_occupancy"][
                    "compiled_compute_bound_occupancy_percent"
                ],
                0.0,
            )
            self.assertTrue(report["summary"]["top_npu_throughput"]["available"])
            self.assertEqual(
                report["summary"]["top_npu_throughput"]["estimation_model"],
                "peak_scaled_by_compute_duty_cycle",
            )
            self.assertEqual(report["summary"]["top_npu_throughput"]["total_cycles"], 45)
            self.assertEqual(report["summary"]["top_npu_throughput"]["compute_cycles"], 5)
            self.assertEqual(
                report["summary"]["top_npu_throughput"]["scheduler_overhead_cycles"],
                40,
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
                0.111111,
            )
            self.assertEqual(
                report["summary"]["top_npu_throughput"]["compute_duty_cycle_while_busy"],
                0.135135,
            )
            self.assertEqual(
                report["summary"]["top_npu_throughput"]["estimated_effective_ops_per_cycle"],
                5688.888889,
            )
            self.assertEqual(
                report["summary"]["top_npu_throughput"]["estimated_effective_tops"],
                5.688889,
            )

            report_payload = json.loads(Path(report["path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(report_payload["cases"]), 4)
            self.assertEqual(len(report_payload["stress_cases"]), 3)
            self.assertEqual(len(report_payload["internal_stress_cases"]), 3)
            self.assertEqual(report["summary"]["stress_verification"]["stress_case_count"], 3)
            self.assertEqual(report["summary"]["stress_verification"]["backpressure_case_count"], 2)
            self.assertEqual(report["summary"]["stress_verification"]["flush_case_count"], 2)
            self.assertEqual(report["summary"]["stress_verification"]["multi_tile_case_count"], 2)
            self.assertEqual(report["summary"]["stress_verification"]["max_stress_tile_count"], 3)
            self.assertIn(
                "randomized_multitile_backpressure_seed17",
                report["summary"]["stress_verification"]["case_names"],
            )
            self.assertIn(
                "flush",
                report["summary"]["stress_verification"]["covered_tags"],
            )
            self.assertEqual(
                report["summary"]["internal_stress_verification"]["stress_case_count"],
                3,
            )
            self.assertEqual(
                report["summary"]["internal_stress_verification"]["scheduler_case_count"],
                1,
            )
            self.assertEqual(
                report["summary"]["internal_stress_verification"]["cluster_control_case_count"],
                1,
            )
            self.assertEqual(
                report["summary"]["internal_stress_verification"]["cluster_interconnect_case_count"],
                1,
            )
            self.assertEqual(
                report["summary"]["internal_stress_verification"]["covered_modules"],
                ["cluster_control", "cluster_interconnect", "scheduler"],
            )
            self.assertIn(
                "scheduler:restart_window_seed41",
                report["summary"]["internal_stress_verification"]["case_names"],
            )
            self.assertEqual(report_payload["cases"][1]["name"], "single_slot_single_compute_top")
            self.assertEqual(report_payload["cases"][2]["name"], "dual_tile_broadcast_compute_top")
            self.assertEqual(report_payload["cases"][3]["name"], "single_tile_backpressure_top")
            self.assertEqual(report_payload["cases"][1]["program"]["slot_count_i"], 1)
            self.assertEqual(report_payload["cases"][1]["program"]["load_iterations_i"], 2)
            self.assertEqual(report_payload["cases"][1]["program"]["compute_iterations_i"], 1)
            self.assertEqual(report_payload["cases"][1]["program"]["clear_on_done_i"], 0)
            self.assertEqual(report_payload["cases"][2]["tile_count"], 2)
            self.assertEqual(report_payload["cases"][2]["program"]["tile_enable_i"], [1, 1])
            self.assertEqual(report_payload["cases"][3]["program"]["tile_dma_ready_i"], [1])
            self.assertEqual(report_payload["cases"][3]["program"]["tile_load_ready_i"], [1])
            self.assertEqual(report_payload["cases"][3]["program"]["store_ready_i"], 1)
            self.assertEqual(report_payload["cases"][0]["trace"][0]["control_path"]["tile_dma_valid"], [1])
            self.assertEqual(
                report_payload["cases"][0]["trace"][0]["interconnect_path"]["tile_dma_valid"],
                [1],
            )
            self.assertEqual(report_payload["cases"][0]["trace"][2]["control_path"]["dma_bank"], 1)
            self.assertEqual(
                report_payload["cases"][0]["trace"][4]["control_path"]["tile_load_vector_en"],
                [1],
            )
            self.assertEqual(
                report_payload["cases"][0]["trace"][8]["control_path"]["tile_store_results_en"],
                [1],
            )
            self.assertEqual(report_payload["cases"][0]["trace"][0]["memory_path"]["dma_bank"], 0)
            self.assertEqual(report_payload["cases"][0]["trace"][2]["memory_path"]["dma_bank"], 1)
            self.assertEqual(
                report_payload["cases"][0]["trace"][5]["memory_path"]["activation_read_bank"],
                1,
            )
            self.assertEqual(
                report_payload["cases"][0]["trace"][5]["memory_path"]["weight_read_bank"],
                1,
            )
            self.assertEqual(report_payload["cases"][0]["trace"][8]["memory_path"]["store_valid"], 1)
            self.assertEqual(
                report_payload["cases"][0]["trace"][8]["memory_path"]["store_payload"],
                [42, 16, 0, 0],
            )
            self.assertEqual(
                report_payload["cases"][0]["trace"][8]["memory_path"]["store_valid_mask"],
                [1, 1, 0, 0],
            )
            self.assertEqual(
                report_payload["cases"][0]["trace"][8]["interconnect_path"]["result_write_valid_mask"],
                [1, 1, 0, 0],
            )
            self.assertEqual(report_payload["cases"][0]["trace"][9]["memory_path"]["store_valid"], 1)
            self.assertEqual(
                report_payload["cases"][0]["trace"][9]["memory_path"]["store_payload"],
                [0, 0, 40, 24],
            )
            self.assertIn("flush_pipeline", report_payload["cases"][0]["trace"][10]["event_tags"])
            self.assertEqual(report_payload["cases"][1]["trace"][5]["memory_path"]["store_valid"], 1)
            self.assertEqual(
                report_payload["cases"][1]["trace"][6]["memory_path"]["store_valid_mask"],
                [0, 0, 1, 1],
            )
            self.assertEqual(
                report_payload["cases"][2]["top_npu_throughput"]["seed_peak_macs_per_cycle"],
                8,
            )
            self.assertEqual(
                report_payload["cases"][2]["trace"][4]["control_path"]["tile_compute_en"],
                [1, 1],
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
                7.314286,
            )
            self.assertEqual(
                report_payload["cases"][1]["top_npu_throughput"]["estimated_effective_tops"],
                5.12,
            )
            self.assertEqual(
                report_payload["cases"][2]["top_npu_throughput"]["estimated_effective_tops"],
                5.12,
            )
            self.assertEqual(
                report_payload["cases"][3]["trace"][1]["scheduler_state"]["name"],
                "DMA_ACT",
            )
            self.assertEqual(
                report_payload["cases"][3]["trace"][1]["interconnect_path"]["dma_backpressure"],
                1,
            )
            self.assertEqual(
                report_payload["cases"][3]["trace"][4]["interconnect_path"]["load_backpressure"],
                1,
            )
            self.assertEqual(
                report_payload["cases"][3]["trace"][7]["interconnect_path"]["store_backpressure"],
                1,
            )
            self.assertEqual(
                report_payload["cases"][3]["summary"]["interconnect_path"]["dma_backpressure_cycles"],
                1,
            )
            self.assertEqual(
                report_payload["cases"][3]["summary"]["interconnect_path"]["load_backpressure_cycles"],
                1,
            )
            self.assertEqual(
                report_payload["cases"][3]["summary"]["interconnect_path"]["store_backpressure_cycles"],
                1,
            )
            self.assertEqual(
                report_payload["cases"][3]["summary"]["memory_path"]["store_cycles"],
                1,
            )
            self.assertEqual(
                report_payload["cases"][3]["top_npu_throughput"]["estimated_effective_tops"],
                4.654545,
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
            self.assertIn(
                "compiled_program.json",
                " ".join(balanced_candidate["generated"]["supporting_files"]),
            )
            self.assertIn("compiled_program", report["summary"])
            self.assertIn("tiling_strategy", report["summary"]["compiled_program"])
            self.assertIn("problem_shape", report["summary"]["compiled_program"])
            self.assertIn("operator_descriptors", report["summary"]["compiled_program"])

    def test_pipeline_handles_convolution_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CreateNPUPipeline(base_output_dir=Path(temp_dir))
            result = pipeline.run(
                "Voglio una NPU INT8 da 8 TOPS per CNN conv2d kernel 3x3 "
                "a bassa latenza con HBM e weight-stationary dataflow, batch 1-2.",
                num_candidates=1,
            )

            self.assertEqual(result.spec.workload_type, "convolution")
            self.assertEqual(result.architecture.family, "weight_stationary_array")
            report_payload = json.loads(
                Path(result.report["path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                report_payload["summary"]["workload_profile"]["workload_type"],
                "convolution",
            )
            self.assertEqual(
                report_payload["summary"]["workload_profile"]["preferred_architecture_family"],
                "weight_stationary_array",
            )
            self.assertEqual(
                report_payload["summary"]["requirement_profile"]["optimization_priority"],
                "latency",
            )
            self.assertEqual(
                report_payload["summary"]["requirement_profile"]["offchip_memory_type"],
                "HBM",
            )
            self.assertEqual(
                report_payload["summary"]["requirement_profile"]["preferred_dataflow"],
                "weight_stationary",
            )
            self.assertEqual(
                report_payload["summary"]["requirement_profile"]["resolved_dataflow"],
                "weight_stationary",
            )
            self.assertEqual(
                report_payload["summary"]["requirement_profile"]["kernel_size"],
                3,
            )
            self.assertEqual(
                report_payload["summary"]["compiled_program"]["tiling_strategy"],
                "weight_stationary_window",
            )
            self.assertEqual(
                report_payload["summary"]["compiled_program"]["slot_count"],
                2,
            )
            self.assertEqual(
                report_payload["summary"]["compiled_program"]["operator_descriptors"][0]["op_type"],
                "conv2d",
            )

    def test_pipeline_propagates_output_stationary_dataflow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CreateNPUPipeline(base_output_dir=Path(temp_dir))
            result = pipeline.run(
                "Voglio una NPU INT8 da 4 TOPS per dense GEMM output-stationary, batch 1-2.",
                num_candidates=1,
            )

            self.assertEqual(result.architecture.family, "output_stationary_array")
            compiled_program = json.loads(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "compiled_program.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(compiled_program["mapping_plan"]["dataflow"], "output_stationary")
            self.assertEqual(compiled_program["mapping_plan"]["output_stationary_enabled"], 1)
            self.assertEqual(compiled_program["mapping_plan"]["preload_enabled"], 1)
            self.assertEqual(compiled_program["mapping_plan"]["transpose_inputs"], 1)
            top_npu_rtl = (
                Path(result.output_dir)
                / "candidates"
                / "balanced"
                / "rtl"
                / "top_npu.sv"
            ).read_text(encoding="utf-8")
            tile_rtl = (
                Path(result.output_dir)
                / "candidates"
                / "balanced"
                / "rtl"
                / "systolic_tile.sv"
            ).read_text(encoding="utf-8")
            self.assertIn("input  logic output_stationary_i", top_npu_rtl)
            self.assertIn("input  logic preload_en_i", top_npu_rtl)
            self.assertIn("input  logic transpose_inputs_i", top_npu_rtl)
            self.assertIn("input  logic output_stationary_i", tile_rtl)
            self.assertIn("input  logic preload_en_i", tile_rtl)
            self.assertIn("input  logic transpose_inputs_i", tile_rtl)

            report_payload = json.loads(Path(result.report["path"]).read_text(encoding="utf-8"))
            self.assertEqual(
                report_payload["summary"]["dataflow_profile"]["compiled_program_dataflow"],
                "output_stationary",
            )
            self.assertEqual(
                report_payload["summary"]["dataflow_profile"]["rtl_output_stationary_enabled"],
                1,
            )
            self.assertEqual(
                report_payload["summary"]["dataflow_profile"]["rtl_preload_enabled"],
                1,
            )
            self.assertEqual(
                report_payload["summary"]["dataflow_profile"]["rtl_transpose_inputs_enabled"],
                1,
            )
            self.assertEqual(report_payload["cases"][0]["program"]["output_stationary_i"], 1)
            self.assertEqual(report_payload["cases"][0]["program"]["preload_en_i"], 1)
            self.assertEqual(report_payload["cases"][0]["program"]["transpose_inputs_i"], 1)
            self.assertEqual(
                report_payload["summary"]["compiled_program"]["tiling_strategy"],
                "output_stationary_blocking",
            )
            verification_vectors = json.loads(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "verification_vectors.json"
                ).read_text(encoding="utf-8")
            )
            systolic_case_names = {
                case["name"] for case in verification_vectors["systolic_tile"]
            }
            tile_case_names = {
                case["name"] for case in verification_vectors["tile_compute_unit"]
            }
            self.assertIn("preload_transpose_output_stationary", systolic_case_names)
            self.assertIn(
                "preload_transpose_output_stationary_from_scratchpad",
                tile_case_names,
            )

    def test_pipeline_emits_decoupled_scheduler_reference_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CreateNPUPipeline(base_output_dir=Path(temp_dir))
            result = pipeline.run(
                "Voglio una NPU INT8 da 6 TOPS per dense GEMM con batch 1-2.",
                num_candidates=1,
            )

            verification_vectors = json.loads(
                (
                    Path(result.output_dir)
                    / "candidates"
                    / "balanced"
                    / "verification_vectors.json"
                ).read_text(encoding="utf-8")
            )
            scheduler_case = next(
                case
                for case in verification_vectors["scheduler"]
                if case["name"] == "decoupled_overlap_with_hazard_tracking"
            )
            self.assertTrue(
                any(
                    int(step["expected"]["load_vector_en_o"])
                    and int(step["expected"]["compute_en_o"])
                    for step in scheduler_case["steps"]
                )
            )
            self.assertTrue(
                any(
                    int(step["expected"]["store_results_en_o"])
                    and int(step["expected"]["compute_en_o"])
                    for step in scheduler_case["steps"]
                )
            )
            self.assertTrue(
                any(int(step["expected"].get("hazard_wait_o", 0)) for step in scheduler_case["steps"])
            )
            self.assertTrue(
                any(
                    int(step["expected"].get("decoupled_mode_o", 0))
                    and int(step["expected"].get("execute_queue_depth_o", 0)) > 0
                    for step in scheduler_case["steps"]
                )
            )

    def test_pipeline_report_exposes_scheduler_overlap_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CreateNPUPipeline(base_output_dir=Path(temp_dir))
            result = pipeline.run(
                "Voglio una NPU INT8 da 6 TOPS per dense GEMM con batch 1-2.",
                num_candidates=1,
            )

            report_payload = json.loads(Path(result.report["path"]).read_text(encoding="utf-8"))
            self.assertIn("scheduler_queue_metrics", report_payload["summary"])
            self.assertIn("overlap_metrics", report_payload["summary"])
            self.assertIn("performance_counters", report_payload["summary"])
            self.assertIn("scheduler_status", report_payload["cases"][0]["trace"][0])
            self.assertIn("hazard_wait", report_payload["cases"][0]["trace"][0]["scheduler_status"])
            self.assertIn("load_queue_depth", report_payload["cases"][0]["trace"][0]["scheduler_status"])
            self.assertIn("store_queue_depth", report_payload["cases"][0]["trace"][0]["scheduler_status"])
            self.assertGreaterEqual(
                report_payload["summary"]["scheduler_queue_metrics"]["max_load_queue_depth"],
                0,
            )
            self.assertGreaterEqual(
                report_payload["summary"]["scheduler_queue_metrics"]["max_execute_queue_depth"],
                0,
            )
            self.assertGreaterEqual(
                report_payload["summary"]["scheduler_queue_metrics"]["max_store_queue_depth"],
                0,
            )
            self.assertGreaterEqual(
                report_payload["summary"]["overlap_metrics"]["memory_compute_overlap_cycles"],
                report_payload["summary"]["overlap_metrics"]["dma_compute_overlap_cycles"],
            )
            self.assertGreaterEqual(
                report_payload["summary"]["overlap_metrics"]["memory_compute_overlap_cycles"],
                report_payload["summary"]["overlap_metrics"]["load_compute_overlap_cycles"],
            )
            self.assertEqual(
                report_payload["summary"]["performance_counters"]["compute"]["active_cycles"],
                report_payload["summary"]["compute_path"]["compute_cycles"],
            )
            self.assertEqual(
                report_payload["summary"]["performance_counters"]["dma"]["active_cycles"],
                report_payload["summary"]["memory_path"]["dma_cycles"],
            )
            self.assertEqual(
                report_payload["summary"]["performance_counters"]["stall"]["hazard_wait_cycles"],
                report_payload["summary"]["scheduler_queue_metrics"]["hazard_wait_cycles"],
            )
            self.assertEqual(
                report_payload["summary"]["performance_counters"]["overlap"][
                    "memory_compute_overlap_cycles"
                ],
                report_payload["summary"]["overlap_metrics"]["memory_compute_overlap_cycles"],
            )
            self.assertAlmostEqual(
                report_payload["summary"]["performance_counters"]["compute"]["duty_cycle"],
                report_payload["summary"]["compute_path"]["compute_cycles"]
                / float(report_payload["summary"]["total_cycles"]),
                places=6,
            )
            self.assertGreaterEqual(
                report_payload["summary"]["performance_counters"]["cluster_occupancy"][
                    "runtime_tile_utilization_while_compute_percent"
                ],
                report_payload["summary"]["performance_counters"]["cluster_occupancy"][
                    "runtime_tile_utilization_percent"
                ],
            )

    def test_pipeline_archives_dataset_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_output_dir = Path(temp_dir)
            pipeline = CreateNPUPipeline(base_output_dir=base_output_dir)
            result = pipeline.run(
                "Voglio una NPU INT8 da 50 TOPS con supporto transformer e batch 1-4.",
                num_candidates=3,
            )

            self.assertIn("dataset", result.to_dict())
            dataset = result.dataset
            self.assertTrue(Path(dataset["dataset_dir"]).exists())
            self.assertTrue(Path(dataset["run_sample_path"]).exists())
            self.assertTrue(Path(dataset["candidate_samples_path"]).exists())
            self.assertTrue(Path(dataset["run_samples_jsonl"]).exists())
            self.assertTrue(Path(dataset["candidate_samples_jsonl"]).exists())
            self.assertTrue(Path(dataset["manifest_path"]).exists())

            manifest = json.loads(Path(dataset["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_sample_count"], 1)
            self.assertEqual(manifest["candidate_sample_count"], 3)
            self.assertEqual(manifest["good_candidate_count"], 1)
            self.assertEqual(manifest["bad_candidate_count"], 2)
            self.assertEqual(manifest["accepted_candidate_count"], 1)
            self.assertEqual(manifest["rejected_candidate_count"], 2)
            self.assertIn("average_candidate_reward", manifest)

            run_sample = json.loads(Path(dataset["run_sample_path"]).read_text(encoding="utf-8"))
            self.assertEqual(run_sample["selected_candidate_id"], result.architecture.candidate_id)
            self.assertEqual(run_sample["candidate_labels"]["balanced"], "good")
            self.assertEqual(run_sample["candidate_labels"]["throughput_max"], "bad")
            self.assertEqual(run_sample["candidate_labels"]["efficiency"], "bad")
            self.assertEqual(run_sample["learning_feedback_summary"]["accepted_candidate_count"], 1)

            candidate_samples = json.loads(
                Path(dataset["candidate_samples_path"]).read_text(encoding="utf-8")
            )
            self.assertEqual(len(candidate_samples), 3)
            selected_sample = next(sample for sample in candidate_samples if sample["selected"])
            self.assertEqual(selected_sample["candidate_id"], result.architecture.candidate_id)
            self.assertEqual(selected_sample["label"], "good")
            self.assertIn("selected_best", selected_sample["label_reasons"])
            self.assertTrue(selected_sample["learning_feedback"]["accept"])
            self.assertEqual(
                selected_sample["learning_feedback"]["feedback_bucket"],
                "accept_selected_verified",
            )

    def test_pipeline_best_of_n_and_learning_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CreateNPUPipeline(base_output_dir=Path(temp_dir))
            result = pipeline.run(
                "Voglio una NPU INT8 da 20 TOPS con supporto transformer e batch 1-4.",
                num_candidates=5,
            )

            self.assertEqual(len(result.candidate_results), 5)
            self.assertEqual(
                [candidate["candidate_id"] for candidate in result.candidate_results],
                ["balanced", "throughput_max", "efficiency", "balanced_b1", "throughput_max_b1"],
            )
            self.assertEqual(result.search["strategy"], "deterministic_best_of_n")
            self.assertEqual(result.search["requested_candidate_count"], 5)
            self.assertEqual(result.search["generated_candidate_count"], 5)
            self.assertEqual(result.search["variant_candidate_count"], 2)
            self.assertEqual(result.search["profile_counts"]["balanced"], 2)
            self.assertEqual(result.search["profile_counts"]["throughput_max"], 2)
            self.assertEqual(result.search["profile_counts"]["efficiency"], 1)
            self.assertEqual(
                result.search["ranked_candidates"][0]["candidate_id"],
                result.architecture.candidate_id,
            )
            self.assertEqual(result.learning_feedback["accepted_candidate_count"], 1)
            self.assertEqual(result.learning_feedback["rejected_candidate_count"], 4)
            self.assertEqual(
                result.learning_feedback["reward_ranking"][0]["candidate_id"],
                result.architecture.candidate_id,
            )
            self.assertTrue(
                all("learning_feedback" in candidate for candidate in result.candidate_results)
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

    def test_emit_seed_rtl_applies_rtl_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = RequirementSpec(
                original_text="Voglio una NPU INT8 da 10 TOPS per dense GEMM.",
                numeric_precision="INT8",
                throughput_value=10.0,
                throughput_unit="TOPS",
                workload_type="dense_gemm",
                target_frequency_mhz=1000.0,
            )
            architecture = plan_architecture(spec)
            operand_width, _ = _resolve_width(spec.numeric_precision)
            acc_width = max(32, operand_width * 4)
            seed_rows, seed_cols = _seed_tile_shape(
                architecture.tile_rows,
                architecture.tile_cols,
            )
            processing_override = (
                "// llm override processing_element\n"
                + _processing_element_template(
                    operand_width=operand_width,
                    acc_width=acc_width,
                )
            )
            tile_override = (
                "// llm override systolic_tile\n"
                + _systolic_tile_template(
                    operand_width=operand_width,
                    acc_width=acc_width,
                    seed_rows=seed_rows,
                    seed_cols=seed_cols,
                )
            )
            bundle = emit_seed_rtl(
                spec=spec,
                architecture=architecture,
                output_dir=Path(temp_dir),
                candidate_id=architecture.candidate_id,
                generator_backend="llm",
                rtl_overrides={
                    "processing_element.sv": processing_override,
                    "systolic_tile.sv": tile_override,
                },
            )

            self.assertIn("Applicati override RTL", " ".join(bundle.notes))
            processing_payload = (Path(temp_dir) / "rtl" / "processing_element.sv").read_text(
                encoding="utf-8"
            )
            tile_payload = (Path(temp_dir) / "rtl" / "systolic_tile.sv").read_text(
                encoding="utf-8"
            )
            self.assertTrue(processing_payload.startswith("// llm override processing_element"))
            self.assertTrue(tile_payload.startswith("// llm override systolic_tile"))

    def test_pipeline_live_llm_backend_compares_against_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline = CreateNPUPipeline(base_output_dir=Path(temp_dir))

            def fake_live_generation(prompt_payload):
                parsed_payload = {
                    "summary": "Override minimali compatibili con il seed.",
                    "expected_benefits": ["Mantiene lo stesso comportamento del seed."],
                    "verification_risks": ["Basso rischio: vengono alterati solo commenti."],
                    "rtl_overrides": [
                        {
                            "file_name": "processing_element.sv",
                            "module_name": "processing_element",
                            "source": (
                                "// llm live override processing_element\n"
                                + prompt_payload["seed_modules"]["processing_element.sv"]
                            ),
                            "rationale": "Traccia il path live senza cambiare l'interfaccia.",
                        },
                        {
                            "file_name": "systolic_tile.sv",
                            "module_name": "systolic_tile",
                            "source": (
                                "// llm live override systolic_tile\n"
                                + prompt_payload["seed_modules"]["systolic_tile.sv"]
                            ),
                            "rationale": "Traccia il path live senza cambiare l'interfaccia.",
                        },
                    ],
                }
                return parsed_payload, {
                    "id": "resp_test",
                    "output_text": json.dumps(parsed_payload),
                }

            live_status = {
                "requested_backend": "llm",
                "effective_backend": "llm",
                "available": True,
                "reason": "Backend LLM pronto per chiamate live.",
                "model": "gpt-test",
                "base_url": "https://example.invalid/v1",
                "live_generation_enabled": True,
            }

            with patch("create_npu.environment.probe_llm_backend", return_value=live_status):
                with patch("create_npu.generator_backend.probe_llm_backend", return_value=live_status):
                    with patch(
                        "create_npu.generator_backend._run_live_llm_generation",
                        side_effect=fake_live_generation,
                    ):
                        result = pipeline.run(
                            "Voglio una NPU INT8 da 10 TOPS per dense GEMM.",
                            num_candidates=1,
                            generator_backend="llm",
                            llm_model="gpt-test",
                        )

            self.assertEqual(result.generated.generator_backend, "llm")
            self.assertEqual(result.environment["llm"]["effective_backend"], "llm")
            self.assertIn("llm_response.json", " ".join(result.generated.supporting_files))
            self.assertIn(
                "backend_comparison.json",
                " ".join(result.generated.supporting_files),
            )

            candidate = result.candidate_results[0]
            self.assertIn("backend_comparison", candidate)
            self.assertTrue(candidate["backend_comparison"]["selected_llm_variant"])
            self.assertEqual(candidate["backend_comparison"]["score_delta_llm_minus_seed"], 0.0)
            self.assertTrue(str(candidate["output_dir"]).endswith("/llm_variant"))
            self.assertTrue(
                str(candidate["backend_comparison"]["heuristic_seed"]["output_dir"]).endswith(
                    "/heuristic_seed"
                )
            )
            candidate_root = Path(result.output_dir) / "candidates" / candidate["candidate_id"]
            self.assertTrue((candidate_root / "llm_request.json").exists())
            self.assertTrue((candidate_root / "llm_response.json").exists())
            self.assertTrue((candidate_root / "llm_structured_output.json").exists())
            self.assertTrue((candidate_root / "backend_comparison.json").exists())
            processing_payload = (
                Path(candidate["output_dir"]) / "rtl" / "processing_element.sv"
            ).read_text(encoding="utf-8")
            self.assertTrue(processing_payload.startswith("// llm live override processing_element"))


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

    def test_yosys_uses_bounded_top_level_parameters_from_reference_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "candidate"
            output_dir.mkdir(parents=True, exist_ok=True)
            rtl_dir = output_dir / "rtl"
            rtl_dir.mkdir(parents=True, exist_ok=True)
            (rtl_dir / "mac_unit.sv").write_text("module mac_unit; endmodule\n", encoding="utf-8")
            (rtl_dir / "top_npu.sv").write_text(
                "\n".join(
                    [
                        "module top_npu #(",
                        "  parameter int ROWS = 32,",
                        "  parameter int COLS = 32,",
                        "  parameter int DEPTH = 4,",
                        "  parameter int TILE_COUNT = 25",
                        ") ();",
                        "endmodule",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            harness = VerificationHarness(output_dir)
            reference_cases_path = output_dir / "verification_vectors.json"
            reference_cases_path.write_text(
                json.dumps(
                    {
                        "top_npu": [
                            {"name": "single_tile", "rows": 2, "cols": 2, "depth": 4, "tile_count": 1},
                            {"name": "dual_tile", "rows": 2, "cols": 3, "depth": 5, "tile_count": 2},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            bundle = GeneratedDesignBundle(
                rtl_files=[str(rtl_dir / "mac_unit.sv"), str(rtl_dir / "top_npu.sv")],
                testbench_files=["tb/top_npu_tb.sv"],
                primary_module="top_npu",
                reference_cases_path=str(reference_cases_path),
            )
            commands = []

            def fake_run(command, cwd, capture_output, text, check):
                commands.append((command, cwd))
                return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

            with patch("create_npu.harness.shutil.which", return_value="/usr/bin/yosys"):
                with patch("create_npu.harness.subprocess.run", side_effect=fake_run):
                    result = harness._run_yosys_synth(bundle)

            self.assertTrue(result.available)
            self.assertTrue(result.passed)
            self.assertEqual(len(commands), 1)
            script_path = output_dir / "run_yosys.ys"
            self.assertTrue(script_path.exists())
            script_payload = script_path.read_text(encoding="utf-8")
            self.assertIn("chparam -set ROWS 2 -set COLS 3 -set DEPTH 5 -set TILE_COUNT 2 top_npu", script_payload)
            self.assertIn("synth -top top_npu", script_payload)
            bounded_top = output_dir / "yosys_rtl" / "top_npu.sv"
            self.assertTrue(bounded_top.exists())
            bounded_payload = bounded_top.read_text(encoding="utf-8")
            self.assertIn("parameter int ROWS = 2", bounded_payload)
            self.assertIn("parameter int COLS = 3", bounded_payload)
            self.assertIn("parameter int DEPTH = 5", bounded_payload)
            self.assertIn("parameter int TILE_COUNT = 2", bounded_payload)


class BenchmarkTest(unittest.TestCase):
    def test_regression_benchmark_passes_with_strict_toolchain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            def fake_run(*, requirement_text, output_dir, num_candidates, generator_backend, llm_model):
                case_id = Path(output_dir).name
                output_dir.mkdir(parents=True, exist_ok=True)
                if case_id == "convolution_weight_stationary_mapping":
                    return _make_fake_pipeline_result(
                        output_dir=output_dir,
                        requirement_text=requirement_text,
                        candidate_id="balanced",
                        generator_backend="heuristic",
                        requested_backend="heuristic",
                        supporting_files=[str(output_dir / "compiled_program.json")],
                        estimated_effective_tops=0.98304,
                        workload_type="convolution",
                        architecture_family="weight_stationary_array",
                        compiled_program_summary={
                            "tiling_strategy": "weight_stationary_window",
                            "problem_shape": {
                                "kernel_height": 3,
                                "output_channels": 64,
                            },
                            "mapping_plan": {
                                "dataflow": "weight_stationary",
                            },
                            "operator_descriptors": [
                                {
                                    "op_type": "conv2d",
                                }
                            ],
                        },
                    )
                if case_id == "sparse_spmm_mapping":
                    return _make_fake_pipeline_result(
                        output_dir=output_dir,
                        requirement_text=requirement_text,
                        candidate_id="balanced",
                        generator_backend="heuristic",
                        requested_backend="heuristic",
                        supporting_files=[str(output_dir / "compiled_program.json")],
                        estimated_effective_tops=0.4096,
                        workload_type="sparse_linear_algebra",
                        architecture_family="sparse_pe_mesh",
                        compiled_program_summary={
                            "tiling_strategy": "sparse_stream_compaction",
                            "problem_shape": {
                                "density_percent": 50,
                            },
                            "mapping_plan": {
                                "dataflow": "sparse_streaming",
                            },
                            "operator_descriptors": [
                                {
                                    "op_type": "spmm",
                                    "block_structure": "2:4",
                                }
                            ],
                        },
                    )
                if case_id == "tensor_descriptor_gemm":
                    compiled_program_json = output_dir / "compiled_program.json"
                    compiled_program_json.write_text("{}", encoding="utf-8")
                    return _make_fake_pipeline_result(
                        output_dir=output_dir,
                        requirement_text=requirement_text,
                        candidate_id="balanced",
                        generator_backend="heuristic",
                        requested_backend="heuristic",
                        supporting_files=[str(compiled_program_json)],
                        estimated_effective_tops=1.137778,
                        workload_type="dense_gemm",
                        architecture_family="tiled_systolic_array",
                        compiled_program_summary={
                            "tiling_strategy": "gemm_blocking",
                            "problem_shape": {"m": 16, "k": 32, "n": 16},
                            "mapping_plan": {"dataflow": "weight_stationary"},
                            "operator_descriptors": [{"op_type": "gemm", "name": "gemm_main"}],
                            "tensor_descriptors": [
                                {"role": "activation_input", "dtype": "INT8", "shape": [16, 32], "size_bytes": 512, "base_addr": 0, "stride_bytes": 32, "layout": "row_major", "name": "activation_in", "operator_name": "gemm_main", "requires_transpose": False, "quantization_scale": 1.0, "producer_op": None, "consumer_ops": ["gemm_main"]},
                                {"role": "weight_input", "dtype": "INT8", "shape": [32, 16], "size_bytes": 512, "base_addr": 0, "stride_bytes": 16, "layout": "row_major", "name": "weight_in", "operator_name": "gemm_main", "requires_transpose": False, "quantization_scale": 1.0, "producer_op": None, "consumer_ops": ["gemm_main"]},
                                {"role": "output", "dtype": "INT32", "shape": [16, 16], "size_bytes": 1024, "base_addr": 2, "stride_bytes": 64, "layout": "row_major", "name": "result_out", "operator_name": "gemm_main", "requires_transpose": False, "quantization_scale": 1.0, "producer_op": "gemm_main", "consumer_ops": []},
                            ],
                        },
                    )
                if case_id == "gemmini_reference_delta_gemm":
                    compiled_program_json = output_dir / "compiled_program.json"
                    compiled_program_json.write_text("{}", encoding="utf-8")
                    return _make_fake_pipeline_result(
                        output_dir=output_dir,
                        requirement_text=requirement_text,
                        candidate_id="balanced",
                        generator_backend="heuristic",
                        requested_backend="heuristic",
                        supporting_files=[str(compiled_program_json)],
                        estimated_effective_tops=1.137778,
                        workload_type="dense_gemm",
                        architecture_family="tiled_systolic_array",
                        extra_summary={
                            "gemmini_reference_delta": {
                                "reference_name": "gemmini_medium",
                                "reference_config": {
                                    "dataflow": "weight_stationary",
                                    "architecture_family": "tiled_systolic_array",
                                },
                                "candidate_vs_reference": {
                                    "dataflow_match": True,
                                },
                                "convergence": "converges",
                            },
                        },
                    )
                if case_id == "dispatch_schedule_gemm":
                    compiled_program_json = output_dir / "compiled_program.json"
                    compiled_program_json.write_text("{}", encoding="utf-8")
                    return _make_fake_pipeline_result(
                        output_dir=output_dir,
                        requirement_text=requirement_text,
                        candidate_id="balanced",
                        generator_backend="heuristic",
                        requested_backend="heuristic",
                        supporting_files=[str(compiled_program_json)],
                        estimated_effective_tops=1.137778,
                        workload_type="dense_gemm",
                        architecture_family="tiled_systolic_array",
                        compiled_program_summary={
                            "tiling_strategy": "gemm_blocking",
                            "problem_shape": {"m": 16, "k": 32, "n": 16},
                            "mapping_plan": {"dataflow": "weight_stationary"},
                            "operator_descriptors": [{"op_type": "gemm", "name": "gemm_main"}],
                            "dispatch_schedule": {
                                "total_issue_cycles": 2,
                                "barrier_count": 1,
                                "raw_hazard_count": 1,
                                "entries": [
                                    {"op_name": "gemm_main", "engine": "compute",
                                     "barrier_before": False, "depends_on": [],
                                     "issue_cycle": 0, "completion_cycle": 4, "stall_cycles": 0},
                                    {"op_name": "relu_post_gemm", "engine": "vector",
                                     "barrier_before": True, "depends_on": ["gemm_main"],
                                     "issue_cycle": 5, "completion_cycle": 6, "stall_cycles": 4},
                                ],
                            },
                        },
                    )
                if case_id == "lowered_program_ops":
                    compiled_program_json = output_dir / "compiled_program.json"
                    compiled_program_json.write_text("{}", encoding="utf-8")
                    return _make_fake_pipeline_result(
                        output_dir=output_dir,
                        requirement_text=requirement_text,
                        candidate_id="balanced",
                        generator_backend="heuristic",
                        requested_backend="heuristic",
                        supporting_files=[str(compiled_program_json)],
                        estimated_effective_tops=1.137778,
                        workload_type="dense_gemm",
                        architecture_family="tiled_systolic_array",
                        compiled_program_summary={
                            "tiling_strategy": "gemm_blocking",
                            "problem_shape": {"m": 16, "k": 32, "n": 16},
                            "mapping_plan": {"dataflow": "weight_stationary"},
                            "operator_descriptors": [{"op_type": "gemm", "name": "gemm_main"}],
                            "lowered_program": [
                                {"op_type": "gemm", "hardware_primitive": "systolic_gemm",
                                 "fusable_with_next": True, "name": "gemm_main",
                                 "inputs": ["gemm_main_activation_in", "gemm_main_weight_in"],
                                 "outputs": ["gemm_main_result_out"],
                                 "params": {"m": 16, "k": 32, "n": 16}},
                                {"op_type": "relu", "hardware_primitive": "vector_elementwise",
                                 "fusable_with_next": False, "name": "relu_post_gemm",
                                 "inputs": ["gemm_main_result_out"],
                                 "outputs": ["relu_post_gemm_out"],
                                 "params": {"shape": [16, 16]}},
                            ],
                        },
                    )
                if case_id == "memory_planner_gemm":
                    compiled_program_json = output_dir / "compiled_program.json"
                    compiled_program_json.write_text("{}", encoding="utf-8")
                    return _make_fake_pipeline_result(
                        output_dir=output_dir,
                        requirement_text=requirement_text,
                        candidate_id="balanced",
                        generator_backend="heuristic",
                        requested_backend="heuristic",
                        supporting_files=[str(compiled_program_json)],
                        estimated_effective_tops=1.137778,
                        workload_type="dense_gemm",
                        architecture_family="tiled_systolic_array",
                        compiled_program_summary={
                            "tiling_strategy": "gemm_blocking",
                            "problem_shape": {"m": 16, "k": 32, "n": 16},
                            "mapping_plan": {"dataflow": "weight_stationary"},
                            "operator_descriptors": [{"op_type": "gemm", "name": "gemm_main"}],
                            "memory_plan": {
                                "operator_count": 1,
                                "tensor_count": 3,
                                "peak_sram_bytes_no_reuse": 2048,
                                "peak_sram_bytes_with_reuse": 2048,
                                "reuse_savings_pct": 0.0,
                            },
                        },
                    )
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
                        estimated_effective_tops=1.137778,
                    )
                return _make_fake_pipeline_result(
                    output_dir=output_dir,
                    requirement_text=requirement_text,
                    candidate_id="balanced",
                    generator_backend="heuristic",
                    requested_backend="heuristic",
                    supporting_files=[],
                    estimated_effective_tops=5.688889,
                )

            with patch("create_npu.benchmark.CreateNPUPipeline") as pipeline_cls:
                pipeline_cls.return_value.run.side_effect = fake_run
                payload = run_regression_benchmark(
                    output_dir=temp_path / "benchmark",
                    require_full_toolchain=True,
                    llm_model="gpt-test",
                )

            self.assertTrue(payload["passed"])
            self.assertEqual(len(payload["cases"]), 9)
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
                    estimated_effective_tops=5.688889 if generator_backend == "heuristic" else 1.137778,
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
    workload_type: str = "transformer",
    architecture_family: str = "tiled_systolic_transformer",
    compiled_program_summary: Optional[Dict[str, object]] = None,
    extra_summary: Optional[Dict[str, object]] = None,
) -> PipelineResult:
    architecture = ArchitectureCandidate(
        candidate_id=candidate_id,
        family=architecture_family,
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
    for tool_name in (
        "python_reference",
        "reference_coverage",
        "verilator_lint",
        "iverilog_sim",
        "yosys_synth",
    ):
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
            workload_type=workload_type if requested_backend == "heuristic" else "dense_gemm",
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
                "top_level_case_count": 4,
                "total_cycles": 45,
                "busy_cycles": 37,
                "done_cycles": 4,
                "idle_cycles": 4,
                "memory_path": {
                    "dma_cycles": 11,
                    "load_cycles": 8,
                    "store_cycles": 7 if requested_backend == "heuristic" else 4,
                    "working_set_utilization": 0.5,
                    "total_dma_bits_transferred": 176,
                    "total_store_bits_transferred": 576,
                    "peak_external_bandwidth_gb_per_s": 16.0,
                },
                "compute_path": {
                    "compute_cycles": 5,
                    "flush_cycles": 4,
                    "clear_cycles": 1,
                    "estimated_mac_operations": 24,
                },
                "top_npu_throughput": {
                    "estimated_effective_tops": estimated_effective_tops,
                    "theoretical_peak_tops": 51.2,
                },
                "compiled_program": compiled_program_summary
                or {
                    "tiling_strategy": "sequence_blocking",
                    "problem_shape": {
                        "sequence_length": 4096,
                    },
                    "mapping_plan": {
                        "dataflow": "systolic",
                    },
                    "operator_descriptors": [
                        {
                            "op_type": "gemm",
                        }
                    ],
                },
                "workload_profile": {
                    "workload_type": workload_type if requested_backend == "heuristic" else "dense_gemm",
                },
                **(extra_summary or {}),
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
