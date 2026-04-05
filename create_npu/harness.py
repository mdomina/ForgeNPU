import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

from create_npu.golden_model import evaluate_reference_cases
from create_npu.models import GeneratedDesignBundle, ToolResult


class VerificationHarness:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.log_dir = output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def run(self, bundle: GeneratedDesignBundle) -> List[ToolResult]:
        results = [
            self._run_python_reference(bundle),
            self._run_verilator_lint(bundle),
            self._run_iverilog_sim(bundle),
            self._run_yosys_synth(bundle),
        ]
        return results

    def _run_python_reference(self, bundle: GeneratedDesignBundle) -> ToolResult:
        if not bundle.reference_cases_path:
            return ToolResult(
                name="python_reference",
                available=False,
                passed=None,
                return_code=None,
                summary="Manifest dei casi di riferimento assente.",
            )

        passed, summary = evaluate_reference_cases(bundle.reference_cases_path)
        log_path = self.log_dir / "python_reference.log"
        log_path.write_text(summary + "\n", encoding="utf-8")
        return ToolResult(
            name="python_reference",
            available=True,
            passed=passed,
            return_code=0 if passed else 1,
            summary=summary,
            log_path=str(log_path),
        )

    def _run_verilator_lint(self, bundle: GeneratedDesignBundle) -> ToolResult:
        executable = shutil.which("verilator")
        if not executable:
            return ToolResult(
                name="verilator_lint",
                available=False,
                passed=None,
                return_code=None,
                summary="verilator non installato; lint saltato.",
            )

        if not bundle.testbench_files:
            log_path = self.log_dir / "verilator_lint.log"
            command = [
                executable,
                "--lint-only",
                "--sv",
                "-Wno-fatal",
                "--top-module",
                bundle.primary_module,
            ] + _resolve_paths(bundle.rtl_files)
            return self._run_command("verilator_lint", command, log_path)

        aggregate_log = self.log_dir / "verilator_lint.log"
        per_testbench_results = []

        for testbench_path in bundle.testbench_files:
            lint_result = self._run_single_verilator_testbench(
                executable=executable,
                bundle=bundle,
                testbench_path=Path(testbench_path),
            )
            per_testbench_results.append(lint_result)

        aggregate_log.write_text(
            _format_verilator_summary(per_testbench_results) + "\n",
            encoding="utf-8",
        )

        failed_testbenches = [
            result for result in per_testbench_results if not result["lint"].passed
        ]
        if failed_testbenches:
            first_failure = failed_testbenches[0]
            return ToolResult(
                name="verilator_lint",
                available=True,
                passed=False,
                return_code=first_failure["lint"].return_code,
                summary=(
                    "Verilator lint fallito su "
                    f"{first_failure['top']}. Vedi log aggregato."
                ),
                log_path=str(aggregate_log),
            )

        return ToolResult(
            name="verilator_lint",
            available=True,
            passed=True,
            return_code=0,
            summary=f"Verilator valido su {len(per_testbench_results)} testbench.",
            log_path=str(aggregate_log),
        )

    def _run_iverilog_sim(self, bundle: GeneratedDesignBundle) -> ToolResult:
        executable = shutil.which("iverilog")
        if not executable:
            return ToolResult(
                name="iverilog_sim",
                available=False,
                passed=None,
                return_code=None,
                summary="iverilog non installato; simulazione saltata.",
            )

        aggregate_log = self.log_dir / "iverilog_sim.log"
        build_dir = self.output_dir / "iverilog_build"
        build_dir.mkdir(parents=True, exist_ok=True)
        per_testbench_results = []

        for testbench_path in bundle.testbench_files:
            testbench_result = self._run_single_iverilog_testbench(
                executable=executable,
                bundle=bundle,
                testbench_path=Path(testbench_path),
                build_dir=build_dir,
            )
            per_testbench_results.append(testbench_result)

        aggregate_log.write_text(
            _format_iverilog_summary(per_testbench_results) + "\n",
            encoding="utf-8",
        )

        failed_testbenches = [
            result for result in per_testbench_results if not result["compile"].passed or not result["sim"].passed
        ]
        if failed_testbenches:
            first_failure = failed_testbenches[0]
            return ToolResult(
                name="iverilog_sim",
                available=True,
                passed=False,
                return_code=_first_non_zero_exit_code(first_failure),
                summary=(
                    "Simulazione Icarus fallita su "
                    f"{first_failure['top']}. Vedi log aggregato."
                ),
                log_path=str(aggregate_log),
            )

        return ToolResult(
            name="iverilog_sim",
            available=True,
            passed=True,
            return_code=0,
            summary=f"Icarus valido su {len(per_testbench_results)} testbench.",
            log_path=str(aggregate_log),
        )

    def _run_yosys_synth(self, bundle: GeneratedDesignBundle) -> ToolResult:
        executable = shutil.which("yosys")
        if not executable:
            return ToolResult(
                name="yosys_synth",
                available=False,
                passed=None,
                return_code=None,
                summary="yosys non installato; sintesi saltata.",
            )

        yosys_script = self.output_dir / "run_yosys.ys"
        bounded_params = _bounded_synthesis_parameters(bundle)
        rtl_files = (
            self._prepare_bounded_yosys_rtl_files(
                rtl_files=bundle.rtl_files,
                bounded_params=bounded_params,
            )
            if bounded_params
            else bundle.rtl_files
        )
        script_lines = [
            "read_verilog -sv " + " ".join(_resolve_paths(rtl_files)),
        ]
        if bounded_params:
            script_lines.append(
                "chparam "
                + " ".join(
                    f"-set {name} {value}" for name, value in bounded_params.items()
                )
                + f" {bundle.primary_module}"
            )
        script_lines.extend(
            [
                f"synth -top {bundle.primary_module}",
                "stat",
            ]
        )
        script_body = "\n".join(script_lines)
        yosys_script.write_text(script_body + "\n", encoding="utf-8")

        log_path = self.log_dir / "yosys_synth.log"
        command = [executable, "-s", str(yosys_script.resolve())]
        return self._run_command("yosys_synth", command, log_path)

    def _prepare_bounded_yosys_rtl_files(
        self,
        rtl_files: List[str],
        bounded_params: Dict[str, int],
    ) -> List[str]:
        bounded_dir = self.output_dir / "yosys_rtl"
        bounded_dir.mkdir(parents=True, exist_ok=True)
        bounded_files = []

        for rtl_file in rtl_files:
            source_path = Path(rtl_file)
            bounded_path = bounded_dir / source_path.name
            source_text = source_path.read_text(encoding="utf-8")
            bounded_path.write_text(
                _rewrite_parameter_defaults(source_text, bounded_params),
                encoding="utf-8",
            )
            bounded_files.append(str(bounded_path))

        return bounded_files

    def _run_command(self, name: str, command: List[str], log_path: Path) -> ToolResult:
        completed = subprocess.run(
            command,
            cwd=self.output_dir.resolve(),
            capture_output=True,
            text=True,
            check=False,
        )
        combined_output = completed.stdout + completed.stderr
        log_path.write_text(combined_output, encoding="utf-8")

        passed = completed.returncode == 0
        summary = "OK" if passed else f"Comando fallito con exit code {completed.returncode}."
        return ToolResult(
            name=name,
            available=True,
            passed=passed,
            return_code=completed.returncode,
            summary=summary,
            log_path=str(log_path),
        )

    def _run_single_iverilog_testbench(
        self,
        executable: str,
        bundle: GeneratedDesignBundle,
        testbench_path: Path,
        build_dir: Path,
    ) -> Dict[str, object]:
        top_module = testbench_path.stem
        build_path = build_dir / f"{top_module}.out"
        compile_log = self.log_dir / f"{top_module}_iverilog_compile.log"
        sim_log = self.log_dir / f"{top_module}_iverilog_sim.log"
        compile_command = [
            executable,
            "-g2012",
            "-s",
            top_module,
            "-o",
            str(build_path.resolve()),
        ] + _resolve_paths(bundle.rtl_files + [str(testbench_path)])
        compile_result = self._run_command(
            name=f"iverilog_compile_{top_module}",
            command=compile_command,
            log_path=compile_log,
        )

        if not compile_result.passed:
            return {
                "top": top_module,
                "compile": compile_result,
                "sim": ToolResult(
                    name=f"iverilog_sim_{top_module}",
                    available=True,
                    passed=False,
                    return_code=compile_result.return_code,
                    summary="Simulazione non eseguita per errore di compilazione.",
                    log_path=str(sim_log),
                ),
            }

        sim_result = self._run_command(
            name=f"iverilog_sim_{top_module}",
            command=[str(build_path.resolve())],
            log_path=sim_log,
        )
        return {
            "top": top_module,
            "compile": compile_result,
            "sim": sim_result,
        }

    def _run_single_verilator_testbench(
        self,
        executable: str,
        bundle: GeneratedDesignBundle,
        testbench_path: Path,
    ) -> Dict[str, object]:
        top_module = testbench_path.stem
        log_path = self.log_dir / f"{top_module}_verilator_lint.log"
        command = [
            executable,
            "--lint-only",
            "--sv",
            "--timing",
            "-Wno-fatal",
            "--top-module",
            top_module,
        ] + _resolve_paths(bundle.rtl_files + [str(testbench_path)])
        lint_result = self._run_command(
            name=f"verilator_lint_{top_module}",
            command=command,
            log_path=log_path,
        )
        return {
            "top": top_module,
            "lint": lint_result,
        }


def _format_iverilog_summary(per_testbench_results: List[Dict[str, object]]) -> str:
    lines = []
    for result in per_testbench_results:
        compile_result = result["compile"]
        sim_result = result["sim"]
        lines.append(
            f"[{result['top']}] compile={compile_result.summary} sim={sim_result.summary}"
        )
        lines.append(f"  compile_log={compile_result.log_path}")
        lines.append(f"  sim_log={sim_result.log_path}")
    return "\n".join(lines)


def _format_verilator_summary(per_testbench_results: List[Dict[str, object]]) -> str:
    lines = []
    for result in per_testbench_results:
        lint_result = result["lint"]
        lines.append(f"[{result['top']}] lint={lint_result.summary}")
        lines.append(f"  lint_log={lint_result.log_path}")
    return "\n".join(lines)


def _first_non_zero_exit_code(result: Dict[str, object]) -> int:
    compile_result = result["compile"]
    sim_result = result["sim"]
    if compile_result.return_code not in (None, 0):
        return int(compile_result.return_code)
    if sim_result.return_code not in (None, 0):
        return int(sim_result.return_code)
    return 1


def _resolve_paths(paths: List[str]) -> List[str]:
    return [str(Path(path).resolve()) for path in paths]


def _bounded_synthesis_parameters(bundle: GeneratedDesignBundle) -> Dict[str, int]:
    if bundle.primary_module != "top_npu" or not bundle.reference_cases_path:
        return {}

    payload = json.loads(Path(bundle.reference_cases_path).read_text(encoding="utf-8"))
    top_level_cases = payload.get("top_npu", [])
    if not top_level_cases:
        return {}

    return {
        "ROWS": max(1, max(int(case.get("rows", 1)) for case in top_level_cases)),
        "COLS": max(1, max(int(case.get("cols", 1)) for case in top_level_cases)),
        "DEPTH": max(1, max(int(case.get("depth", 1)) for case in top_level_cases)),
        "TILE_COUNT": max(1, max(int(case.get("tile_count", 1)) for case in top_level_cases)),
    }


def _rewrite_parameter_defaults(source_text: str, bounded_params: Dict[str, int]) -> str:
    rewritten = source_text
    for param_name, param_value in bounded_params.items():
        rewritten = re.sub(
            rf"(parameter int {param_name} = )[^,\n]+",
            rf"\g<1>{int(param_value)}",
            rewritten,
        )
    return rewritten
