import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from create_npu.architect import generate_candidate_architectures
from create_npu.environment import collect_environment_snapshot
from create_npu.generator_backend import prepare_backend_context
from create_npu.harness import VerificationHarness
from create_npu.models import PipelineResult
from create_npu.requirement_parser import RequirementParser
from create_npu.reporting import generate_execution_report
from create_npu.rtl_generator import emit_seed_rtl
from create_npu.scorer import score_design


class CreateNPUPipeline:
    def __init__(self, base_output_dir: Optional[Path] = None):
        self.base_output_dir = Path(base_output_dir or "runs")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.parser = RequirementParser()

    def run(
        self,
        requirement_text: str,
        output_dir: Optional[Path] = None,
        num_candidates: int = 3,
        generator_backend: str = "heuristic",
        llm_model: Optional[str] = None,
    ) -> PipelineResult:
        spec = self.parser.parse(requirement_text)
        run_dir = output_dir or self._new_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        environment = collect_environment_snapshot(
            requested_backend=generator_backend,
            llm_model=llm_model,
        )

        candidate_results = []
        for architecture in generate_candidate_architectures(spec, max_candidates=num_candidates):
            candidate_dir = run_dir / "candidates" / architecture.candidate_id
            candidate_dir.mkdir(parents=True, exist_ok=True)
            backend_context = prepare_backend_context(
                spec=spec,
                architecture=architecture,
                output_dir=candidate_dir,
                requested_backend=generator_backend,
                llm_model=llm_model,
            )

            generated = emit_seed_rtl(
                spec=spec,
                architecture=architecture,
                output_dir=candidate_dir,
                candidate_id=architecture.candidate_id,
                generator_backend=str(backend_context["effective_backend"]),
                extra_notes=backend_context["notes"],
                extra_supporting_files=backend_context["supporting_files"],
            )
            report = generate_execution_report(
                bundle=generated,
                output_dir=candidate_dir,
                architecture=architecture,
            )
            generated.supporting_files.append(str(report["path"]))
            tool_results = VerificationHarness(candidate_dir).run(generated)
            score = score_design(spec=spec, architecture=architecture, tool_results=tool_results)
            candidate_results.append(
                {
                    "candidate_id": architecture.candidate_id,
                    "architecture": architecture.to_dict(),
                    "generated": generated.to_dict(),
                    "tool_results": [result.to_dict() for result in tool_results],
                    "report": report,
                    "score": score,
                    "output_dir": str(candidate_dir),
                }
            )

        best_candidate = max(candidate_results, key=lambda candidate: candidate["score"])
        selected_architecture = best_candidate["architecture"]
        selected_generated = best_candidate["generated"]
        selected_tool_results = best_candidate["tool_results"]
        score = best_candidate["score"]

        self._write_candidate_index(run_dir=run_dir, candidate_results=candidate_results)

        result = PipelineResult(
            spec=spec,
            architecture=self._architecture_from_dict(selected_architecture),
            generated=self._generated_from_dict(selected_generated),
            tool_results=self._tool_results_from_dicts(selected_tool_results),
            score=score,
            output_dir=str(run_dir),
            report=best_candidate.get("report", {}),
            candidate_results=candidate_results,
            environment=environment,
        )
        self._write_environment(run_dir=run_dir, environment=environment)
        self._write_result(run_dir=run_dir, result=result)
        return result

    def _new_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.base_output_dir / f"output_{timestamp}"

    def _write_result(self, run_dir: Path, result: PipelineResult) -> None:
        result_path = run_dir / "result.json"
        result_path.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_candidate_index(self, run_dir: Path, candidate_results: list) -> None:
        index_path = run_dir / "candidates.json"
        index_path.write_text(
            json.dumps(candidate_results, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _write_environment(self, run_dir: Path, environment: dict) -> None:
        environment_path = run_dir / "environment.json"
        environment_path.write_text(
            json.dumps(environment, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _architecture_from_dict(self, payload: dict):
        from create_npu.models import ArchitectureCandidate

        return ArchitectureCandidate(**payload)

    def _generated_from_dict(self, payload: dict):
        from create_npu.models import GeneratedDesignBundle

        return GeneratedDesignBundle(**payload)

    def _tool_results_from_dicts(self, payload: list):
        from create_npu.models import ToolResult

        return [ToolResult(**item) for item in payload]
