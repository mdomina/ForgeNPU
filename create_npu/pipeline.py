import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from create_npu.candidate_space import canonical_candidate_profile, candidate_variant_index
from create_npu.architect import generate_candidate_architectures
from create_npu.dataset import archive_pipeline_run, finalize_candidate_dataset_labels
from create_npu.environment import collect_environment_snapshot
from create_npu.generator_backend import prepare_backend_context
from create_npu.harness import VerificationHarness
from create_npu.learning_feedback import attach_learning_feedback
from create_npu.models import PipelineResult
from create_npu.requirement_parser import RequirementParser
from create_npu.reporting import generate_execution_report
from create_npu.rtl_generator import emit_seed_rtl
from create_npu.scorer import build_score_breakdown


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

            if bool(backend_context.get("compare_against_seed")):
                heuristic_result = self._evaluate_candidate_bundle(
                    spec=spec,
                    architecture=architecture,
                    output_dir=candidate_dir / "heuristic_seed",
                    generator_backend="heuristic",
                    extra_notes=[],
                    extra_supporting_files=[],
                    rtl_overrides={},
                )
                llm_result = self._evaluate_candidate_bundle(
                    spec=spec,
                    architecture=architecture,
                    output_dir=candidate_dir / "llm_variant",
                    generator_backend="llm",
                    extra_notes=list(backend_context["notes"]),
                    extra_supporting_files=list(backend_context["supporting_files"]),
                    rtl_overrides=dict(backend_context["rtl_overrides"]),
                )
                comparison = _build_backend_comparison(
                    candidate_id=architecture.candidate_id,
                    requested_backend=generator_backend,
                    heuristic_result=heuristic_result,
                    llm_result=llm_result,
                )
                comparison_path = candidate_dir / "backend_comparison.json"
                comparison_path.write_text(
                    json.dumps(comparison, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                selected_result = llm_result if bool(comparison["selected_llm_variant"]) else heuristic_result
                _extend_unique(
                    selected_result["generated"].supporting_files,
                    list(backend_context["supporting_files"]) + [str(comparison_path)],
                )
                selected_result["generated"].notes.append(str(comparison["selection_note"]))
                candidate_payload = _candidate_payload(
                    architecture=architecture,
                    evaluation=selected_result,
                )
                candidate_payload["backend_comparison"] = comparison
                candidate_results.append(candidate_payload)
            else:
                evaluation = self._evaluate_candidate_bundle(
                    spec=spec,
                    architecture=architecture,
                    output_dir=candidate_dir,
                    generator_backend=str(backend_context["effective_backend"]),
                    extra_notes=list(backend_context["notes"]),
                    extra_supporting_files=list(backend_context["supporting_files"]),
                    rtl_overrides=dict(backend_context["rtl_overrides"]),
                )
                candidate_results.append(
                    _candidate_payload(
                        architecture=architecture,
                        evaluation=evaluation,
                    )
                )

        best_candidate = max(candidate_results, key=lambda candidate: candidate["score"])
        finalize_candidate_dataset_labels(
            candidate_results=candidate_results,
            selected_candidate_id=str(best_candidate["candidate_id"]),
        )
        search_summary = _build_search_summary(
            candidate_results=candidate_results,
            requested_candidate_count=num_candidates,
            selected_candidate_id=str(best_candidate["candidate_id"]),
        )
        learning_feedback_summary = attach_learning_feedback(
            candidate_results=candidate_results,
            selected_candidate_id=str(best_candidate["candidate_id"]),
        )
        selected_architecture = best_candidate["architecture"]
        selected_generated = best_candidate["generated"]
        selected_tool_results = best_candidate["tool_results"]
        score = best_candidate["score"]
        score_breakdown = best_candidate.get("score_breakdown", {})

        self._write_candidate_index(run_dir=run_dir, candidate_results=candidate_results)

        result = PipelineResult(
            spec=spec,
            architecture=self._architecture_from_dict(selected_architecture),
            generated=self._generated_from_dict(selected_generated),
            tool_results=self._tool_results_from_dicts(selected_tool_results),
            score=score,
            output_dir=str(run_dir),
            score_breakdown=score_breakdown,
            report=best_candidate.get("report", {}),
            candidate_results=candidate_results,
            environment=environment,
            search=search_summary,
            learning_feedback=learning_feedback_summary,
        )
        result.dataset = archive_pipeline_run(
            base_output_dir=self.base_output_dir,
            run_dir=run_dir,
            result=result,
        )
        self._write_environment(run_dir=run_dir, environment=environment)
        self._write_result(run_dir=run_dir, result=result)
        return result

    def _evaluate_candidate_bundle(
        self,
        spec,
        architecture,
        output_dir: Path,
        generator_backend: str,
        extra_notes: list,
        extra_supporting_files: list,
        rtl_overrides: dict,
    ) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        generated = emit_seed_rtl(
            spec=spec,
            architecture=architecture,
            output_dir=output_dir,
            candidate_id=architecture.candidate_id,
            generator_backend=generator_backend,
            extra_notes=extra_notes,
            extra_supporting_files=extra_supporting_files,
            rtl_overrides=rtl_overrides,
        )
        report = generate_execution_report(
            bundle=generated,
            output_dir=output_dir,
            architecture=architecture,
            spec=spec,
        )
        generated.supporting_files.append(str(report["path"]))
        tool_results = VerificationHarness(output_dir).run(generated)
        score_breakdown = build_score_breakdown(
            spec=spec,
            architecture=architecture,
            tool_results=tool_results,
            report_summary=report["summary"],
            supporting_files=generated.supporting_files,
        )
        score = float(score_breakdown["total_score"])
        return {
            "generated": generated,
            "tool_results": tool_results,
            "report": report,
            "score": score,
            "score_breakdown": score_breakdown,
            "output_dir": str(output_dir),
        }

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


def _candidate_payload(architecture, evaluation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": architecture.candidate_id,
        "architecture": architecture.to_dict(),
        "generated": evaluation["generated"].to_dict(),
        "tool_results": [result.to_dict() for result in evaluation["tool_results"]],
        "report": evaluation["report"],
        "score": evaluation["score"],
        "score_breakdown": evaluation.get("score_breakdown", {}),
        "output_dir": evaluation["output_dir"],
    }


def _build_backend_comparison(
    candidate_id: str,
    requested_backend: str,
    heuristic_result: Dict[str, Any],
    llm_result: Dict[str, Any],
) -> Dict[str, Any]:
    heuristic_score = float(heuristic_result["score"])
    llm_score = float(llm_result["score"])
    selected_llm_variant = llm_score >= heuristic_score
    selection_note = (
        "Confronto backend completato: variante LLM selezionata "
        f"({llm_score:.2f} vs seed {heuristic_score:.2f})."
    )
    if not selected_llm_variant:
        selection_note = (
            "Confronto backend completato: seed euristico mantenuto "
            f"({heuristic_score:.2f} vs LLM {llm_score:.2f})."
        )
    return {
        "candidate_id": candidate_id,
        "requested_backend": requested_backend,
        "selected_llm_variant": selected_llm_variant,
        "selected_backend": "llm" if selected_llm_variant else "heuristic",
        "selection_note": selection_note,
        "score_delta_llm_minus_seed": round(llm_score - heuristic_score, 2),
        "heuristic_seed": _variant_summary(heuristic_result),
        "llm_variant": _variant_summary(llm_result),
    }


def _variant_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    tool_results = [tool.to_dict() for tool in result["tool_results"]]
    return {
        "generator_backend": result["generated"].generator_backend,
        "score": float(result["score"]),
        "output_dir": result["output_dir"],
        "report": result["report"],
        "tool_results": tool_results,
        "score_breakdown": result.get("score_breakdown", {}),
        "passed_tools": [
            tool["name"]
            for tool in tool_results
            if tool["available"] and tool["passed"] is True
        ],
        "failed_tools": [
            tool["name"]
            for tool in tool_results
            if tool["available"] and tool["passed"] is False
        ],
        "missing_tools": [
            tool["name"]
            for tool in tool_results
            if not tool["available"]
        ],
    }


def _extend_unique(target: list, extra_items: list) -> None:
    for item in extra_items:
        if item not in target:
            target.append(item)


def _build_search_summary(
    candidate_results: list,
    requested_candidate_count: int,
    selected_candidate_id: str,
) -> dict:
    ranked_candidates = sorted(
        candidate_results,
        key=lambda candidate: float(candidate["score"]),
        reverse=True,
    )
    selected_rank = next(
        index
        for index, candidate in enumerate(ranked_candidates, start=1)
        if str(candidate["candidate_id"]) == selected_candidate_id
    )
    return {
        "strategy": "deterministic_best_of_n",
        "requested_candidate_count": requested_candidate_count,
        "generated_candidate_count": len(candidate_results),
        "selected_candidate_id": selected_candidate_id,
        "selected_candidate_rank": selected_rank,
        "profile_counts": {
            "balanced": sum(
                1
                for candidate in candidate_results
                if canonical_candidate_profile(str(candidate["candidate_id"])) == "balanced"
            ),
            "throughput_max": sum(
                1
                for candidate in candidate_results
                if canonical_candidate_profile(str(candidate["candidate_id"])) == "throughput_max"
            ),
            "efficiency": sum(
                1
                for candidate in candidate_results
                if canonical_candidate_profile(str(candidate["candidate_id"])) == "efficiency"
            ),
        },
        "variant_candidate_count": sum(
            1
            for candidate in candidate_results
            if candidate_variant_index(str(candidate["candidate_id"])) > 0
        ),
        "ranked_candidates": [
            {
                "candidate_id": str(candidate["candidate_id"]),
                "score": float(candidate["score"]),
                "score_breakdown": candidate.get("score_breakdown", {}),
                "profile": canonical_candidate_profile(str(candidate["candidate_id"])),
                "variant_index": candidate_variant_index(str(candidate["candidate_id"])),
            }
            for candidate in ranked_candidates
        ],
    }
