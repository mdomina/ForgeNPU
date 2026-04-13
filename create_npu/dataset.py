import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from create_npu.models import PipelineResult


def finalize_candidate_dataset_labels(
    candidate_results: List[Dict[str, Any]],
    selected_candidate_id: str,
) -> None:
    if not candidate_results:
        return

    for candidate in candidate_results:
        label, reasons = _classify_candidate(
            candidate=candidate,
            selected_candidate_id=selected_candidate_id,
        )
        candidate["dataset_label"] = label
        candidate["dataset_label_reasons"] = reasons


def archive_pipeline_run(
    base_output_dir: Path,
    run_dir: Path,
    result: PipelineResult,
) -> Dict[str, Any]:
    dataset_dir = base_output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    run_sample = _build_run_sample(run_dir=run_dir, result=result)
    candidate_samples = _build_candidate_samples(run_id=run_dir.name, result=result)

    run_sample_path = run_dir / "run_dataset_sample.json"
    candidate_samples_path = run_dir / "candidate_dataset_samples.json"
    run_sample_path.write_text(
        json.dumps(run_sample, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidate_samples_path.write_text(
        json.dumps(candidate_samples, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _append_jsonl(dataset_dir / "run_samples.jsonl", run_sample)
    _append_jsonl_many(dataset_dir / "candidate_samples.jsonl", candidate_samples)

    manifest = _update_manifest(
        dataset_dir=dataset_dir,
        run_sample_path=run_sample_path,
        candidate_samples_path=candidate_samples_path,
        candidate_samples=candidate_samples,
    )

    return {
        "dataset_dir": str(dataset_dir),
        "run_sample_path": str(run_sample_path),
        "candidate_samples_path": str(candidate_samples_path),
        "run_samples_jsonl": str(dataset_dir / "run_samples.jsonl"),
        "candidate_samples_jsonl": str(dataset_dir / "candidate_samples.jsonl"),
        "manifest_path": str(dataset_dir / "manifest.json"),
        "manifest_summary": manifest,
    }


def _classify_candidate(
    candidate: Dict[str, Any],
    selected_candidate_id: str,
) -> tuple[str, List[str]]:
    reasons: List[str] = []
    candidate_id = str(candidate["candidate_id"])
    tool_results = candidate.get("tool_results", [])

    python_reference_ok = False
    failing_tools = []
    missing_tools = []
    for tool_result in tool_results:
        tool_name = str(tool_result.get("name"))
        if tool_name == "python_reference" and tool_result.get("passed") is True:
            python_reference_ok = True
        if tool_result.get("available") and tool_result.get("passed") is False:
            failing_tools.append(tool_name)
        if not tool_result.get("available"):
            missing_tools.append(tool_name)

    if candidate_id == selected_candidate_id:
        reasons.append("selected_best")
    else:
        reasons.append("not_selected")
    if python_reference_ok:
        reasons.append("python_reference_passed")
    else:
        reasons.append("python_reference_missing_or_failed")
    if failing_tools:
        reasons.append("verification_failures:" + ",".join(sorted(failing_tools)))
    if missing_tools:
        reasons.append("incomplete_toolchain:" + ",".join(sorted(missing_tools)))

    is_good = (
        candidate_id == selected_candidate_id
        and python_reference_ok
        and not failing_tools
    )

    return ("good" if is_good else "bad"), reasons


def _build_run_sample(run_dir: Path, result: PipelineResult) -> Dict[str, Any]:
    candidate_labels = {
        str(candidate["candidate_id"]): str(candidate.get("dataset_label", "unlabeled"))
        for candidate in result.candidate_results
    }
    return {
        "sample_type": "pipeline_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "output_dir": str(run_dir),
        "requirement_text": result.spec.original_text,
        "selected_candidate_id": result.architecture.candidate_id,
        "selected_score": result.score,
        "selected_score_breakdown": result.score_breakdown,
        "candidate_count": len(result.candidate_results),
        "candidate_labels": candidate_labels,
        "spec": result.spec.to_dict(),
        "architecture": result.architecture.to_dict(),
        "report_summary": result.report.get("summary", {}),
        "environment": result.environment,
        "search": result.search,
        "learning_feedback_summary": result.learning_feedback,
        "artifacts": {
            "result_path": str(run_dir / "result.json"),
            "candidates_index_path": str(run_dir / "candidates.json"),
            "environment_path": str(run_dir / "environment.json"),
        },
    }


def _build_candidate_samples(run_id: str, result: PipelineResult) -> List[Dict[str, Any]]:
    samples = []
    for candidate in result.candidate_results:
        generated = candidate.get("generated", {})
        tool_results = candidate.get("tool_results", [])
        samples.append(
            {
                "sample_type": "candidate",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "candidate_id": candidate["candidate_id"],
                "selected": candidate["candidate_id"] == result.architecture.candidate_id,
                "label": candidate.get("dataset_label", "unlabeled"),
                "label_reasons": candidate.get("dataset_label_reasons", []),
                "score": candidate["score"],
                "score_breakdown": candidate.get("score_breakdown", {}),
                "spec": result.spec.to_dict(),
                "architecture": candidate.get("architecture", {}),
                "report_summary": candidate.get("report", {}).get("summary", {}),
                "tool_results": tool_results,
                "learning_feedback": candidate.get("learning_feedback", {}),
                "artifacts": {
                    "output_dir": candidate.get("output_dir"),
                    "rtl_files": generated.get("rtl_files", []),
                    "testbench_files": generated.get("testbench_files", []),
                    "supporting_files": generated.get("supporting_files", []),
                    "report_path": candidate.get("report", {}).get("path"),
                    "log_paths": [
                        tool_result.get("log_path")
                        for tool_result in tool_results
                        if tool_result.get("log_path")
                    ],
                },
            }
        )
    return samples


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _append_jsonl_many(path: Path, payloads: List[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _update_manifest(
    dataset_dir: Path,
    run_sample_path: Path,
    candidate_samples_path: Path,
    candidate_samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "run_sample_count": 0,
            "candidate_sample_count": 0,
            "good_candidate_count": 0,
            "bad_candidate_count": 0,
            "accepted_candidate_count": 0,
            "rejected_candidate_count": 0,
            "total_candidate_reward": 0.0,
        }

    manifest["run_sample_count"] = int(manifest.get("run_sample_count", 0)) + 1
    manifest["candidate_sample_count"] = int(manifest.get("candidate_sample_count", 0)) + len(
        candidate_samples
    )
    manifest["good_candidate_count"] = int(manifest.get("good_candidate_count", 0)) + sum(
        1 for sample in candidate_samples if sample.get("label") == "good"
    )
    manifest["bad_candidate_count"] = int(manifest.get("bad_candidate_count", 0)) + sum(
        1 for sample in candidate_samples if sample.get("label") == "bad"
    )
    manifest["accepted_candidate_count"] = int(manifest.get("accepted_candidate_count", 0)) + sum(
        1 for sample in candidate_samples if sample.get("learning_feedback", {}).get("accept")
    )
    manifest["rejected_candidate_count"] = int(manifest.get("rejected_candidate_count", 0)) + sum(
        1 for sample in candidate_samples if not sample.get("learning_feedback", {}).get("accept")
    )
    manifest["total_candidate_reward"] = round(
        float(manifest.get("total_candidate_reward", 0.0))
        + sum(
            float(sample.get("learning_feedback", {}).get("reward", 0.0))
            for sample in candidate_samples
        ),
        6,
    )
    if int(manifest["candidate_sample_count"]) > 0:
        manifest["average_candidate_reward"] = round(
            float(manifest["total_candidate_reward"]) / int(manifest["candidate_sample_count"]),
            6,
        )
    manifest["latest_run_sample_path"] = str(run_sample_path)
    manifest["latest_candidate_samples_path"] = str(candidate_samples_path)
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
