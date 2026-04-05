from typing import Any, Dict, List


def attach_learning_feedback(
    candidate_results: List[Dict[str, Any]],
    selected_candidate_id: str,
) -> Dict[str, Any]:
    if not candidate_results:
        return {
            "accepted_candidate_count": 0,
            "rejected_candidate_count": 0,
            "average_reward": 0.0,
            "max_reward": 0.0,
            "min_reward": 0.0,
        }

    rewards: List[float] = []
    accepted_count = 0
    rejected_count = 0

    for candidate in candidate_results:
        feedback = _build_candidate_feedback(
            candidate=candidate,
            selected_candidate_id=selected_candidate_id,
        )
        candidate["learning_feedback"] = feedback
        rewards.append(float(feedback["reward"]))
        if feedback["accept"]:
            accepted_count += 1
        else:
            rejected_count += 1

    ranked_candidates = sorted(
        candidate_results,
        key=lambda candidate: float(candidate["learning_feedback"]["reward"]),
        reverse=True,
    )
    return {
        "accepted_candidate_count": accepted_count,
        "rejected_candidate_count": rejected_count,
        "average_reward": round(sum(rewards) / len(rewards), 6),
        "max_reward": round(max(rewards), 6),
        "min_reward": round(min(rewards), 6),
        "accepted_candidate_ids": [
            str(candidate["candidate_id"])
            for candidate in candidate_results
            if candidate["learning_feedback"]["accept"]
        ],
        "reward_ranking": [
            {
                "candidate_id": str(candidate["candidate_id"]),
                "reward": float(candidate["learning_feedback"]["reward"]),
                "accept": bool(candidate["learning_feedback"]["accept"]),
            }
            for candidate in ranked_candidates
        ],
        "selected_candidate_id": selected_candidate_id,
    }


def _build_candidate_feedback(
    candidate: Dict[str, Any],
    selected_candidate_id: str,
) -> Dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    score = float(candidate["score"])
    selected = candidate_id == selected_candidate_id
    dataset_label = str(candidate.get("dataset_label", "bad"))
    tool_results = candidate.get("tool_results", [])

    tool_pass_count = 0
    tool_fail_count = 0
    tool_missing_count = 0
    python_reference_passed = False
    failing_tool_names: List[str] = []
    missing_tool_names: List[str] = []

    for tool_result in tool_results:
        tool_name = str(tool_result.get("name"))
        if tool_name == "python_reference" and tool_result.get("passed") is True:
            python_reference_passed = True
        if tool_result.get("available") and tool_result.get("passed") is True:
            tool_pass_count += 1
        elif tool_result.get("available") and tool_result.get("passed") is False:
            tool_fail_count += 1
            failing_tool_names.append(tool_name)
        else:
            tool_missing_count += 1
            missing_tool_names.append(tool_name)

    accept = dataset_label == "good"
    reward = score
    reward += 10.0 if selected else 0.0
    reward += 10.0 if accept else -5.0
    reward += tool_pass_count * 3.0
    reward -= tool_fail_count * 12.0
    reward -= tool_missing_count * 2.0
    reward = round(reward, 6)

    normalized_reward = round(max(-1.0, min(1.0, (reward - 50.0) / 50.0)), 6)
    feedback_bucket = _feedback_bucket(
        accept=accept,
        python_reference_passed=python_reference_passed,
        tool_fail_count=tool_fail_count,
        tool_missing_count=tool_missing_count,
        selected=selected,
    )

    rejection_reasons = []
    if not accept:
        rejection_reasons.extend(candidate.get("dataset_label_reasons", []))
        if failing_tool_names:
            rejection_reasons.append("failing_tools:" + ",".join(sorted(failing_tool_names)))
        if missing_tool_names:
            rejection_reasons.append("missing_tools:" + ",".join(sorted(missing_tool_names)))
        rejection_reasons.append(feedback_bucket)

    return {
        "accept": accept,
        "reward": reward,
        "normalized_reward": normalized_reward,
        "feedback_bucket": feedback_bucket,
        "python_reference_passed": python_reference_passed,
        "tool_pass_count": tool_pass_count,
        "tool_fail_count": tool_fail_count,
        "tool_missing_count": tool_missing_count,
        "rejection_reasons": rejection_reasons,
    }


def _feedback_bucket(
    *,
    accept: bool,
    python_reference_passed: bool,
    tool_fail_count: int,
    tool_missing_count: int,
    selected: bool,
) -> str:
    if accept and selected:
        return "accept_selected_verified"
    if tool_fail_count > 0:
        return "reject_verification_failure"
    if not python_reference_passed:
        return "reject_reference_failure"
    if tool_missing_count > 0:
        return "reject_incomplete_toolchain"
    return "reject_ranked_out"
