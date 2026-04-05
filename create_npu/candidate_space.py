import re
from typing import List


DEFAULT_CANDIDATE_PROFILES = ["balanced", "throughput_max", "efficiency"]
_CANDIDATE_PATTERN = re.compile(
    r"^(balanced|throughput_max|efficiency)(?:_b(\d+))?$"
)


def build_candidate_ids(max_candidates: int) -> List[str]:
    resolved_count = max(1, max_candidates)
    candidate_ids: List[str] = []
    for index in range(resolved_count):
        profile = DEFAULT_CANDIDATE_PROFILES[index % len(DEFAULT_CANDIDATE_PROFILES)]
        variant_index = index // len(DEFAULT_CANDIDATE_PROFILES)
        if variant_index == 0:
            candidate_ids.append(profile)
        else:
            candidate_ids.append(f"{profile}_b{variant_index}")
    return candidate_ids


def canonical_candidate_profile(candidate_id: str) -> str:
    match = _CANDIDATE_PATTERN.match(candidate_id)
    if not match:
        return candidate_id
    return str(match.group(1))


def candidate_variant_index(candidate_id: str) -> int:
    match = _CANDIDATE_PATTERN.match(candidate_id)
    if not match or match.group(2) is None:
        return 0
    return int(match.group(2))


def is_candidate_variant(candidate_id: str) -> bool:
    return candidate_variant_index(candidate_id) > 0
