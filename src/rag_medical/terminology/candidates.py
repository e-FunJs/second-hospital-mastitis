"""记录并聚合未自动修复的医学术语候选。

输入：``NormalizationResult`` 中的 unresolved_spans，或既有 observations JSONL。
输出：追加式 ``observations.jsonl`` 事件，以及按 compact_key/候选概念聚合的
      ``candidates.jsonl`` 记录。该模块绝不读取候选来修改正文，也不改正式词典。

同一文献中的多个 chunk 只计一个独立来源，避免重复切块虚增候选可信度。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from rag_medical.terminology.dictionary import TermDictionary, compact_key
from rag_medical.terminology.normalizer import NormalizationResult


@dataclass(frozen=True)
class CandidateObservation:
    observed: str
    compact_key: str
    candidate: str
    concept_id: str
    language: str
    category: str
    score: float
    match_type: str
    document_id: str
    source_file: str
    observed_at: str


def observations_from_result(
    result: NormalizationResult,
    *,
    dictionary: TermDictionary,
    document_id: str,
    source_file: str,
    language: str = "en",
    observed_at: str | None = None,
) -> list[CandidateObservation]:
    """把纯规范化结果转换为日志事件；此函数本身不写文件。"""

    timestamp = observed_at or datetime.now(timezone.utc).isoformat()
    observations: list[CandidateObservation] = []
    for span in result.unresolved_spans:
        candidate = span.candidates[0] if span.candidates else ""
        score = span.candidate_scores[0] if span.candidate_scores else 0.0
        resolved = dictionary.resolve_compact(candidate, language) if candidate else None
        observations.append(
            CandidateObservation(
                observed=span.text,
                compact_key=compact_key(span.text),
                candidate=candidate,
                concept_id=resolved.concept_id if resolved else "",
                language=language,
                category=resolved.category if resolved else "",
                score=round(score, 6),
                match_type=span.reason,
                document_id=document_id,
                source_file=source_file,
                observed_at=timestamp,
            )
        )
    return observations


def append_observations(path: Path, observations: Iterable[CandidateObservation]) -> int:
    """追加日志并返回写入条数；正式词典路径不应传入本函数。"""

    records = list(observations)
    if not records:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return len(records)


def read_observations(path: Path) -> list[CandidateObservation]:
    if not path.exists():
        return []
    observations: list[CandidateObservation] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                observations.append(CandidateObservation(**payload))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"invalid observation at {path}:{line_number}: {exc}") from exc
    return observations


def aggregate_observations(
    observations: Iterable[CandidateObservation],
) -> list[dict[str, object]]:
    """聚合候选，冲突映射标记为 conflict，不允许后续静默晋升。"""

    groups: dict[tuple[str, str, str, str], list[CandidateObservation]] = defaultdict(list)
    concepts_by_observed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in observations:
        identity = (item.language, item.compact_key, item.candidate, item.concept_id)
        groups[identity].append(item)
        if item.concept_id:
            concepts_by_observed[(item.language, item.compact_key)].add(item.concept_id)

    candidates: list[dict[str, object]] = []
    for identity, items in sorted(groups.items()):
        language, key, candidate, concept_id = identity
        timestamps = sorted(item.observed_at for item in items)
        document_ids = sorted({item.document_id for item in items if item.document_id})
        source_files = sorted({item.source_file for item in items if item.source_file})
        conflict = len(concepts_by_observed[(language, key)]) > 1
        candidates.append(
            {
                "observed": items[0].observed,
                "compact_key": key,
                "candidate": candidate,
                "concept_id": concept_id,
                "language": language,
                "category": items[0].category,
                "score": round(max(item.score for item in items), 6),
                "match_type": items[0].match_type,
                "occurrences": len(items),
                "unique_document_count": len(document_ids),
                "document_ids": document_ids,
                "first_seen": timestamps[0] if timestamps else "",
                "last_seen": timestamps[-1] if timestamps else "",
                "source_files": source_files,
                "status": "conflict" if conflict else "candidate",
            }
        )
    return candidates


def write_candidates(path: Path, candidates: Iterable[dict[str, object]]) -> int:
    """覆盖写入离线聚合结果；该文件不是运行时正式词典。"""

    records = list(candidates)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return len(records)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate terminology observations by independent source document."
    )
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        observations = read_observations(args.observations)
        candidates = aggregate_observations(observations)
        count = write_candidates(args.output, candidates)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"observations={len(observations)} candidates={count} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
