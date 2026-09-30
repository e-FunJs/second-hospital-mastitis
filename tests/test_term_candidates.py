"""候选观察与聚合测试。

输入：人工构造的未决规范化结果与临时 observations JSONL。
输出：测试临时文件；验证追加日志、独立文献计数和冲突隔离，不触碰正式词典。
"""

from __future__ import annotations

from pathlib import Path

from rag_medical.terminology.candidates import (
    CandidateObservation,
    aggregate_observations,
    append_observations,
    observations_from_result,
    read_observations,
    write_candidates,
)
from rag_medical.terminology.normalizer import (
    NormalizationResult,
    UnresolvedSpan,
    load_normalizer_resources,
)


ROOT = Path(__file__).resolve().parents[1]


def observation(document_id: str, *, concept_id: str = "DIS_IGM") -> CandidateObservation:
    return CandidateObservation(
        observed="idiopathc granulomatous mastitis",
        compact_key="idiopathcgranulomatousmastitis",
        candidate="idiopathic granulomatous mastitis",
        concept_id=concept_id,
        language="en",
        category="disease",
        score=0.97,
        match_type="ambiguous_fuzzy_candidates",
        document_id=document_id,
        source_file=f"{document_id}.json",
        observed_at="2026-09-11T10:00:00+00:00",
    )


def test_builds_observation_without_writing_from_pure_result() -> None:
    dictionary, _, _ = load_normalizer_resources(ROOT / "configs/terminology.yaml")
    result = NormalizationResult(
        original_text="idiopathc granulomatous mastitis",
        normalized_text="idiopathc granulomatous mastitis",
        unresolved_spans=(
            UnresolvedSpan(
                text="idiopathc granulomatous mastitis",
                start=0,
                end=33,
                reason="ambiguous_fuzzy_candidates",
                candidates=("idiopathic granulomatous mastitis",),
                candidate_scores=(0.97,),
            ),
        ),
    )

    records = observations_from_result(
        result,
        dictionary=dictionary,
        document_id="DOC-1",
        source_file="evidence.json",
        observed_at="2026-09-11T10:00:00+00:00",
    )

    assert records[0].concept_id == "DIS_IGM"
    assert records[0].score == 0.97


def test_append_and_read_observations(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"

    assert append_observations(path, [observation("DOC-1")]) == 1
    assert append_observations(path, [observation("DOC-2")]) == 1
    loaded = read_observations(path)

    assert [item.document_id for item in loaded] == ["DOC-1", "DOC-2"]


def test_aggregation_counts_unique_documents_not_chunks() -> None:
    candidates = aggregate_observations(
        [observation("DOC-1"), observation("DOC-1"), observation("DOC-2")]
    )

    assert candidates[0]["occurrences"] == 3
    assert candidates[0]["unique_document_count"] == 2
    assert candidates[0]["document_ids"] == ["DOC-1", "DOC-2"]
    assert candidates[0]["status"] == "candidate"


def test_conflicting_concepts_are_isolated() -> None:
    candidates = aggregate_observations(
        [observation("DOC-1", concept_id="DIS_IGM"), observation("DOC-2", concept_id="DIS_GM")]
    )

    assert {item["status"] for item in candidates} == {"conflict"}


def test_write_candidates_uses_separate_data_file(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    records = aggregate_observations([observation("DOC-1")])

    assert write_candidates(path, records) == 1
    assert '"status": "candidate"' in path.read_text(encoding="utf-8")
