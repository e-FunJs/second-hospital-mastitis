"""离线候选晋升测试。

输入：临时正式词典、配置和 candidates JSONL。
输出：临时提案/报告/别名文件；验证默认不改词典，只有 ``--apply`` 才显式晋升。
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from rag_medical.terminology.dictionary import load_dictionary
from rag_medical.terminology.normalizer import NormalizerSettings
from rag_medical.terminology.promote_candidates import (
    build_promotion_proposal,
    main,
    read_jsonl,
)


def candidate(**updates):
    record = {
        "observed": "pyrazinamid",
        "compact_key": "pyrazinamid",
        "candidate": "pyrazinamide",
        "concept_id": "DRUG_PYRAZINAMIDE",
        "language": "en",
        "category": "drug",
        "score": 0.92,
        "match_type": "ambiguous_fuzzy_candidates",
        "occurrences": 4,
        "unique_document_count": 3,
        "document_ids": ["D1", "D2", "D3"],
        "source_files": ["a.json", "b.json", "c.json"],
        "status": "approved",
    }
    record.update(updates)
    return record


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )


def make_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    resources = tmp_path / "resources/terminology"
    data = tmp_path / "data/terminology"
    configs = tmp_path / "configs"
    resources.mkdir(parents=True)
    data.mkdir(parents=True)
    configs.mkdir(parents=True)
    canonical_path = resources / "canonical_terms.jsonl"
    aliases_path = resources / "aliases.jsonl"
    candidates_path = data / "candidates.jsonl"
    write_jsonl(
        canonical_path,
        [
            {
                "concept_id": "DRUG_PYRAZINAMIDE",
                "canonical": "pyrazinamide",
                "language": "en",
                "category": "drug",
                "source": "test",
                "confidence": 1.0,
            }
        ],
    )
    write_jsonl(aliases_path, [])
    write_jsonl(candidates_path, [candidate()])
    config_path = configs / "terminology.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dictionary": {
                    "canonical_path": "resources/terminology/canonical_terms.jsonl",
                    "aliases_path": "resources/terminology/aliases.jsonl",
                    "ngram_size": 3,
                },
                "normalization": {
                    "min_fuzzy_length": 6,
                    "top_margin": 0.05,
                    "fuzzy_thresholds": [
                        {
                            "min_length": 9,
                            "max_length": 15,
                            "max_distance": 1,
                            "min_similarity": 0.88,
                        }
                    ],
                },
                "observations": {
                    "candidates_path": "data/terminology/candidates.jsonl",
                    "minimum_unique_documents": 3,
                    "minimum_score": 0.90,
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return config_path, aliases_path, candidates_path


def test_proposal_requires_approval_documents_score_and_safe_form(tmp_path: Path) -> None:
    config_path, aliases_path, _ = make_project(tmp_path)
    dictionary = load_dictionary(
        config_path.parent.parent / "resources/terminology/canonical_terms.jsonl",
        aliases_path,
    )
    records = [
        candidate(),
        candidate(observed="pyrazinami", status="candidate"),
        candidate(observed="pyrazinamd", unique_document_count=2),
        candidate(observed="pyrazinarnide", score=0.80),
        candidate(observed="pyrazinamid5mg"),
    ]

    proposals, decisions = build_promotion_proposal(
        records,
        dictionary=dictionary,
        settings=NormalizerSettings(),
        minimum_unique_documents=3,
        minimum_score=0.90,
    )

    assert [item["alias"] for item in proposals] == ["pyrazinamid"]
    assert {item["decision"] for item in decisions} >= {
        "eligible",
        "not_approved",
        "insufficient_independent_documents",
        "score_below_threshold",
        "protected_number_or_unit",
    }


def test_cli_defaults_to_proposal_only_then_requires_apply(tmp_path: Path) -> None:
    config_path, aliases_path, _ = make_project(tmp_path)
    proposal_path = tmp_path / "proposal.jsonl"
    report_path = tmp_path / "report.json"
    original_aliases = aliases_path.read_bytes()

    first_status = main(
        [
            "--config",
            str(config_path),
            "--proposal",
            str(proposal_path),
            "--report",
            str(report_path),
        ]
    )

    assert first_status == 0
    assert aliases_path.read_bytes() == original_aliases
    assert read_jsonl(proposal_path)[0]["alias"] == "pyrazinamid"

    apply_status = main(
        [
            "--config",
            str(config_path),
            "--proposal",
            str(proposal_path),
            "--report",
            str(report_path),
            "--apply",
        ]
    )

    assert apply_status == 0
    assert read_jsonl(aliases_path)[0]["alias"] == "pyrazinamid"
