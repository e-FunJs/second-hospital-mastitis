"""
用途：验证 step08 在保持 evidence 与 trace 不变时安全接入 query plan 输出。
输入：测试内构造的固定证据、临时配置和 monkeypatch 故障。
输出：pytest 对正常写入、三类失败隔离及临时目录清理的断言结果。
不做什么：不加载 BGE、FAISS、Qwen，不运行真实索引或查询翻译。
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import pytest

from rag_medical.common import step08_rag_answer as step08


class FixedDateTime(datetime):
    """固定 evidence 的 created_at，避免时间变化掩盖输出回归。"""

    @classmethod
    def now(cls, tz=None):  # type: ignore[no-untyped-def]
        return cls(2026, 9, 30, 12, 0, 0, tzinfo=tz)


def _write_config(project_root: Path) -> None:
    config_dir = project_root / "configs"
    aliases_dir = project_root / "resources" / "evidence_expansion"
    config_dir.mkdir(parents=True)
    aliases_dir.mkdir(parents=True)
    (config_dir / "evidence_expansion.yaml").write_text(
        """query_expansion:
  aliases_path: resources/evidence_expansion/query_aliases.jsonl
  max_queries: 12
  min_combination_drugs: 3
  max_combination_drugs: 4
""",
        encoding="utf-8",
    )
    alias_records = [
        {
            "concept_id": "DIS_GLM",
            "category": "disease",
            "language": "zh",
            "term": "肉芽肿性小叶性乳腺炎",
            "term_kind": "canonical",
            "source": "trusted_manual",
        },
        {
            "concept_id": "DIS_GLM",
            "category": "disease",
            "language": "en",
            "term": "granulomatous lobular mastitis",
            "term_kind": "canonical",
            "source": "trusted_manual",
        },
        {
            "concept_id": "DIS_GLM",
            "category": "disease",
            "language": "en",
            "term": "GLM",
            "term_kind": "abbreviation",
            "source": "trusted_manual",
        },
    ]
    (aliases_dir / "query_aliases.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in alias_records),
        encoding="utf-8",
    )


def _write_existing_outputs(
    monkeypatch: pytest.MonkeyPatch,
    output_dir: Path,
) -> tuple[Path, Path, bytes, bytes]:
    monkeypatch.setattr(step08, "datetime", FixedDateTime)
    evidence_records = step08.make_evidence_records(
        [
            {
                "rank": 1,
                "score": 0.91,
                "chunk_id": "CNKI-001::001",
                "language": "zh",
                "text": "固定证据。",
            }
        ],
        "GLM有哪些临床表现？",
    )
    step08.write_rag_package(
        output_dir,
        "GLM有哪些临床表现？",
        evidence_records,
        "glm_features",
    )
    evidence_path = output_dir / "glm_features_evidence.json"
    trace_path = output_dir / "glm_features_retrieval_trace.jsonl"
    return evidence_path, trace_path, evidence_path.read_bytes(), trace_path.read_bytes()


def _assert_existing_outputs_unchanged(
    evidence_path: Path,
    trace_path: Path,
    evidence_before: bytes,
    trace_before: bytes,
) -> None:
    assert evidence_path.read_bytes() == evidence_before
    assert trace_path.read_bytes() == trace_before


def _assert_no_temporary_directory(output_dir: Path) -> None:
    assert not list(output_dir.glob(".query_plan_tmp_*"))


def test_normal_path_writes_valid_query_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "outputs"
    baseline = _write_existing_outputs(monkeypatch, output_dir)

    plan_path = step08._try_write_query_plan(
        "GLM有哪些临床表现？",
        output_dir,
        "glm_features",
    )
    payload = json.loads(plan_path.read_text(encoding="utf-8"))

    assert plan_path == output_dir / "glm_features_query_plan.json"
    assert payload["schema_version"] == "1.0"
    assert payload["query_record"]["query_id"] == (
        "Q-" + hashlib.sha256("GLM有哪些临床表现？".encode("utf-8")).hexdigest()
    )
    assert payload["retrieval_queries"]
    assert [item["query_index"] for item in payload["retrieval_queries"]] == list(
        range(len(payload["retrieval_queries"]))
    )
    _assert_existing_outputs_unchanged(*baseline)
    _assert_no_temporary_directory(output_dir)


def test_alias_loading_failure_preserves_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "outputs"
    baseline = _write_existing_outputs(monkeypatch, output_dir)
    monkeypatch.setattr(step08, "load_aliases", lambda _path: (_ for _ in ()).throw(ValueError("alias failure")))
    caplog.set_level(logging.WARNING)

    result = step08._try_write_query_plan("GLM有哪些临床表现？", output_dir, "glm_features")

    assert result is None
    assert not (output_dir / "glm_features_query_plan.json").exists()
    assert "query plan generation failed" in caplog.text
    _assert_existing_outputs_unchanged(*baseline)


def test_build_failure_preserves_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "outputs"
    baseline = _write_existing_outputs(monkeypatch, output_dir)
    monkeypatch.setattr(step08, "load_aliases", lambda _path: {})
    monkeypatch.setattr(step08, "build_query_plan", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("build failure")))
    caplog.set_level(logging.WARNING)

    result = step08._try_write_query_plan("GLM有哪些临床表现？", output_dir, "glm_features")

    assert result is None
    assert not (output_dir / "glm_features_query_plan.json").exists()
    assert "query plan generation failed" in caplog.text
    _assert_existing_outputs_unchanged(*baseline)


def test_write_failure_removes_partial_file_and_temporary_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _write_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    output_dir = tmp_path / "outputs"
    baseline = _write_existing_outputs(monkeypatch, output_dir)
    monkeypatch.setattr(step08, "load_aliases", lambda _path: {})
    monkeypatch.setattr(step08, "build_query_plan", lambda *_args, **_kwargs: [])

    def fail_after_partial_write(_record, _queries, temporary_dir):  # type: ignore[no-untyped-def]
        (temporary_dir / "partial_query_plan.json").write_text("{", encoding="utf-8")
        raise OSError("write failure")

    monkeypatch.setattr(step08, "write_query_plan", fail_after_partial_write)
    caplog.set_level(logging.WARNING)

    result = step08._try_write_query_plan("GLM有哪些临床表现？", output_dir, "glm_features")

    assert result is None
    assert not (output_dir / "glm_features_query_plan.json").exists()
    assert "query plan generation failed" in caplog.text
    _assert_existing_outputs_unchanged(*baseline)
    _assert_no_temporary_directory(output_dir)
