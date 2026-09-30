"""离线审查并晋升术语候选为正式别名。

输入：``data/terminology/candidates.jsonl``、正式 canonical/alias 词典和配置文件。
输出：默认生成 ``promotion_proposal.jsonl`` 与审查报告，不改正式词典；只有显式
      ``--apply`` 才把通过安全校验的别名追加到 ``resources/terminology/aliases.jsonl``。

本工具仅能为已有 concept_id 增加 OCR/拼写别名，不能自动创造医学概念。候选必须先被
离线标为 ``approved``，当前问答请求不能自行触发晋升。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from rag_medical.terminology.dictionary import TermDictionary, compact_key
from rag_medical.terminology.normalizer import NormalizerSettings, load_normalizer_resources


DIGIT_OR_UNIT_RE = re.compile(
    r"\d|(?<![A-Za-z])(?:mg|g|kg|ml|l|mmol|mol|μg|ug|iu|u|qd|bid|tid|qid|q\d+h|%)(?![A-Za-z])",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(record), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _candidate_reason(
    record: Mapping[str, Any],
    *,
    dictionary: TermDictionary,
    settings: NormalizerSettings,
    minimum_unique_documents: int,
    minimum_score: float,
) -> tuple[str, dict[str, Any] | None]:
    if record.get("status") != "approved":
        return "not_approved", None
    if int(record.get("unique_document_count", 0)) < minimum_unique_documents:
        return "insufficient_independent_documents", None

    observed = str(record.get("observed") or "").strip()
    candidate = str(record.get("candidate") or "").strip()
    concept_id = str(record.get("concept_id") or "").strip()
    language = str(record.get("language") or "").strip()
    category = str(record.get("category") or "").strip()
    score = float(record.get("score", 0.0))
    if not all((observed, candidate, concept_id, language, category)):
        return "missing_required_fields", None
    if score < minimum_score:
        return "score_below_threshold", None
    if DIGIT_OR_UNIT_RE.search(observed) or DIGIT_OR_UNIT_RE.search(candidate):
        return "protected_number_or_unit", None

    canonical = dictionary.canonical_for(concept_id, language)
    if canonical is None:
        return "unknown_concept", None
    if canonical.category != category or canonical.canonical != candidate:
        return "concept_category_or_canonical_mismatch", None
    if len(compact_key(observed)) < settings.min_fuzzy_length:
        return "short_form_not_auto_promotable", None

    existing = dictionary.resolve_compact(observed, language)
    if existing is not None:
        if existing.concept_id == concept_id:
            return "already_known", None
        return "existing_concept_conflict", None
    conflict_keys = {(item.language, item.key) for item in dictionary.conflicts}
    if (language, compact_key(observed)) in conflict_keys:
        return "dictionary_key_conflict", None

    threshold = settings.threshold_for(len(compact_key(observed)))
    if threshold is None:
        return "no_fuzzy_threshold", None
    matches = dictionary.fuzzy_matches(
        observed,
        language,
        allowed_categories={category},
        max_distance=threshold.max_distance,
        min_similarity=threshold.min_similarity,
        limit=settings.max_fuzzy_candidates,
        shortlist_limit=settings.max_shortlist_size,
    )
    if not matches or matches[0].term.concept_id != concept_id:
        return "current_dictionary_does_not_confirm_candidate", None
    if len(matches) > 1 and matches[0].similarity - matches[1].similarity < settings.top_margin:
        return "ambiguous_current_candidates", None

    proposal = {
        "alias": observed,
        "concept_id": concept_id,
        "language": language,
        "category": category,
        "source": "promoted_candidate",
        "confidence": round(min(score, matches[0].similarity), 6),
        "auto_replace": True,
    }
    return "eligible", proposal


def build_promotion_proposal(
    records: Iterable[Mapping[str, Any]],
    *,
    dictionary: TermDictionary,
    settings: NormalizerSettings,
    minimum_unique_documents: int,
    minimum_score: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """返回可晋升别名和逐条拒绝原因；不执行文件写入。"""

    proposals: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    seen_aliases: set[tuple[str, str]] = set()
    for record in records:
        reason, proposal = _candidate_reason(
            record,
            dictionary=dictionary,
            settings=settings,
            minimum_unique_documents=minimum_unique_documents,
            minimum_score=minimum_score,
        )
        decisions.append(
            {
                "observed": record.get("observed", ""),
                "concept_id": record.get("concept_id", ""),
                "decision": reason,
            }
        )
        if proposal is None:
            continue
        identity = (proposal["language"], compact_key(proposal["alias"]))
        if identity not in seen_aliases:
            proposals.append(proposal)
            seen_aliases.add(identity)
    return proposals, decisions


def apply_aliases(alias_path: Path, proposals: Iterable[Mapping[str, Any]]) -> int:
    """显式应用提案，使用临时文件原子替换；Git 负责版本审查和回滚。"""

    existing = read_jsonl(alias_path)
    identities = {
        (str(item.get("language", "")), compact_key(str(item.get("alias", ""))))
        for item in existing
    }
    added = 0
    for proposal in proposals:
        identity = (
            str(proposal.get("language", "")),
            compact_key(str(proposal.get("alias", ""))),
        )
        if identity in identities:
            continue
        existing.append(dict(proposal))
        identities.add(identity)
        added += 1
    temporary = alias_path.with_suffix(alias_path.suffix + ".tmp")
    write_jsonl(temporary, existing)
    temporary.replace(alias_path)
    return added


def _resolve_project_path(value: object, config_path: Path, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing config field: {field_name}")
    path = Path(value)
    return path if path.is_absolute() else config_path.resolve().parent.parent / path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propose or apply approved terminology aliases.")
    parser.add_argument("--config", type=Path, default=Path("configs/terminology.yaml"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        dictionary, settings, config = load_normalizer_resources(args.config)
        observations_config = config.get("observations", {})
        dictionary_config = config.get("dictionary", {})
        input_path = args.input or _resolve_project_path(
            observations_config.get("candidates_path"), args.config, "candidates_path"
        )
        data_root = input_path.parent
        proposal_path = args.proposal or data_root / "promotion_proposal.jsonl"
        report_path = args.report or data_root / "promotion_report.json"
        minimum_documents = int(observations_config.get("minimum_unique_documents", 3))
        minimum_score = float(observations_config.get("minimum_score", 0.90))
        records = read_jsonl(input_path)
        proposals, decisions = build_promotion_proposal(
            records,
            dictionary=dictionary,
            settings=settings,
            minimum_unique_documents=minimum_documents,
            minimum_score=minimum_score,
        )
        write_jsonl(proposal_path, proposals)
        applied = 0
        if args.apply:
            alias_path = _resolve_project_path(
                dictionary_config.get("aliases_path"), args.config, "aliases_path"
            )
            applied = apply_aliases(alias_path, proposals)
        report = {
            "input": str(input_path),
            "proposal": str(proposal_path),
            "candidate_count": len(records),
            "eligible_count": len(proposals),
            "applied_count": applied,
            "apply_requested": bool(args.apply),
            "decisions": decisions,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        f"candidates={len(records)} eligible={len(proposals)} applied={applied} "
        f"proposal={proposal_path} report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
