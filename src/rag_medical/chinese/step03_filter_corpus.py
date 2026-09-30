"""
用途：用可审计的中文医学规则筛选 CNKI 文献和语义 chunk。

默认输入：
    1. data/registry/chinese/processed/literature_registry.csv
    2. data/articles/processed/chinese/article_chunks.jsonl

默认输出：
    1. data/registry/chinese/filtered/
       literature_registry_{strict,review,excluded}.csv、view 文件和 filter_report.md。
    2. data/articles/processed/chinese/filtered/
       rag_chunks_{strict,review,excluded}.jsonl。

说明：
    - strict：明确属于人类非哺乳期乳腺炎，允许进入后续默认医学检索。
    - review：可能相关但需人工复核，本轮不进入 strict 知识库。
    - excluded：动物乳腺炎、纯哺乳期乳腺炎、纯肿瘤或无目标疾病证据。
    - 每条记录保存命中词、分数和理由；本模块不调用 LLM，也不生成 embedding。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DISEASE_TERMS: tuple[tuple[str, int], ...] = (
    ("非哺乳期乳腺炎", 10),
    ("非哺乳期乳腺炎症", 10),
    ("非产褥期乳腺炎", 10),
    ("肉芽肿性小叶性乳腺炎", 10),
    ("特发性肉芽肿性乳腺炎", 10),
    ("肉芽肿性乳腺炎", 9),
    ("浆细胞性乳腺炎", 9),
    ("浆细胞乳腺炎", 9),
    ("乳腺导管扩张症", 8),
    ("乳管扩张症", 8),
    ("导管周围乳腺炎", 9),
    ("乳管周围炎", 8),
    ("乳腺导管炎", 7),
    ("乳房结核", 9),
    ("结核性乳腺炎", 9),
    ("克氏棒状杆菌", 7),
    ("复发性乳腺炎", 6),
)

TREATMENT_TERMS: tuple[tuple[str, int], ...] = (
    ("药物治疗", 4),
    ("联合用药", 4),
    ("联合治疗", 3),
    ("三联疗法", 5),
    ("三联治疗", 5),
    ("抗结核治疗", 5),
    ("利福平", 5),
    ("异烟肼", 5),
    ("乙胺丁醇", 5),
    ("吡嗪酰胺", 4),
    ("甲氨蝶呤", 4),
    ("泼尼松", 4),
    ("强的松", 4),
    ("甲泼尼龙", 4),
    ("地塞米松", 4),
    ("糖皮质激素", 3),
    ("类固醇激素", 3),
    ("抗生素", 3),
    ("阿莫西林", 3),
    ("克拉霉素", 3),
    ("阿奇霉素", 3),
    ("左氧氟沙星", 3),
    ("庆大霉素", 3),
    ("治疗", 2),
    ("疗法", 2),
    ("疗效", 2),
    ("诊治", 2),
    ("疗程", 3),
    ("复发", 2),
    ("缓解", 2),
    ("随访", 1),
    ("预后", 1),
)

HUMAN_CONTEXT_TERMS: tuple[tuple[str, int], ...] = (
    ("患者", 3),
    ("女性", 2),
    ("病例", 2),
    ("临床", 2),
    ("例", 1),
    ("乳腺", 2),
    ("乳房", 2),
    ("病人", 2),
    ("住院", 1),
)

ANIMAL_TERMS: tuple[tuple[str, int], ...] = (
    ("奶牛乳腺炎", 12),
    ("牛乳腺炎", 12),
    ("奶牛", 10),
    ("乳牛", 10),
    ("奶山羊", 10),
    ("山羊乳腺炎", 12),
    ("绵羊乳腺炎", 12),
    ("羊乳腺炎", 11),
    ("兽医", 10),
    ("畜牧", 8),
    ("牧场", 7),
    ("奶牛场", 9),
    ("产奶量", 8),
    ("体细胞数", 8),
    ("乳头浸浴", 8),
    ("乳区", 5),
    ("泌乳牛", 9),
)

LACTATIONAL_TERMS: tuple[tuple[str, int], ...] = (
    ("哺乳期乳腺炎", 8),
    ("产褥期乳腺炎", 8),
    ("母乳喂养", 6),
    ("产后乳腺炎", 7),
    ("通乳", 5),
    ("积乳", 4),
)

OTHER_EXCLUDE_TERMS: tuple[tuple[str, int], ...] = (
    ("乳腺癌", 5),
    ("炎性乳腺癌", 7),
    ("乳腺肿瘤", 5),
    ("细胞株", 5),
    ("动物模型", 7),
)

STRONG_DISEASE_TERMS = {term for term, weight in DISEASE_TERMS if weight >= 8}


@dataclass(frozen=True)
class Decision:
    decision: str
    include_score: int
    exclude_score: int
    include_matches: list[str]
    exclude_matches: list[str]
    review_reason: str


# -----------------------------------------------------------------------------
# 规则计算
# -----------------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", "", text)


def matched_terms(
    text: str,
    weighted_terms: Iterable[tuple[str, int]],
) -> tuple[list[str], int]:
    matches: list[str] = []
    score = 0
    for term, weight in weighted_terms:
        if normalize_text(term) in text:
            matches.append(term)
            score += weight
    return matches, score


def classify_text(text: str) -> Decision:
    normalized = normalize_text(text)
    disease_matches, disease_score = matched_terms(normalized, DISEASE_TERMS)
    treatment_matches, treatment_score = matched_terms(normalized, TREATMENT_TERMS)
    human_matches, human_score = matched_terms(normalized, HUMAN_CONTEXT_TERMS)
    animal_matches, animal_score = matched_terms(normalized, ANIMAL_TERMS)
    lactational_matches, lactational_score = matched_terms(normalized, LACTATIONAL_TERMS)
    other_matches, other_score = matched_terms(normalized, OTHER_EXCLUDE_TERMS)

    include_matches = list(
        dict.fromkeys(disease_matches + treatment_matches + human_matches)
    )
    exclude_matches = list(
        dict.fromkeys(animal_matches + lactational_matches + other_matches)
    )
    include_score = disease_score + treatment_score + human_score
    exclude_score = animal_score + lactational_score + other_score
    strong_disease = any(term in STRONG_DISEASE_TERMS for term in disease_matches)

    if animal_score > 0:
        return Decision(
            "exclude",
            include_score,
            exclude_score,
            include_matches,
            exclude_matches,
            "动物或兽医乳腺炎证据明确",
        )
    if lactational_score >= 6 and not strong_disease:
        return Decision(
            "exclude",
            include_score,
            exclude_score,
            include_matches,
            exclude_matches,
            "仅见哺乳期或产褥期乳腺炎证据",
        )
    if other_score >= 7 and not strong_disease:
        return Decision(
            "exclude",
            include_score,
            exclude_score,
            include_matches,
            exclude_matches,
            "肿瘤或实验模型信号占主导",
        )
    if strong_disease and include_score >= exclude_score + 6:
        return Decision(
            "include",
            include_score,
            exclude_score,
            include_matches,
            exclude_matches,
            "",
        )
    if disease_score >= 6:
        return Decision(
            "review",
            include_score,
            exclude_score,
            include_matches,
            exclude_matches,
            "目标疾病相关，但题名或正文证据不足以自动进入 strict",
        )
    if treatment_score >= 3 and human_score >= 2 and "乳腺炎" in normalized:
        return Decision(
            "review",
            include_score,
            exclude_score,
            include_matches,
            exclude_matches,
            "人类乳腺炎治疗相关，但未明确非哺乳期亚型",
        )
    return Decision(
        "exclude",
        include_score,
        exclude_score,
        include_matches,
        exclude_matches,
        "没有明确的目标非哺乳期乳腺炎证据",
    )


def record_text(record: dict[str, Any]) -> str:
    return " ".join(
        str(record.get(field) or "")
        for field in ("title", "keywords", "abstract", "journal")
    )


def chunk_text(chunk: dict[str, Any]) -> str:
    return " ".join(
        str(chunk.get(field) or "") for field in ("section", "text")
    )


def classify_record(record: dict[str, Any]) -> Decision:
    return classify_text(record_text(record))


def classify_chunk(chunk: dict[str, Any]) -> Decision:
    return classify_text(chunk_text(chunk))


# -----------------------------------------------------------------------------
# registry 与 chunk 分层
# -----------------------------------------------------------------------------


def decision_fields(decision: Decision) -> dict[str, Any]:
    return {
        "filter_decision": decision.decision,
        "filter_include_score": decision.include_score,
        "filter_exclude_score": decision.exclude_score,
        "filter_include_matches": "; ".join(decision.include_matches),
        "filter_exclude_matches": "; ".join(decision.exclude_matches),
        "filter_review_reason": decision.review_reason,
    }


def split_registry(
    records: list[dict[str, str]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Decision],
]:
    strict: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    decisions: dict[str, Decision] = {}
    for record in records:
        decision = classify_record(record)
        annotated = dict(record)
        annotated.update(decision_fields(decision))
        target = {"include": strict, "review": review, "exclude": excluded}[decision.decision]
        target.append(annotated)
        document_id = str(record.get("document_id") or "")
        if document_id:
            decisions[document_id] = decision
    return strict, review, excluded, decisions


def annotate_chunk(
    chunk: dict[str, Any],
    decision: Decision,
    level: str,
) -> dict[str, Any]:
    annotated = dict(chunk)
    annotated.update(decision_fields(decision))
    annotated["filter_level"] = level
    return annotated


def filter_chunks(
    chunks: list[dict[str, Any]],
    article_decisions: dict[str, Decision],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    strict: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    for chunk in chunks:
        document_id = str(chunk.get("document_id") or "")
        article_decision = article_decisions.get(document_id)
        chunk_decision = classify_chunk(chunk)

        if article_decision is None:
            review.append(annotate_chunk(chunk, chunk_decision, "missing_registry"))
            continue
        if article_decision.decision == "exclude":
            excluded.append(annotate_chunk(chunk, article_decision, "article"))
            continue
        if article_decision.decision == "review":
            review.append(annotate_chunk(chunk, chunk_decision, "article_review"))
            continue

        # 对已明确属于目标疾病的文章，Methods/Results 不一定反复出现完整病名。
        # 因此保留其所有非动物、非纯哺乳期 chunk，而不是要求每段再次命中题名词。
        animal_matches, _ = matched_terms(normalize_text(chunk_text(chunk)), ANIMAL_TERMS)
        if animal_matches:
            review.append(
                annotate_chunk(
                    chunk,
                    Decision(
                        "review",
                        chunk_decision.include_score,
                        chunk_decision.exclude_score,
                        chunk_decision.include_matches,
                        animal_matches,
                        "目标文章内部出现动物语境，需人工复核",
                    ),
                    "chunk_animal_review",
                )
            )
            continue
        strict.append(annotate_chunk(chunk, article_decision, "article"))

    return strict, review, excluded


# -----------------------------------------------------------------------------
# 文件输出
# -----------------------------------------------------------------------------


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
    return records


def csv_cell(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        value = "; ".join(str(item) for item in value)
    return " ".join(str(value or "").split())


def write_csv(path: Path, records: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({field: csv_cell(record.get(field, "")) for field in fields})


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


VIEW_FIELDS = [
    "document_id",
    "title",
    "page_count",
    "native_pages",
    "ocr_pages",
    "filter_decision",
    "filter_include_score",
    "filter_exclude_score",
    "filter_include_matches",
    "filter_exclude_matches",
    "filter_review_reason",
    "source_path",
]


def write_views(
    view_dir: Path,
    groups: dict[str, list[dict[str, Any]]],
) -> None:
    for name, records in groups.items():
        write_csv(view_dir / f"literature_registry_{name}_view.csv", records, VIEW_FIELDS)
        path = view_dir / f"literature_registry_{name}_view.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=VIEW_FIELDS,
                extrasaction="ignore",
                delimiter="\t",
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {field: csv_cell(record.get(field, "")) for field in VIEW_FIELDS}
                )


def write_report(
    path: Path,
    registry_counts: Counter[str],
    chunk_counts: Counter[str],
    registry_total: int,
    chunk_total: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 中文医学语料严格筛选报告",
        "",
        f"- 创建时间：{datetime.now(timezone.utc).isoformat()}",
        "",
        "## 文献",
        f"- 总数：{registry_total}",
        f"- strict：{registry_counts.get('include', 0)}",
        f"- review：{registry_counts.get('review', 0)}",
        f"- excluded：{registry_counts.get('exclude', 0)}",
        "",
        "## Chunks",
        f"- 总数：{chunk_total}",
        f"- strict：{chunk_counts.get('include', 0)}",
        f"- review：{chunk_counts.get('review', 0)}",
        f"- excluded：{chunk_counts.get('exclude', 0)}",
        "",
        "## 规则说明",
        "",
        "- strict 仅包含明确的人类非哺乳期乳腺炎文献。",
        "- 纯哺乳期、动物、纯肿瘤和无目标病种文献不进入 strict。",
        "- review 保留弱相关或混合证据，等待医学人员抽查。",
        "- 本步骤没有调用 LLM，也没有生成知识库 embedding。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly filter Chinese CNKI mastitis literature and chunks."
    )
    parser.add_argument(
        "--registry-in",
        type=Path,
        default=Path("data/registry/chinese/processed/literature_registry.csv"),
    )
    parser.add_argument(
        "--chunks-in",
        type=Path,
        default=Path("data/articles/processed/chinese/article_chunks.jsonl"),
    )
    parser.add_argument(
        "--registry-out-dir",
        type=Path,
        default=Path("data/registry/chinese/filtered"),
    )
    parser.add_argument(
        "--chunks-out-dir",
        type=Path,
        default=Path("data/articles/processed/chinese/filtered"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.registry_in.exists():
        print(f"registry not found: {args.registry_in}", file=sys.stderr)
        return 2
    if not args.chunks_in.exists():
        print(f"chunks not found: {args.chunks_in}", file=sys.stderr)
        return 2

    registry = read_csv(args.registry_in)
    chunks = read_jsonl(args.chunks_in)
    strict_registry, review_registry, excluded_registry, decisions = split_registry(registry)
    strict_chunks, review_chunks, excluded_chunks = filter_chunks(chunks, decisions)

    filter_fields = [
        "filter_decision",
        "filter_include_score",
        "filter_exclude_score",
        "filter_include_matches",
        "filter_exclude_matches",
        "filter_review_reason",
    ]
    base_fields = list(registry[0].keys()) if registry else []
    groups = {
        "strict": strict_registry,
        "review": review_registry,
        "excluded": excluded_registry,
    }
    for name, records in groups.items():
        write_csv(
            args.registry_out_dir / f"literature_registry_{name}.csv",
            records,
            base_fields + filter_fields,
        )
    write_views(args.registry_out_dir / "view", groups)

    write_jsonl(args.chunks_out_dir / "rag_chunks_strict.jsonl", strict_chunks)
    write_jsonl(args.chunks_out_dir / "rag_chunks_review.jsonl", review_chunks)
    write_jsonl(args.chunks_out_dir / "rag_chunks_excluded.jsonl", excluded_chunks)
    write_report(
        args.registry_out_dir / "filter_report.md",
        Counter(
            include=len(strict_registry),
            review=len(review_registry),
            exclude=len(excluded_registry),
        ),
        Counter(
            include=len(strict_chunks),
            review=len(review_chunks),
            exclude=len(excluded_chunks),
        ),
        len(registry),
        len(chunks),
    )

    print(f"registry_strict={len(strict_registry)}")
    print(f"registry_review={len(review_registry)}")
    print(f"registry_excluded={len(excluded_registry)}")
    print(f"chunks_strict={len(strict_chunks)}")
    print(f"chunks_review={len(review_chunks)}")
    print(f"chunks_excluded={len(excluded_chunks)}")
    print(f"registry_out_dir={args.registry_out_dir}")
    print(f"chunks_out_dir={args.chunks_out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
