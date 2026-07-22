"""
用途：在第一轮严格医学筛选之后，用本地 BGE 模型复核动物乳腺炎污染。

运行顺序：
    1. ``bash scripts/filter_corpus.sh``
    2. ``bash scripts/animal_filter.sh``
    3. ``bash scripts/build_strict_index.sh``

默认输入：
    - data/registry/filtered/literature_registry_{strict,review,excluded}.csv
    - data/articles/processed/rag_chunks_{strict,review,excluded}.jsonl
    - data/articles/processed/rag_chunks.jsonl（用于浏览候选文章的全部可用 chunk）
    - models/bge/bge-m3

默认输出：
    - data/registry/filtered/semantic/literature_registry_{strict,review,excluded}.csv
    - data/registry/filtered/semantic/animal_audit.csv
    - data/registry/filtered/semantic/anchors.jsonl
    - data/registry/filtered/semantic/filter_report.md
    - data/articles/processed/semantic/rag_chunks_{strict,review,excluded}.jsonl

安全说明：本脚本不覆盖第一轮筛选结果。每篇文献的 BGE 分数、动物 chunk 比例、
锚点来源和最终去向均会保存，便于医学人员抽查后再调整阈值。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from rag_medical.animal_rules import (
    Anchor,
    ArticleDecision,
    SemanticConfig,
    build_chunk_scores,
    decide_article,
    extract_anchors,
    top_k_mean_similarity,
)
from rag_medical.build_embeddings import encode_texts, load_sentence_transformer, resolve_device


# -----------------------------------------------------------------------------
# 文件读取与稳定标识
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
                raise ValueError(f"invalid JSON in {path} line {line_number}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def csv_cell(value: Any) -> str:
    """压平换行并将列表转为分号文本，保证 VS Code/Excel 可稳定打开。"""

    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        value = "; ".join(str(item) for item in value)
    return " ".join(str(value).split())


def write_csv(path: Path, records: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # UTF-8 BOM 可让 Windows Excel 和常见 VS Code CSV 插件正确识别中文。
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({field: csv_cell(record.get(field, "")) for field in fieldnames})


def normalized_id(value: Any) -> str:
    return str(value or "").strip().lower()


def record_ids(record: dict[str, Any]) -> tuple[str, ...]:
    ids = []
    for field in ("pmcid", "pmid", "doi"):
        value = normalized_id(record.get(field))
        if value:
            ids.append(f"{field}:{value}")
    return tuple(ids)


def primary_id(record: dict[str, Any], fallback: str) -> str:
    ids = record_ids(record)
    return ids[0] if ids else fallback


def chunk_identity(chunk: dict[str, Any], fallback_index: int) -> str:
    value = str(chunk.get("chunk_id") or "").strip()
    return value or f"row:{fallback_index}"


# -----------------------------------------------------------------------------
# 候选文章全文组织
# -----------------------------------------------------------------------------


def index_chunks_by_article(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """为每个 PMCID/PMID/DOI 建立 chunk 索引，支持 registry 与全文互相匹配。"""

    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for position, chunk in enumerate(chunks):
        identity = chunk_identity(chunk, position)
        for item_id in record_ids(chunk):
            if identity not in seen[item_id]:
                index[item_id].append(chunk)
                seen[item_id].add(identity)
    return dict(index)


def chunks_for_record(
    record: dict[str, Any],
    chunk_index: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """合并一个 registry 记录通过多个标识找到的 chunk，并按 chunk_id 去重。"""

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item_id in record_ids(record):
        for position, chunk in enumerate(chunk_index.get(item_id, [])):
            identity = chunk_identity(chunk, position)
            if identity in seen:
                continue
            seen.add(identity)
            result.append(chunk)
    return result


def fallback_abstract_chunk(record: dict[str, Any]) -> dict[str, Any] | None:
    """无 PMC 全文时使用 PubMed 摘要；没有摘要则返回 None 并在审计表标记。"""

    abstract = " ".join(str(record.get("abstract") or "").split())
    if not abstract:
        return None
    return {
        "chunk_id": f"{primary_id(record, 'unknown')}::registry_abstract",
        "pmcid": record.get("pmcid", ""),
        "pmid": record.get("pmid", ""),
        "doi": record.get("doi", ""),
        "title": record.get("title", ""),
        "section": "Abstract",
        "source_type": "registry_abstract_fallback",
        "text": abstract,
    }


def semantic_text(chunk: dict[str, Any]) -> str:
    """构造 BGE 输入；全文不重复题名，避免题名让每个 chunk 获得相同偏置。"""

    section = " ".join(str(chunk.get("section") or "").split())
    text = " ".join(str(chunk.get("text") or "").split())
    if chunk.get("source_type") == "registry_abstract_fallback":
        title = " ".join(str(chunk.get("title") or "").split())
        return f"Title: {title}\nAbstract: {text}" if title else text
    return f"Section: {section}\nText: {text}" if section else text


# -----------------------------------------------------------------------------
# 模型与配置
# -----------------------------------------------------------------------------


def model_path_from_config(config_path: Path) -> Path | None:
    if not config_path.exists():
        return None
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    value = config.get("embedding", {}).get("local_model_path")
    return Path(value) if value else None


def anchor_record(anchor: Anchor) -> dict[str, Any]:
    record = asdict(anchor)
    record["matched_terms"] = list(anchor.matched_terms)
    record["anchor_source"] = "existing_first_pass_corpus"
    return record


def decision_fields(decision: ArticleDecision, original_status: str) -> dict[str, Any]:
    return {
        "semantic_decision": decision.decision,
        "semantic_original_status": original_status,
        "semantic_final_status": decision.final_status,
        "semantic_chunk_count": decision.chunk_count,
        "semantic_animal_hits": decision.animal_hits,
        "semantic_animal_ratio": round(decision.animal_ratio, 6),
        "semantic_lexical_hits": decision.lexical_animal_hits,
        "semantic_max_animal_score": round(decision.max_animal_score, 6),
        "semantic_mean_animal_score": round(decision.mean_animal_score, 6),
        "semantic_mean_human_score": round(decision.mean_human_score, 6),
        "semantic_mean_margin": round(decision.mean_margin, 6),
        "semantic_title_animal_terms": "; ".join(decision.title_animal_terms),
        "semantic_title_human_terms": "; ".join(decision.title_human_terms),
        "semantic_reason": decision.reason,
    }


# -----------------------------------------------------------------------------
# 语义复核主计算
# -----------------------------------------------------------------------------


def assess_records(
    records_with_status: list[tuple[dict[str, Any], str]],
    broad_chunks: list[dict[str, Any]],
    animal_anchors: list[Anchor],
    human_anchors: list[Anchor],
    model: Any,
    config: SemanticConfig,
    batch_size: int,
) -> tuple[list[tuple[dict[str, Any], str, ArticleDecision]], dict[str, ArticleDecision]]:
    """一次编码全部候选 chunk，再按文章切片聚合，避免反复调用 GPU。"""

    by_article = index_chunks_by_article(broad_chunks)
    all_texts: list[str] = []
    article_slices: list[tuple[int, int]] = []

    for record, _status in records_with_status:
        article_chunks = chunks_for_record(record, by_article)
        if not article_chunks:
            fallback = fallback_abstract_chunk(record)
            article_chunks = [fallback] if fallback else []
        start = len(all_texts)
        all_texts.extend(semantic_text(chunk) for chunk in article_chunks if chunk)
        article_slices.append((start, len(all_texts)))

    animal_anchor_texts = [anchor.text for anchor in animal_anchors]
    human_anchor_texts = [anchor.text for anchor in human_anchors]
    if not animal_anchor_texts or not human_anchor_texts:
        raise ValueError("animal and human anchor sets must both be non-empty")

    anchor_embeddings = encode_texts(
        model,
        animal_anchor_texts + human_anchor_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    split_at = len(animal_anchor_texts)
    animal_anchor_embeddings = anchor_embeddings[:split_at]
    human_anchor_embeddings = anchor_embeddings[split_at:]

    if all_texts:
        text_embeddings = encode_texts(
            model,
            all_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        animal_scores = top_k_mean_similarity(
            text_embeddings, animal_anchor_embeddings, config.anchor_top_k
        )
        human_scores = top_k_mean_similarity(
            text_embeddings, human_anchor_embeddings, config.anchor_top_k
        )
        chunk_scores = build_chunk_scores(animal_scores, human_scores, all_texts, config)
    else:
        chunk_scores = []

    assessed: list[tuple[dict[str, Any], str, ArticleDecision]] = []
    decisions_by_id: dict[str, ArticleDecision] = {}
    for (record, original_status), (start, end) in zip(records_with_status, article_slices):
        decision = decide_article(
            original_status=original_status,
            title=str(record.get("title") or ""),
            scores=chunk_scores[start:end],
            config=config,
        )
        assessed.append((record, original_status, decision))
        for item_id in record_ids(record):
            decisions_by_id[item_id] = decision
    return assessed, decisions_by_id


# -----------------------------------------------------------------------------
# 最终 registry/chunk 路由
# -----------------------------------------------------------------------------


def existing_excluded_decision() -> ArticleDecision:
    return ArticleDecision(
        decision="not_checked_existing_exclude",
        final_status="excluded",
        reason="already excluded by first-pass medical rules",
        chunk_count=0,
        animal_hits=0,
        animal_ratio=0.0,
        lexical_animal_hits=0,
        max_animal_score=0.0,
        mean_animal_score=0.0,
        mean_human_score=0.0,
        mean_margin=0.0,
        title_animal_terms=(),
        title_human_terms=(),
    )


def annotate_registry(
    assessed: list[tuple[dict[str, Any], str, ArticleDecision]],
    excluded_records: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    outputs: dict[str, list[dict[str, Any]]] = {"strict": [], "review": [], "excluded": []}
    audit: list[dict[str, Any]] = []

    for record, original_status, decision in assessed:
        annotated = dict(record)
        annotated.update(decision_fields(decision, original_status))
        outputs[decision.final_status].append(annotated)
        audit.append(annotated)

    old_decision = existing_excluded_decision()
    for record in excluded_records:
        annotated = dict(record)
        annotated.update(decision_fields(old_decision, "excluded"))
        outputs["excluded"].append(annotated)
    return outputs, audit


def decision_for_chunk(
    chunk: dict[str, Any], decisions_by_id: dict[str, ArticleDecision]
) -> ArticleDecision | None:
    for item_id in record_ids(chunk):
        if item_id in decisions_by_id:
            return decisions_by_id[item_id]
    return None


def annotate_chunk(
    chunk: dict[str, Any], decision: ArticleDecision | None, original_bucket: str
) -> dict[str, Any]:
    annotated = dict(chunk)
    if decision is None:
        annotated.update(
            {
                "semantic_decision": "not_matched_to_registry",
                "semantic_original_status": original_bucket,
                "semantic_final_status": original_bucket,
                "semantic_reason": "chunk identifiers did not match a registry record",
            }
        )
    else:
        annotated.update(decision_fields(decision, original_bucket))
    return annotated


def route_chunks(
    strict_chunks: list[dict[str, Any]],
    review_chunks: list[dict[str, Any]],
    excluded_chunks: list[dict[str, Any]],
    decisions_by_id: dict[str, ArticleDecision],
) -> dict[str, list[dict[str, Any]]]:
    """保留第一轮 chunk 质量判断，只对文章级降级进行移动。"""

    outputs: dict[str, list[dict[str, Any]]] = {"strict": [], "review": [], "excluded": []}

    for chunk in strict_chunks:
        decision = decision_for_chunk(chunk, decisions_by_id)
        final_bucket = decision.final_status if decision else "strict"
        outputs[final_bucket].append(annotate_chunk(chunk, decision, "strict"))

    for chunk in review_chunks:
        decision = decision_for_chunk(chunk, decisions_by_id)
        # 第一轮已经判为 review 的 chunk 不会因文章通过语义检查而自动升级为 strict。
        final_bucket = "excluded" if decision and decision.final_status == "excluded" else "review"
        outputs[final_bucket].append(annotate_chunk(chunk, decision, "review"))

    for chunk in excluded_chunks:
        # 第一轮已排除的噪声 chunk 永不被第二轮重新提升。
        outputs["excluded"].append(annotate_chunk(chunk, None, "excluded"))
    return outputs


# -----------------------------------------------------------------------------
# 审计文件与报告
# -----------------------------------------------------------------------------


SEMANTIC_FIELDS = [
    "semantic_decision",
    "semantic_original_status",
    "semantic_final_status",
    "semantic_chunk_count",
    "semantic_animal_hits",
    "semantic_animal_ratio",
    "semantic_lexical_hits",
    "semantic_max_animal_score",
    "semantic_mean_animal_score",
    "semantic_mean_human_score",
    "semantic_mean_margin",
    "semantic_title_animal_terms",
    "semantic_title_human_terms",
    "semantic_reason",
]

AUDIT_FIELDS = [
    "pmid",
    "pmcid",
    "doi",
    "title",
    "year",
    "journal",
] + SEMANTIC_FIELDS


def write_report(
    path: Path,
    config: SemanticConfig,
    animal_anchors: list[Anchor],
    human_anchors: list[Anchor],
    assessed: list[tuple[dict[str, Any], str, ArticleDecision]],
    registry_outputs: dict[str, list[dict[str, Any]]],
    chunk_outputs: dict[str, list[dict[str, Any]]],
    model_path: Path,
) -> None:
    decision_counts = Counter(decision.decision for _, _, decision in assessed)
    moved_to_excluded = [item for item in assessed if item[2].final_status == "excluded"]
    moved_to_review = [
        item for item in assessed if item[1] == "strict" and item[2].final_status == "review"
    ]

    lines = [
        "# Animal Semantic Filter Report",
        "",
        f"created_at: {datetime.now(timezone.utc).isoformat()}",
        f"model_path: {model_path}",
        "anchor_source: real sentences extracted from first-pass corpus",
        f"animal_anchors: {len(animal_anchors)}",
        f"human_anchors: {len(human_anchors)}",
        "",
        "## Thresholds",
        f"chunk_animal_threshold: {config.chunk_animal_threshold}",
        f"animal_human_margin: {config.animal_human_margin}",
        f"no_term_animal_threshold: {config.no_term_animal_threshold}",
        f"no_term_animal_margin: {config.no_term_animal_margin}",
        f"article_animal_ratio_strictly_greater_than: {config.article_animal_ratio}",
        f"review_animal_ratio: {config.review_animal_ratio}",
        f"min_animal_hits: {config.min_animal_hits}",
        f"single_chunk_threshold: {config.single_chunk_threshold}",
        f"anchor_top_k: {config.anchor_top_k}",
        "",
        "## Decisions",
    ]
    for name in ("keep", "review_animal", "exclude_animal", "not_assessed"):
        lines.append(f"{name}: {decision_counts.get(name, 0)}")
    lines.extend(
        [
            f"moved_to_excluded: {len(moved_to_excluded)}",
            f"strict_moved_to_review: {len(moved_to_review)}",
            "",
            "## Final Registry",
            f"strict: {len(registry_outputs['strict'])}",
            f"review: {len(registry_outputs['review'])}",
            f"excluded: {len(registry_outputs['excluded'])}",
            "",
            "## Final Chunks",
            f"strict: {len(chunk_outputs['strict'])}",
            f"review: {len(chunk_outputs['review'])}",
            f"excluded: {len(chunk_outputs['excluded'])}",
            "",
            "## Notes",
            "- The 40% threshold is an article-level animal chunk ratio, not a cosine percentage.",
            "- Existing first-pass files are not overwritten.",
            "- Inspect animal_audit.csv before treating the final strict corpus as medically approved.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# 命令行与完整流程
# -----------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use corpus-derived BGE anchors to remove animal mastitis articles after first-pass filtering."
    )
    parser.add_argument("--registry-dir", type=Path, default=Path("data/registry/filtered"))
    parser.add_argument("--chunk-dir", type=Path, default=Path("data/articles/processed"))
    parser.add_argument("--registry-out-dir", type=Path, default=Path("data/registry/filtered/semantic"))
    parser.add_argument("--chunk-out-dir", type=Path, default=Path("data/articles/processed/semantic"))
    parser.add_argument("--config", type=Path, default=Path("configs/embedding.yaml"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-anchors", type=int, default=96)
    parser.add_argument("--chunk-threshold", type=float, default=0.52)
    parser.add_argument("--animal-margin", type=float, default=0.02)
    parser.add_argument("--no-term-threshold", type=float, default=0.62)
    parser.add_argument("--no-term-margin", type=float, default=0.06)
    parser.add_argument("--article-ratio", type=float, default=0.40)
    parser.add_argument("--review-ratio", type=float, default=0.20)
    parser.add_argument("--min-animal-hits", type=int, default=2)
    parser.add_argument("--single-chunk-threshold", type=float, default=0.60)
    parser.add_argument("--anchor-top-k", type=int, default=3)
    parser.add_argument(
        "--limit-records",
        type=int,
        help="Only assess the first N strict/review records; use temporary output dirs for smoke tests.",
    )
    return parser.parse_args(argv)


def required_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "registry_strict": args.registry_dir / "literature_registry_strict.csv",
        "registry_review": args.registry_dir / "literature_registry_review.csv",
        "registry_excluded": args.registry_dir / "literature_registry_excluded.csv",
        "chunks_broad": args.chunk_dir / "rag_chunks.jsonl",
        "chunks_strict": args.chunk_dir / "rag_chunks_strict.jsonl",
        "chunks_review": args.chunk_dir / "rag_chunks_review.jsonl",
        "chunks_excluded": args.chunk_dir / "rag_chunks_excluded.jsonl",
    }


def validate_ratio(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = required_paths(args)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        print("first-pass filter outputs are missing:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
        print("run: bash scripts/filter_corpus.sh", file=sys.stderr)
        return 2

    model_path = args.model_path or model_path_from_config(args.config)
    if model_path is None or not model_path.exists():
        print(f"BGE model path not found: {model_path}", file=sys.stderr)
        return 2

    for name, value in (
        ("chunk_threshold", args.chunk_threshold),
        ("no_term_threshold", args.no_term_threshold),
        ("article_ratio", args.article_ratio),
        ("review_ratio", args.review_ratio),
        ("single_chunk_threshold", args.single_chunk_threshold),
    ):
        validate_ratio(name, value)
    if args.review_ratio > args.article_ratio:
        print("review_ratio must not exceed article_ratio", file=sys.stderr)
        return 2

    config = SemanticConfig(
        chunk_animal_threshold=args.chunk_threshold,
        animal_human_margin=args.animal_margin,
        no_term_animal_threshold=args.no_term_threshold,
        no_term_animal_margin=args.no_term_margin,
        article_animal_ratio=args.article_ratio,
        review_animal_ratio=args.review_ratio,
        min_animal_hits=args.min_animal_hits,
        single_chunk_threshold=args.single_chunk_threshold,
        anchor_top_k=args.anchor_top_k,
    )

    strict_records = read_csv(paths["registry_strict"])
    review_records = read_csv(paths["registry_review"])
    excluded_records = read_csv(paths["registry_excluded"])
    broad_chunks = read_jsonl(paths["chunks_broad"])
    strict_chunks = read_jsonl(paths["chunks_strict"])
    review_chunks = read_jsonl(paths["chunks_review"])
    excluded_chunks = read_jsonl(paths["chunks_excluded"])

    # 锚点来源完全可追溯：动物锚点来自第一轮 excluded，人类对照锚点来自 strict。
    animal_anchors = extract_anchors(excluded_chunks, "animal", max_anchors=args.max_anchors)
    human_anchors = extract_anchors(strict_chunks, "human", max_anchors=args.max_anchors)
    if len(animal_anchors) < 8 or len(human_anchors) < 8:
        print(
            f"insufficient corpus-derived anchors: animal={len(animal_anchors)}, human={len(human_anchors)}",
            file=sys.stderr,
        )
        return 2

    device = resolve_device(args.device)
    print(f"device={device}")
    print(f"animal_anchors={len(animal_anchors)}")
    print(f"human_anchors={len(human_anchors)}")
    model = load_sentence_transformer(model_path, device)

    records_with_status = [
        *((record, "strict") for record in strict_records),
        *((record, "review") for record in review_records),
    ]
    if args.limit_records is not None:
        if args.limit_records <= 0:
            print("limit_records must be greater than zero", file=sys.stderr)
            return 2
        records_with_status = records_with_status[: args.limit_records]
    assessed, decisions_by_id = assess_records(
        records_with_status=records_with_status,
        broad_chunks=broad_chunks,
        animal_anchors=animal_anchors,
        human_anchors=human_anchors,
        model=model,
        config=config,
        batch_size=args.batch_size,
    )

    registry_outputs, audit = annotate_registry(assessed, excluded_records)
    chunk_outputs = route_chunks(
        strict_chunks, review_chunks, excluded_chunks, decisions_by_id
    )

    base_fields = list(strict_records[0].keys()) if strict_records else []
    registry_fields = list(dict.fromkeys(base_fields + SEMANTIC_FIELDS))
    for status in ("strict", "review", "excluded"):
        write_csv(
            args.registry_out_dir / f"literature_registry_{status}.csv",
            registry_outputs[status],
            registry_fields,
        )
        write_jsonl(
            args.chunk_out_dir / f"rag_chunks_{status}.jsonl",
            chunk_outputs[status],
        )

    # 审计表按动物比例和最大动物分数降序，打开后最需要人工核查的文章排在最前。
    audit.sort(
        key=lambda row: (
            -float(row.get("semantic_animal_ratio") or 0),
            -float(row.get("semantic_max_animal_score") or 0),
            str(row.get("title") or ""),
        )
    )
    write_csv(args.registry_out_dir / "animal_audit.csv", audit, AUDIT_FIELDS)
    write_jsonl(
        args.registry_out_dir / "anchors.jsonl",
        [anchor_record(anchor) for anchor in animal_anchors + human_anchors],
    )
    write_report(
        args.registry_out_dir / "filter_report.md",
        config,
        animal_anchors,
        human_anchors,
        assessed,
        registry_outputs,
        chunk_outputs,
        model_path,
    )

    decision_counts = Counter(decision.decision for _, _, decision in assessed)
    print(f"assessed_articles={len(assessed)}")
    print(f"kept={decision_counts.get('keep', 0)}")
    print(f"moved_to_review={decision_counts.get('review_animal', 0)}")
    print(f"moved_to_excluded={decision_counts.get('exclude_animal', 0)}")
    print(f"not_assessed={decision_counts.get('not_assessed', 0)}")
    print(f"final_strict_articles={len(registry_outputs['strict'])}")
    print(f"final_review_articles={len(registry_outputs['review'])}")
    print(f"final_excluded_articles={len(registry_outputs['excluded'])}")
    print(f"audit={args.registry_out_dir / 'animal_audit.csv'}")
    print(f"report={args.registry_out_dir / 'filter_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
