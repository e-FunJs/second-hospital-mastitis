"""
用途：为动物乳腺炎语义复核提供锚点抽取、chunk 判定和文章级决策规则。

输入：第一轮严格筛选产生的真实文献 chunk，以及 BGE 计算出的归一化向量。
输出：本模块不直接写文件；返回语义锚点、chunk 分数和文章级决策，供
      ``animal_filter.py`` 生成最终语料与审计报告。

设计说明：
1. 动物/人类锚点都从当前知识库真实文献中抽取，不在代码中编造医学句子。
2. “动物相关性超过 40%”表示一篇文章中动物语义 chunk 的占比，而不是把
   余弦相似度误读为百分比。
3. 动物判断同时比较动物锚点和人类临床锚点，降低综述中偶然提及动物研究
   导致整篇人类临床文献被误删的风险。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from rag_medical.semantic_chunk import split_sentences, word_count


# -----------------------------------------------------------------------------
# 语义词表
# -----------------------------------------------------------------------------
# 这些词只用于：
# - 从已排除语料中定位“明确动物语境”的真实句子，作为 BGE 动物锚点；
# - 为 BGE 结果增加可审计的词面证据。
# 最终是否排除文章，仍由 BGE 语义分数、动物/人类分数差和文章级占比共同决定。

ANIMAL_TERMS: tuple[str, ...] = (
    "bovine",
    "dairy cow",
    "dairy cows",
    "cow",
    "cows",
    "cattle",
    "dairy cattle",
    "dairy farm",
    "dairy farms",
    "bulk tank milk",
    "herd",
    "udder",
    "teat",
    "intramammary",
    "somatic cell count",
    "milk yield",
    "veterinary",
    "goat",
    "goats",
    "sheep",
    "ewe",
    "ewes",
    "camel",
    "camels",
    "buffalo",
    "buffaloes",
    "porcine",
    "sow",
    "sows",
    "murine mastitis",
    "mouse model",
    "mice",
    "rat model",
    "rats",
)

HUMAN_TARGET_TERMS: tuple[str, ...] = (
    "idiopathic granulomatous mastitis",
    "granulomatous lobular mastitis",
    "granulomatous mastitis",
    "non-puerperal mastitis",
    "nonpuerperal mastitis",
    "non-lactational mastitis",
    "nonlactational mastitis",
    "periductal mastitis",
    "plasma cell mastitis",
    "mammary duct ectasia",
    "breast tuberculosis",
    "mammary tuberculosis",
    "tuberculous mastitis",
    "tubercular mastitis",
)

HUMAN_CONTEXT_TERMS: tuple[str, ...] = (
    "patient",
    "patients",
    "woman",
    "women",
    "female patient",
    "female patients",
    "human breast",
    "clinical presentation",
    "clinical treatment",
)

HUMAN_TITLE_TERMS: tuple[str, ...] = (
    "human",
    "patient",
    "patients",
    "woman",
    "women",
    "mother",
    "mothers",
    "infant",
    "infants",
    "breastfeeding",
)


@dataclass(frozen=True)
class SemanticConfig:
    """集中保存可调阈值，避免阈值散落在流程代码中。"""

    chunk_animal_threshold: float = 0.52
    animal_human_margin: float = 0.02
    no_term_animal_threshold: float = 0.62
    no_term_animal_margin: float = 0.06
    article_animal_ratio: float = 0.40
    review_animal_ratio: float = 0.20
    min_animal_hits: int = 2
    single_chunk_threshold: float = 0.60
    anchor_top_k: int = 3


@dataclass(frozen=True)
class Anchor:
    """从真实文献中抽取的一条可追溯语义锚点。"""

    label: str
    text: str
    source_id: str
    chunk_id: str
    title: str
    matched_terms: tuple[str, ...]
    selection_score: int


@dataclass(frozen=True)
class ChunkScore:
    """一个文献 chunk 相对于动物和人类锚点的语义检查结果。"""

    animal_score: float
    human_score: float
    margin: float
    is_animal: bool
    animal_terms: tuple[str, ...]


@dataclass(frozen=True)
class ArticleDecision:
    """文章级语义复核决策及其审计指标。"""

    decision: str
    final_status: str
    reason: str
    chunk_count: int
    animal_hits: int
    animal_ratio: float
    lexical_animal_hits: int
    max_animal_score: float
    mean_animal_score: float
    mean_human_score: float
    mean_margin: float
    title_animal_terms: tuple[str, ...]
    title_human_terms: tuple[str, ...]


# -----------------------------------------------------------------------------
# 词面匹配与锚点抽取
# -----------------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    """统一大小写、连字符和空白，保证术语匹配稳定。"""

    text = " ".join(str(value or "").lower().split())
    return text.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")


def contains_term(text: str, term: str) -> bool:
    """按英文词边界匹配，避免 ``cow`` 意外命中更长单词。"""

    escaped = re.escape(term.lower())
    plural = "s?" if not term.endswith("s") else ""
    pattern = r"(?<![a-z0-9])" + escaped + plural + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def matching_terms(text: str, terms: Iterable[str]) -> tuple[str, ...]:
    normalized = normalize_text(text)
    return tuple(term for term in terms if contains_term(normalized, term))


def source_id(record: dict[str, Any]) -> str:
    """优先选择稳定的 PMC/PubMed/DOI 标识，便于回到原文核查。"""

    for field in ("pmcid", "pmid", "doi"):
        value = str(record.get(field) or "").strip()
        if value:
            return f"{field}:{value}"
    return f"chunk:{record.get('chunk_id', 'unknown')}"


def _anchor_candidate(sentence: str, chunk: dict[str, Any], label: str) -> Anchor | None:
    """判断真实句子能否作为动物或人类锚点。"""

    sentence = " ".join(sentence.split())
    sentence_words = word_count(sentence)
    if sentence_words < 12 or sentence_words > 100:
        return None

    animal_matches = matching_terms(sentence, ANIMAL_TERMS)
    human_target_matches = matching_terms(sentence, HUMAN_TARGET_TERMS)
    human_context_matches = matching_terms(sentence, HUMAN_CONTEXT_TERMS)
    has_mastitis = contains_term(normalize_text(sentence), "mastitis")

    if label == "animal":
        # 动物锚点必须在同一句中同时出现 mastitis 与动物语境。只出现 milk/cow
        # 而不讨论乳腺炎的句子不能代表本任务要排除的语义。
        if not has_mastitis or not animal_matches:
            return None
        selection_score = len(animal_matches) * 4 + min(sentence_words // 20, 4)
        matched = animal_matches
    elif label == "human":
        # 人类锚点必须明确指向目标疾病；同时要求临床人群词或治疗语境，且不能
        # 含动物词，以免把比较性综述中的动物段落混进人类对照锚点。
        treatment_context = any(
            contains_term(normalize_text(sentence), term)
            for term in ("treatment", "therapy", "patients", "diagnosis", "recurrence")
        )
        if not human_target_matches or animal_matches or not (human_context_matches or treatment_context):
            return None
        selection_score = len(human_target_matches) * 4 + len(human_context_matches) * 2
        matched = tuple(dict.fromkeys(human_target_matches + human_context_matches))
    else:
        raise ValueError(f"unsupported anchor label: {label}")

    return Anchor(
        label=label,
        text=sentence,
        source_id=source_id(chunk),
        chunk_id=str(chunk.get("chunk_id") or ""),
        title=str(chunk.get("title") or ""),
        matched_terms=matched,
        selection_score=selection_score,
    )


def extract_anchors(
    chunks: list[dict[str, Any]],
    label: str,
    max_anchors: int = 96,
    max_per_article: int = 3,
) -> list[Anchor]:
    """从语料抽取去重且来源分散的真实句子锚点。

    每篇文章最多贡献 ``max_per_article`` 条，防止某一篇长综述支配整个锚点集合。
    排序优先考虑术语覆盖度，再保持输入顺序，从而使重复运行结果稳定。
    """

    candidates: list[Anchor] = []
    seen_text: set[str] = set()
    for chunk in chunks:
        for sentence in split_sentences(str(chunk.get("text") or "")):
            candidate = _anchor_candidate(sentence, chunk, label)
            if candidate is None:
                continue
            normalized = normalize_text(candidate.text)
            if normalized in seen_text:
                continue
            seen_text.add(normalized)
            candidates.append(candidate)

    candidates.sort(key=lambda item: (-item.selection_score, item.source_id, item.text))
    selected: list[Anchor] = []
    article_counts: dict[str, int] = {}
    for candidate in candidates:
        count = article_counts.get(candidate.source_id, 0)
        if count >= max_per_article:
            continue
        selected.append(candidate)
        article_counts[candidate.source_id] = count + 1
        if len(selected) >= max_anchors:
            break
    return selected


# -----------------------------------------------------------------------------
# BGE 分数计算
# -----------------------------------------------------------------------------


def top_k_mean_similarity(
    text_embeddings: np.ndarray,
    anchor_embeddings: np.ndarray,
    top_k: int,
) -> np.ndarray:
    """计算每个文本对最相近 K 个锚点的平均余弦相似度。

    输入向量必须已归一化，此时矩阵点积就是余弦相似度。使用 top-k 平均值比
    单个最大值稳健：某条偶然相似的锚点不会单独决定整段文本的类别。
    """

    if text_embeddings.ndim != 2 or anchor_embeddings.ndim != 2:
        raise ValueError("text_embeddings and anchor_embeddings must be 2-D")
    if text_embeddings.shape[1] != anchor_embeddings.shape[1]:
        raise ValueError("text and anchor embedding dimensions must match")
    if anchor_embeddings.shape[0] == 0:
        raise ValueError("at least one anchor embedding is required")

    similarities = text_embeddings @ anchor_embeddings.T
    effective_k = min(max(1, top_k), similarities.shape[1])
    top_values = np.partition(similarities, -effective_k, axis=1)[:, -effective_k:]
    return np.mean(top_values, axis=1, dtype=np.float64).astype(np.float32)


def build_chunk_scores(
    animal_scores: np.ndarray,
    human_scores: np.ndarray,
    texts: list[str],
    config: SemanticConfig,
) -> list[ChunkScore]:
    """把连续相似度转换为可解释的 chunk 动物语义标签。"""

    if len(animal_scores) != len(human_scores) or len(animal_scores) != len(texts):
        raise ValueError("score arrays and texts must have the same length")

    results: list[ChunkScore] = []
    for animal_score, human_score, text in zip(animal_scores, human_scores, texts):
        animal_value = float(animal_score)
        human_value = float(human_score)
        margin = animal_value - human_value
        terms = matching_terms(text, ANIMAL_TERMS)

        # 有动物词时采用基础 BGE 阈值；没有动物词时必须同时达到更高的绝对分数
        # 和更大的 animal-human 差值。后者可以发现未使用既有词表的动物语境，
        # 但不会把仅仅讨论“乳汁/细菌/乳腺炎”的人类文章轻易判成动物 chunk。
        has_lexical_evidence = bool(terms)
        is_animal = (
            animal_value >= config.chunk_animal_threshold
            and margin >= config.animal_human_margin
            and (
                has_lexical_evidence
                or (
                    animal_value >= config.no_term_animal_threshold
                    and margin >= config.no_term_animal_margin
                )
            )
        )
        results.append(
            ChunkScore(
                animal_score=animal_value,
                human_score=human_value,
                margin=margin,
                is_animal=is_animal,
                animal_terms=terms,
            )
        )
    return results


# -----------------------------------------------------------------------------
# 文章级决策
# -----------------------------------------------------------------------------


def decide_article(
    original_status: str,
    title: str,
    scores: list[ChunkScore],
    config: SemanticConfig,
) -> ArticleDecision:
    """根据全文 chunk 占比决定保留、转复核或排除。

    多 chunk 文献按动物 chunk 比例判定；只有一个摘要 chunk 的文献无法计算稳定
    占比，因此必须同时满足更高语义阈值、正 margin 和明确动物词面证据。
    """

    title_terms = matching_terms(title, ANIMAL_TERMS)
    title_human_terms = matching_terms(title, HUMAN_TITLE_TERMS)
    if not scores:
        return ArticleDecision(
            decision="not_assessed",
            final_status=original_status,
            reason="no full-text chunk or abstract available for semantic checking",
            chunk_count=0,
            animal_hits=0,
            animal_ratio=0.0,
            lexical_animal_hits=0,
            max_animal_score=0.0,
            mean_animal_score=0.0,
            mean_human_score=0.0,
            mean_margin=0.0,
            title_animal_terms=title_terms,
            title_human_terms=title_human_terms,
        )

    animal_hits = sum(score.is_animal for score in scores)
    lexical_hits = sum(bool(score.animal_terms) for score in scores)
    ratio = animal_hits / len(scores)
    max_animal = max(score.animal_score for score in scores)
    mean_animal = sum(score.animal_score for score in scores) / len(scores)
    mean_human = sum(score.human_score for score in scores) / len(scores)
    mean_margin = sum(score.margin for score in scores) / len(scores)

    # 动物模型或奶牛乳腺炎若已明确写在题名中，属于比 40% chunk 比例更可靠的
    # 文章级证据。仍要求正文至少有一个 BGE 动物命中，防止仅凭词面误删。
    explicit_animal_title = bool(title_terms)
    human_title_protected = bool(title_human_terms) and not explicit_animal_title

    should_exclude = False
    exclusion_reason = ""
    if explicit_animal_title and animal_hits > 0:
        should_exclude = True
        exclusion_reason = (
            "explicit animal study terms in title confirmed by BGE: "
            + "; ".join(title_terms)
        )
    elif len(scores) == 1:
        only_score = scores[0]
        should_exclude = (
            only_score.is_animal
            and only_score.animal_score >= config.single_chunk_threshold
            and bool(only_score.animal_terms or title_terms)
            and not human_title_protected
        )
        if should_exclude:
            exclusion_reason = "single abstract has strong BGE and animal lexical evidence"
    else:
        # 严格执行“超过 40%”而不是“大于等于 40%”。同时要求至少两个动物
        # chunk，避免很短的文章因一个背景句被整篇排除。
        should_exclude = (
            ratio > config.article_animal_ratio
            and animal_hits >= config.min_animal_hits
            and lexical_hits >= config.min_animal_hits
            and not human_title_protected
        )
        if should_exclude:
            exclusion_reason = (
                f"animal semantic chunks {animal_hits}/{len(scores)} "
                f"({ratio:.1%}) exceed article threshold {config.article_animal_ratio:.0%}"
            )

    if should_exclude:
        return ArticleDecision(
            decision="exclude_animal",
            final_status="excluded",
            reason=exclusion_reason,
            chunk_count=len(scores),
            animal_hits=animal_hits,
            animal_ratio=ratio,
            lexical_animal_hits=lexical_hits,
            max_animal_score=max_animal,
            mean_animal_score=mean_animal,
            mean_human_score=mean_human,
            mean_margin=mean_margin,
            title_animal_terms=title_terms,
            title_human_terms=title_human_terms,
        )

    suspicious = (
        animal_hits > 0
        and (ratio >= config.review_animal_ratio or bool(title_terms))
        and bool(lexical_hits or title_terms)
        and not human_title_protected
    )
    if suspicious and original_status == "strict":
        return ArticleDecision(
            decision="review_animal",
            final_status="review",
            reason=(
                f"animal semantic signal requires review: {animal_hits}/{len(scores)} "
                f"chunks ({ratio:.1%})"
            ),
            chunk_count=len(scores),
            animal_hits=animal_hits,
            animal_ratio=ratio,
            lexical_animal_hits=lexical_hits,
            max_animal_score=max_animal,
            mean_animal_score=mean_animal,
            mean_human_score=mean_human,
            mean_margin=mean_margin,
            title_animal_terms=title_terms,
            title_human_terms=title_human_terms,
        )

    return ArticleDecision(
        decision="keep",
        final_status=original_status,
        reason="animal semantic evidence below exclusion threshold",
        chunk_count=len(scores),
        animal_hits=animal_hits,
        animal_ratio=ratio,
        lexical_animal_hits=lexical_hits,
        max_animal_score=max_animal,
        mean_animal_score=mean_animal,
        mean_human_score=mean_human,
        mean_margin=mean_margin,
        title_animal_terms=title_terms,
        title_human_terms=title_human_terms,
    )
