"""根据清洗后的 RAG 证据构造 Qwen 报告生成 prompt。

输入：step09_clean_evidence.py 输出的一个 ``*_cleaned_evidence.json``。
输出：默认在输入文件同目录生成同名前缀的 ``*_prompt.txt``。
说明：本步骤只负责 prompt 格式与报告要求，不加载 Qwen、不重新检索、不筛除
      因治疗方案不完全一致而仍具有参考价值的类比证据。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_cleaned_evidence(path: Path) -> dict[str, Any]:
    """读取并验证 step09 的核心输出契约。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("cleaned evidence JSON root must be an object")
    question = payload.get("question")
    evidence = payload.get("evidence")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("cleaned evidence JSON must contain a non-empty 'question'")
    if not isinstance(evidence, list):
        raise ValueError("cleaned evidence JSON must contain an 'evidence' list")
    return payload


def _citation_text(record: dict[str, Any]) -> str:
    citation = str(record.get("citation") or "").strip()
    if citation:
        return citation
    parts = [
        str(record.get("title") or "").strip(),
        str(record.get("journal") or "").strip(),
        str(record.get("year") or "").strip(),
        str(record.get("section") or "").strip(),
    ]
    return " | ".join(part for part in parts if part) or "出处信息缺失"


def build_evidence_blocks(evidence_records: list[dict[str, Any]]) -> str:
    """只读取 step09 的 cleaned 字段，绝不回退到未经清洗的原始 text。"""

    blocks: list[str] = []
    seen_ids: set[str] = set()
    for record in evidence_records:
        if not isinstance(record, dict):
            raise ValueError("each evidence record must be an object")
        evidence_id = str(record.get("evidence_id") or "").strip()
        if not evidence_id:
            raise ValueError("each evidence record must have a non-empty evidence_id")
        if evidence_id in seen_ids:
            raise ValueError(f"duplicate evidence_id: {evidence_id}")
        seen_ids.add(evidence_id)

        context = str(record.get("cleaned_text") or "").strip()
        if not context:
            raise ValueError(f"{evidence_id} has no cleaned_text; run step09 first")
        matched = str(record.get("cleaned_matched_text") or "").strip()
        lines = [
            f"### [{evidence_id}]",
            f"出处：{_citation_text(record)}",
            f"检索语言：{record.get('retrieval_language') or '未知'}",
        ]
        # 命中子块与扩展上下文相同时不重复写入，避免无意义地增加 Qwen 输入长度。
        if matched and matched != context:
            lines.extend([f"命中片段：{matched}", f"扩展上下文：{context}"])
        else:
            lines.append(f"证据正文：{context}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "（没有可用证据）"


def select_evidence_records(
    evidence_records: list[dict[str, Any]], max_evidence: int = 10
) -> list[dict[str, Any]]:
    """最多保留指定数量的证据，并在中英文证据同时存在时保持基本平衡。"""

    if max_evidence <= 0:
        raise ValueError("max_evidence must be greater than zero")

    language_indices: dict[str, list[int]] = {"zh": [], "en": []}
    for index, record in enumerate(evidence_records):
        language = str(record.get("retrieval_language") or "").lower()
        if language in language_indices:
            language_indices[language].append(index)

    if not language_indices["zh"] or not language_indices["en"]:
        return evidence_records[:max_evidence]

    # step08 的结果按语言成组排列。这里按组内 rerank 次序交错放入中英文证据，
    # 让两边的高排名结果都较早出现，同时不改写用于追溯的 E 编号。
    selected_indices: list[int] = []
    for rank in range(max(map(len, language_indices.values()))):
        for language in ("zh", "en"):
            if rank < len(language_indices[language]):
                selected_indices.append(language_indices[language][rank])
            if len(selected_indices) >= max_evidence:
                break
        if len(selected_indices) >= max_evidence:
            break

    selected_set = set(selected_indices)
    for index in range(len(evidence_records)):
        if len(selected_indices) >= max_evidence:
            break
        if index not in selected_set:
            selected_indices.append(index)
            selected_set.add(index)
    return [evidence_records[index] for index in selected_indices]


def build_report_prompt(payload: dict[str, Any], max_evidence: int = 10) -> str:
    """构造以回答用户问题为中心、可独立迭代的医学报告 prompt。"""

    question = str(payload.get("question") or "").strip()
    evidence_records = payload.get("evidence")
    if not question:
        raise ValueError("question is empty")
    if not isinstance(evidence_records, list):
        raise ValueError("evidence must be a list")
    selected_records = select_evidence_records(evidence_records, max_evidence)
    evidence_text = build_evidence_blocks(selected_records)

    return f"""你是一名严谨的医学研究证据分析助手。你的任务是参考给定证据回答用户问题，并生成供医生和研究人员讨论的中文报告，而不是逐篇复述文献，也不是替医生作出诊断或处方决定。

## 核心规则
1. 只能使用下方提供的证据，不得把模型自身知识写成已被文献证明的事实。
2. 首先直接回答用户问题。证据是回答的依据，不是报告的目录；必须围绕问题综合多条证据，不要按 E1、E2、E3 的顺序逐篇摘要。
3. 每个关键事实、判断或推论后必须引用证据编号，例如 [E1] 或 [E1][E3]。引用必须确实支持紧邻的表述。
4. 优先采用与问题直接匹配的证据；不要因为文献中的疾病亚型、药物组合或治疗条件与问题不完全一致，就删除高度相关的类比证据。
5. 使用类比证据时，必须明确写出目标问题与原研究在药物、疾病亚型、患者群体、疗程或结局上的相同点和不同点。
6. 原研究的结果只能归属于原研究方案；不得把相近方案的疗效直接表述为目标方案已经得到验证。允许作审慎的证据综合或推论，但必须明确标为“基于间接证据的推论”，并说明不能替代直接验证。
7. 若没有直接研究，仍需针对问题作答：先明确“未检索到直接证据”或“现有证据不足以确定”，再说明间接证据支持什么、不支持什么及其可迁移边界。
8. 不得补造样本量、剂量、疗程、疗效、不良反应或统计结果，也不得用模型的一般医学知识填补证据空白。
9. 若证据块中仍夹杂明显与问题无关的版面拼接残留，只忽略这些残留，不得据此编造结论；但不能把“方案不完全一致”本身视为噪声。
10. 不得把甲方案的剂量、疗程或结局移植给乙方案；不同药物组合必须分别表述。不得把“无效”“复发”等观察结果自行解释为耐药、依从性差或其他未被证据明确报告的原因。
11. 对“未发现研究”“缺少随机对照试验”等判断，只能限定为“本次提供的证据中未见”，不得据此概括全部医学文献。
12. 生成前先在内部逐条核对每个 E 的疾病、完整药物组合、研究设计、样本量、剂量、疗程和结局；一句话中的这些信息必须来自同一个 E。多个 E 可以共同支持概括性判断，但不得拼成一项并不存在的研究。
13. 单药研究必须明确称为单药研究，不能称为“类似药物组合”。若证据未明确陈述疾病机制、方案优劣或因果关系，不得自行推断。
14. 写“未报告”“缺少”或“无法判断”前，必须检查全部证据；只要任一证据已经给出该信息，就不得声称缺失。疗效判定标准、计算公式、研究计划和纳入标准不是实际观察结果，不能作为疗效数据。
15. 若存在直接证据，应先完整、准确地使用其中与问题有关的疗效、安全性、复发和随访结果。直接证据已经回答某项内容时，不得为了增加引用数量再用类比证据重复该内容；只有直接证据确实缺少必要信息时，才可明确标注后用类比证据补充。

## 用户问题
{question}

## 清洗后的检索证据（本次使用 {len(selected_records)} 条）
{evidence_text}

## 报告组织要求
以下三个部分必须出现：
1. 针对问题的回答
2. 证据基础与适用边界
3. 证据缺口与不确定性

以下部分仅在至少一条证据明确提供相应信息时选用，不要求全部输出：
- 疗效与结局
- 治疗方案与疗程
- 安全性与监测
- 复发与随访
- 目标方案与既有方案比较
- 供临床和研究讨论的要点

“针对问题的回答”必须在开头用一个简洁段落给出，并在该段内直接引用支持结论的证据。不要为了凑齐标题而重复内容或补造信息。某类信息完全缺失时，不要建立空泛的独立章节；将必要说明集中写入“证据缺口与不确定性”。报告必须区分“文献直接报告的结果”“综合多条证据得到的判断”和“基于相近研究的审慎推论”。全文尽量控制在 900 个中文字符左右，每个可选部分最多一个紧凑段落；不要输出证据原文堆砌，也不要给具体患者下达治疗指令。

## 输出前静默自检
在输出报告前自行检查，但不要展示检查过程：
- 不要求使用完所有证据；无实际结果、与问题无关或仅含版面残留的证据应忽略。
- 若一条直接证据已经足以回答核心问题，应以它为主，不要为了显得全面而强行加入较弱的类比证据。
- 每句话都应能在所引 E 的正文中找到直接依据；找不到就删除，不能凭标题、常识或其他 E 补齐。
- 再核对一次药物组合、剂量、疗程、研究设计、随访、复发和安全性，确保没有跨 E 搬运。
- 再核对所有“未报告/缺少”表述，确保没有被任一证据中的明确数据反驳。
- 不得引入用户问题中没有提出的目标剂量、疗程、亚型或比较条件，再声称证据与其不一致。
"""


def output_path_for_evidence(evidence_path: Path) -> Path:
    stem = evidence_path.stem
    if stem.endswith("_cleaned_evidence"):
        stem = stem[: -len("_cleaned_evidence")]
    return evidence_path.with_name(f"{stem}_prompt.txt")


def write_prompt(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt.rstrip() + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Qwen report prompt from step09 cleaned evidence."
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-evidence",
        type=int,
        default=10,
        help="送入 Qwen prompt 的最大证据条数；默认 10。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.evidence.exists():
        print(f"cleaned evidence not found: {args.evidence}", file=sys.stderr)
        return 2
    try:
        payload = load_cleaned_evidence(args.evidence)
        selected_records = select_evidence_records(payload["evidence"], args.max_evidence)
        prompt = build_report_prompt(payload, args.max_evidence)
        output_path = args.output or output_path_for_evidence(args.evidence)
        write_prompt(output_path, prompt)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"prompt_path={output_path}")
    print(f"evidence_count={len(selected_records)}")
    print(f"prompt_chars={len(prompt)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
