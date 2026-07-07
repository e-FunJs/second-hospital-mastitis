from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path


LEGACY_SOURCE_FILES = OrderedDict(
    [
        ("core", "core.csv"),
        ("outcome", "outcome.csv"),
        ("ultrasound", "ultrasound.csv"),
        ("therapy", "therapy.csv"),
    ]
)


# -----------------------------------------------------------------------------
# 输入文件发现
# -----------------------------------------------------------------------------
# 现在 PubMed 检索脚本会输出 pubmed_<query_key>.csv。这里优先合并这些文件，
# 这样新增 query 不需要再同步修改 merge 脚本；如果旧项目只有 core.csv 等文件，
# 再回退到 legacy 文件名。


def normalize(value: str | None) -> str:
    return (value or "").strip()


def discover_source_files(source_dir: Path) -> OrderedDict[str, str]:
    pubmed_files = sorted(source_dir.glob("pubmed_*.csv"))
    if pubmed_files:
        return OrderedDict((path.stem.removeprefix("pubmed_"), path.name) for path in pubmed_files)
    return OrderedDict(
        (name, filename)
        for name, filename in LEGACY_SOURCE_FILES.items()
        if (source_dir / filename).exists()
    )


def normalize_pmcid(value: str | None) -> str:
    pmcid = normalize(value).upper()
    if not pmcid:
        return ""
    if pmcid.startswith("PMC"):
        return pmcid
    if pmcid.isdigit():
        return f"PMC{pmcid}"
    return pmcid


def dedupe_key(row: dict[str, str]) -> tuple[str, str]:
    pmid = normalize(row.get("pmid"))
    doi = normalize(row.get("doi")).lower()
    title = " ".join(normalize(row.get("title")).lower().split())
    if pmid:
        return ("pmid", pmid)
    if doi:
        return ("doi", doi)
    return ("title", title)


def read_source(path: Path, source_name: str) -> list[dict[str, str]]:
    if not path.exists():
        print(f"warning: missing input file: {path}", file=sys.stderr)
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row = {key: normalize(value) for key, value in row.items()}
            row["pmcid"] = normalize_pmcid(row.get("pmcid"))
            row["_source_query"] = source_name
            rows.append(row)
        return rows


def merge_rows(source_dir: Path) -> list[dict[str, str]]:
    merged: OrderedDict[tuple[str, str], dict[str, str]] = OrderedDict()

    source_files = discover_source_files(source_dir)

    for source_name, filename in source_files.items():
        for row in read_source(source_dir / filename, source_name):
            key = dedupe_key(row)
            if key not in merged:
                row["source_queries"] = source_name
                row["source_query_count"] = "1"
                row.pop("_source_query", None)
                merged[key] = row
                continue

            existing = merged[key]
            existing_sources = set(existing.get("source_queries", "").split(";"))
            existing_sources.add(source_name)
            ordered_sources = [name for name in source_files if name in existing_sources]
            existing["source_queries"] = ";".join(ordered_sources)
            existing["source_query_count"] = str(len(ordered_sources))

            for field, value in row.items():
                if field.startswith("_"):
                    continue
                if not normalize(existing.get(field)) and normalize(value):
                    existing[field] = value

    return list(merged.values())


def write_csv(rows: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_fields = [
        "pmid",
        "pmcid",
        "title",
        "year",
        "journal",
        "doi",
        "abstract",
        "source_url",
        "disease_type",
        "topic",
        "study_type",
        "evidence_level",
        "has_full_text",
        "source_queries",
        "source_query_count",
    ]
    seen = set(base_fields)
    extra_fields = []
    for row in rows:
        for field in row:
            if field not in seen and not field.startswith("_"):
                seen.add(field)
                extra_fields.append(field)

    fieldnames = base_fields + extra_fields
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_summary(rows: list[dict[str, str]], out_path: Path) -> None:
    total = len(rows)
    with_pmcid = sum(1 for row in rows if normalize(row.get("pmcid")))
    with_doi = sum(1 for row in rows if normalize(row.get("doi")))
    multi_source = sum(1 for row in rows if int(row.get("source_query_count") or "0") > 1)

    lines = [
        "# Literature Registry Summary",
        "",
        f"- unique_records: {total}",
        f"- records_with_pmcid: {with_pmcid}",
        f"- records_with_doi: {with_doi}",
        f"- records_found_by_multiple_queries: {multi_source}",
        "",
        "## Source Query Counts",
        "",
    ]

    for source in sorted({name for row in rows for name in row.get("source_queries", "").split(";") if name}):
        count = sum(1 for row in rows if source in row.get("source_queries", "").split(";"))
        lines.append(f"- {source}: {count}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge and deduplicate PubMed registry CSV files.")
    parser.add_argument("--source-dir", type=Path, default=Path("data/registry/raw"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/registry/processed/literature_registry.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/registry/processed/literature_registry_summary.md"),
    )
    args = parser.parse_args(argv)

    rows = merge_rows(args.source_dir)
    write_csv(rows, args.out)
    write_summary(rows, args.summary)

    print(f"unique_records={len(rows)}")
    print(f"out={args.out}")
    print(f"summary={args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

