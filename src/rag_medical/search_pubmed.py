"""
用途：按 configs/queries.yaml 中的 query 检索 PubMed 文献元数据。
输入：query-key 或自定义 PubMed query。
输出：data/registry/raw/pubmed_<query_key>.csv。
说明：只获得题名、摘要、PMID、PMCID 等元数据，不下载全文。
"""

from __future__ import annotations

import argparse
import csv
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from rag_medical.schema import LiteratureRecord


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


FALLBACK_QUERIES = {
    "core_english": (
        '("non-puerperal mastitis"[Title/Abstract] OR "nonpuerperal mastitis"[Title/Abstract] '
        'OR "non-lactational mastitis"[Title/Abstract] OR "nonlactational mastitis"[Title/Abstract] '
        'OR "idiopathic granulomatous mastitis"[Title/Abstract] '
        'OR "granulomatous mastitis"[Title/Abstract] '
        'OR "granulomatous lobular mastitis"[Title/Abstract] '
        'OR "periductal mastitis"[Title/Abstract] OR "plasma cell mastitis"[Title/Abstract])'
    ),
    "treatment_outcome": (
        '(("idiopathic granulomatous mastitis"[Title/Abstract] '
        'OR "granulomatous mastitis"[Title/Abstract] '
        'OR "non-lactational mastitis"[Title/Abstract] '
        'OR "non-puerperal mastitis"[Title/Abstract]) '
        'AND (treatment[Title/Abstract] OR therapy[Title/Abstract] '
        'OR outcome[Title/Abstract] OR response[Title/Abstract] '
        'OR remission[Title/Abstract] OR recurrence[Title/Abstract] OR relapse[Title/Abstract]))'
    ),
    "ultrasound": (
        '(("idiopathic granulomatous mastitis"[Title/Abstract] '
        'OR "granulomatous mastitis"[Title/Abstract] '
        'OR "non-lactational mastitis"[Title/Abstract]) '
        'AND (ultrasound[Title/Abstract] OR ultrasonography[Title/Abstract] '
        'OR sonographic[Title/Abstract] OR imaging[Title/Abstract]))'
    ),
    "therapies": (
        '(("idiopathic granulomatous mastitis"[Title/Abstract] '
        'OR "granulomatous mastitis"[Title/Abstract]) '
        'AND (corticosteroid[Title/Abstract] OR steroid[Title/Abstract] '
        'OR methotrexate[Title/Abstract] OR antibiotic[Title/Abstract] '
        'OR surgery[Title/Abstract] OR excision[Title/Abstract] '
        'OR drainage[Title/Abstract] OR aspiration[Title/Abstract]))'
    ),
}


def load_pubmed_queries(config_path: Path) -> dict[str, str]:
    if not config_path.exists():
        return FALLBACK_QUERIES

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    query_config = config.get("queries", {})
    loaded: dict[str, str] = {}
    for key, value in query_config.items():
        if not isinstance(value, dict):
            continue
        pubmed_query = value.get("pubmed")
        if isinstance(pubmed_query, str) and pubmed_query.strip():
            loaded[key] = " ".join(pubmed_query.split())

    return loaded or FALLBACK_QUERIES


def request_xml(endpoint: str, params: dict[str, str], verify_ssl: bool = True) -> ET.Element:
    api_key = os.getenv("NCBI_API_KEY")
    email = os.getenv("NCBI_EMAIL")
    tool = "nonpuerperal-mastitis-rag"
    full_params = {"tool": tool, **params}
    if api_key:
        full_params["api_key"] = api_key
    if email:
        full_params["email"] = email

    url = f"{EUTILS_BASE}/{endpoint}?{urllib.parse.urlencode(full_params)}"
    context = None if verify_ssl else ssl._create_unverified_context()
    with urllib.request.urlopen(url, timeout=60, context=context) as response:
        body = response.read()
    return ET.fromstring(body)


def search_pmids(query: str, max_results: int, verify_ssl: bool = True) -> list[str]:
    root = request_xml(
        "esearch.fcgi",
        {
            "db": "pubmed",
            "term": query,
            "retmode": "xml",
            "retmax": str(max_results),
            "sort": "relevance",
        },
        verify_ssl=verify_ssl,
    )
    return [elem.text or "" for elem in root.findall(".//IdList/Id") if elem.text]


def fetch_records(pmids: list[str], verify_ssl: bool = True) -> list[LiteratureRecord]:
    if not pmids:
        return []
    root = request_xml(
        "efetch.fcgi",
        {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        },
        verify_ssl=verify_ssl,
    )
    return [parse_article(article) for article in root.findall(".//PubmedArticle")]


def text_at(parent: ET.Element, path: str) -> str:
    elem = parent.find(path)
    if elem is None or elem.text is None:
        return ""
    return " ".join(elem.text.split())


def collect_abstract(article: ET.Element) -> str:
    parts = []
    for elem in article.findall(".//Abstract/AbstractText"):
        label = elem.attrib.get("Label")
        text = " ".join("".join(elem.itertext()).split())
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts)


def article_id(article: ET.Element, id_type: str) -> str:
    for elem in article.findall(".//ArticleIdList/ArticleId"):
        if elem.attrib.get("IdType") == id_type:
            return elem.text or ""
    return ""


def parse_article(article: ET.Element) -> LiteratureRecord:
    pmid = text_at(article, ".//MedlineCitation/PMID")
    title = " ".join("".join(article.find(".//ArticleTitle").itertext()).split()) if article.find(".//ArticleTitle") is not None else ""
    journal = text_at(article, ".//Journal/Title")
    year = (
        text_at(article, ".//JournalIssue/PubDate/Year")
        or text_at(article, ".//ArticleDate/Year")
        or text_at(article, ".//DateCompleted/Year")
    )
    doi = article_id(article, "doi")
    pmcid = article_id(article, "pmc")
    abstract = collect_abstract(article)

    return LiteratureRecord(
        pmid=pmid,
        pmcid=pmcid,
        title=title,
        year=year,
        journal=journal,
        doi=doi,
        abstract=abstract,
        source_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        has_full_text="yes" if pmcid else "unknown",
    )


def write_csv(records: list[LiteratureRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(LiteratureRecord.csv_header())
        for record in records:
            writer.writerow(record.as_row())


def output_path_for_key(out: Path, query_key: str) -> Path:
    if out.suffix.lower() == ".csv":
        return out.with_name(f"{out.stem}_{query_key}{out.suffix}")
    return out / f"pubmed_{query_key}.csv"


def run_query(
    query_key: str,
    query: str,
    max_results: int,
    out_path: Path,
    sleep_seconds: float,
    verify_ssl: bool,
) -> int:
    pmids = search_pmids(query, max_results, verify_ssl=verify_ssl)
    time.sleep(sleep_seconds)
    records = fetch_records(pmids, verify_ssl=verify_ssl)
    write_csv(records, out_path)
    print(f"query_key={query_key}")
    print(f"records={len(records)}")
    print(f"out={out_path}")
    return len(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search PubMed metadata for the RAG registry.")
    parser.add_argument("--query-key", default="core_english")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every PubMed query defined in the query config.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/queries.yaml"),
        help="YAML file containing query definitions.",
    )
    parser.add_argument("--query", help="Custom PubMed query. Overrides --query-key.")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("data/registry/pubmed_results.csv"))
    parser.add_argument("--sleep", type=float, default=0.34, help="Delay between NCBI calls.")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL verification. Use only for local testing behind a trusted proxy.",
    )
    args = parser.parse_args(argv)

    verify_ssl = not args.insecure
    queries = load_pubmed_queries(args.config)

    if args.query:
        run_query(
            query_key="custom",
            query=args.query,
            max_results=args.max_results,
            out_path=args.out,
            sleep_seconds=args.sleep,
            verify_ssl=verify_ssl,
        )
        return 0

    if args.all:
        total = 0
        for query_key, query in queries.items():
            total += run_query(
                query_key=query_key,
                query=query,
                max_results=args.max_results,
                out_path=output_path_for_key(args.out, query_key),
                sleep_seconds=args.sleep,
                verify_ssl=verify_ssl,
            )
            time.sleep(args.sleep)
        print(f"total_records={total}")
        return 0

    if args.query_key not in queries:
        available = ", ".join(sorted(queries))
        raise SystemExit(f"Unknown query key: {args.query_key}. Available PubMed query keys: {available}")

    run_query(
        query_key=args.query_key,
        query=queries[args.query_key],
        max_results=args.max_results,
        out_path=args.out,
        sleep_seconds=args.sleep,
        verify_ssl=verify_ssl,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
