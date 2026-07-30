"""
用途：根据文献 registry 中的 PMCID 下载 PMC Open Access XML 全文。
输入：data/registry/processed/literature_registry.csv。
输出：data/articles/raw/pmc_xml/*.xml 与 pmc_download_manifest.csv。
说明：只有存在 PMCID 且可公开访问的文章才能下载到全文 XML。
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
from pathlib import Path


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def normalize(value: str | None) -> str:
    return (value or "").strip()


def normalize_pmcid(value: str | None) -> str:
    pmcid = normalize(value).upper()
    if not pmcid:
        return ""
    if pmcid.startswith("PMC"):
        return pmcid
    if pmcid.isdigit():
        return f"PMC{pmcid}"
    return pmcid


def pmc_numeric_id(pmcid: str) -> str:
    return pmcid.upper().removeprefix("PMC")


def read_registry(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [{key: normalize(value) for key, value in row.items()} for row in csv.DictReader(f)]


def request_pmc_xml(pmcid: str, verify_ssl: bool = True) -> bytes:
    params = {
        "db": "pmc",
        "id": pmc_numeric_id(pmcid),
        "retmode": "xml",
        "tool": "nonpuerperal-mastitis-rag",
    }
    api_key = os.getenv("NCBI_API_KEY")
    email = os.getenv("NCBI_EMAIL")
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email

    url = f"{EUTILS_BASE}/efetch.fcgi?{urllib.parse.urlencode(params)}"
    context = None if verify_ssl else ssl._create_unverified_context()
    request = urllib.request.Request(url, headers={"User-Agent": "nonpuerperal-mastitis-rag/0.1"})
    with urllib.request.urlopen(request, timeout=90, context=context) as response:
        return response.read()


def looks_like_article_xml(payload: bytes) -> bool:
    head = payload[:500].decode("utf-8", errors="ignore").lower()
    return "<pmc-articleset" in head or "<article" in head


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pmid",
        "pmcid",
        "title",
        "year",
        "journal",
        "doi",
        "source_queries",
        "status",
        "xml_path",
        "bytes",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def fetch_articles(
    registry_path: Path,
    out_dir: Path,
    manifest_path: Path,
    limit: int | None,
    sleep_seconds: float,
    overwrite: bool,
    verify_ssl: bool,
) -> tuple[int, int, int]:
    rows = read_registry(registry_path)
    candidates = [row for row in rows if normalize_pmcid(row.get("pmcid"))]
    if limit is not None:
        candidates = candidates[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    downloaded = 0
    skipped = 0
    failed = 0

    for index, row in enumerate(candidates, start=1):
        pmcid = normalize_pmcid(row.get("pmcid"))
        xml_path = out_dir / f"{pmcid}.xml"
        manifest_row = {
            "pmid": row.get("pmid", ""),
            "pmcid": pmcid,
            "title": row.get("title", ""),
            "year": row.get("year", ""),
            "journal": row.get("journal", ""),
            "doi": row.get("doi", ""),
            "source_queries": row.get("source_queries", ""),
            "xml_path": str(xml_path),
            "bytes": "",
            "error": "",
        }

        if xml_path.exists() and not overwrite:
            manifest_row["status"] = "skipped_existing"
            manifest_row["bytes"] = str(xml_path.stat().st_size)
            manifest_rows.append(manifest_row)
            skipped += 1
            continue

        try:
            payload = request_pmc_xml(pmcid, verify_ssl=verify_ssl)
            if not looks_like_article_xml(payload):
                raise ValueError("response does not look like PMC article XML")
            xml_path.write_bytes(payload)
            manifest_row["status"] = "downloaded"
            manifest_row["bytes"] = str(len(payload))
            downloaded += 1
        except Exception as exc:  # noqa: BLE001 - keep manifest informative.
            manifest_row["status"] = "failed"
            manifest_row["error"] = f"{type(exc).__name__}: {exc}"
            failed += 1

        manifest_rows.append(manifest_row)
        print(f"[{index}/{len(candidates)}] {pmcid}: {manifest_row['status']}")
        time.sleep(sleep_seconds)

    write_manifest(manifest_path, manifest_rows)
    return downloaded, skipped, failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download PMC XML full text from a literature registry.")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/registry/processed/literature_registry.csv"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/articles/raw/pmc_xml"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/articles/processed/pmc_download_manifest.csv"),
    )
    parser.add_argument("--limit", type=int, help="Download only the first N PMCID records.")
    parser.add_argument("--sleep", type=float, default=0.34, help="Delay between NCBI requests.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download XML files that already exist.")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL verification. Use only for local testing behind a trusted proxy.",
    )
    args = parser.parse_args(argv)

    downloaded, skipped, failed = fetch_articles(
        registry_path=args.registry,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        limit=args.limit,
        sleep_seconds=args.sleep,
        overwrite=args.overwrite,
        verify_ssl=not args.insecure,
    )
    print(f"downloaded={downloaded}")
    print(f"skipped_existing={skipped}")
    print(f"failed={failed}")
    print(f"manifest={args.manifest}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

