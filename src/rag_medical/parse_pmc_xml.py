from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


SKIP_SECTION_KEYWORDS = {
    "acknowledgement",
    "acknowledgment",
    "author contribution",
    "authors' contributions",
    "availability of data",
    "conflict of interest",
    "competing interest",
    "data availability",
    "declaration",
    "ethical approval",
    "ethics approval",
    "funding",
    "informed consent",
    "references",
    "supplementary",
}


# ---------------------------------------------------------------------------
# XML namespace and text utilities
# ---------------------------------------------------------------------------
# PMC XML documents usually use JATS tags. Some files contain XML namespaces and
# some do not. These helper functions keep tag handling consistent so the rest
# of the parser can operate on local tag names such as "article-title", "sec",
# and "p".


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def normalize_space(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return normalize_space(" ".join(element.itertext()))


def first_text(root: ET.Element, path: str) -> str:
    return element_text(root.find(path))


def remove_citation_noise(text: str) -> str:
    # JATS inline citation text often appears as "[1]" or "(1, 2)" after
    # itertext(). Remove only simple citation-like fragments and leave clinical
    # numbers, percentages, and measurements intact.
    text = re.sub(r"\s*\[(?:\d+|,\s*|-|\s*)+\]", "", text)
    return normalize_space(text)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------
# Each output section carries article-level identifiers. This is important for
# RAG because every retrieved paragraph must remain traceable back to a PMID,
# PMCID, DOI, and source XML file.


def article_id(root: ET.Element, id_type: str) -> str:
    for elem in root.iter():
        if local_name(elem.tag) == "article-id" and elem.attrib.get("pub-id-type") == id_type:
            return normalize_space(elem.text)
    return ""


def publication_year(root: ET.Element) -> str:
    for pub_date in root.iter():
        if local_name(pub_date.tag) != "pub-date":
            continue
        year = pub_date.find("./year")
        if year is not None and year.text:
            return normalize_space(year.text)
        for child in pub_date:
            if local_name(child.tag) == "year" and child.text:
                return normalize_space(child.text)
    return ""


def journal_title(root: ET.Element) -> str:
    for elem in root.iter():
        if local_name(elem.tag) in {"journal-title", "journal-id"}:
            text = element_text(elem)
            if text:
                return text
    return ""


def article_title(root: ET.Element) -> str:
    for elem in root.iter():
        if local_name(elem.tag) == "article-title":
            return element_text(elem)
    return ""


def extract_metadata(root: ET.Element, source_path: Path) -> dict[str, str]:
    return {
        "pmcid": article_id(root, "pmc") or source_path.stem,
        "pmid": article_id(root, "pmid"),
        "doi": article_id(root, "doi"),
        "title": article_title(root),
        "journal": journal_title(root),
        "year": publication_year(root),
        "source_path": str(source_path),
    }


# ---------------------------------------------------------------------------
# Section title and filtering rules
# ---------------------------------------------------------------------------
# The parser preserves section names such as Abstract, Methods, Results, and
# Discussion. It filters out sections that are usually not helpful for clinical
# RAG answers, such as references, acknowledgements, funding, and conflict
# statements.


def direct_child(element: ET.Element, tag_name: str) -> ET.Element | None:
    for child in element:
        if local_name(child.tag) == tag_name:
            return child
    return None


def section_title(sec: ET.Element) -> str:
    title = direct_child(sec, "title")
    return element_text(title)


def should_skip_section(title: str) -> bool:
    normalized = title.lower()
    return any(keyword in normalized for keyword in SKIP_SECTION_KEYWORDS)


def body_element(root: ET.Element) -> ET.Element | None:
    for elem in root.iter():
        if local_name(elem.tag) == "body":
            return elem
    return None


def abstract_elements(root: ET.Element) -> list[ET.Element]:
    abstracts = []
    for elem in root.iter():
        if local_name(elem.tag) == "abstract":
            abstracts.append(elem)
    return abstracts


# ---------------------------------------------------------------------------
# Paragraph extraction
# ---------------------------------------------------------------------------
# XML structure is nested. This module extracts paragraph-level records while
# keeping their section path. Paragraphs are intentionally smaller than whole
# sections because the next RAG step will chunk them into retrieval units.


def paragraph_text(paragraph: ET.Element) -> str:
    return remove_citation_noise(element_text(paragraph))


def paragraph_records_from_container(
    container: ET.Element,
    metadata: dict[str, str],
    section_path: list[str],
    min_chars: int,
) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    paragraph_index = 0

    for child in container:
        tag = local_name(child.tag)
        if tag == "p":
            text = paragraph_text(child)
            if len(text) >= min_chars:
                paragraph_index += 1
                section = " > ".join(section_path) if section_path else "Body"
                records.append(
                    {
                        **metadata,
                        "section": section,
                        "paragraph_index": paragraph_index,
                        "text": text,
                        "char_count": len(text),
                    }
                )
        elif tag in {"list", "boxed-text"}:
            text = paragraph_text(child)
            if len(text) >= min_chars:
                paragraph_index += 1
                section = " > ".join(section_path) if section_path else "Body"
                records.append(
                    {
                        **metadata,
                        "section": section,
                        "paragraph_index": paragraph_index,
                        "text": text,
                        "char_count": len(text),
                    }
                )

    return records


def walk_sections(
    sec: ET.Element,
    metadata: dict[str, str],
    parent_path: list[str],
    min_chars: int,
) -> list[dict[str, str | int]]:
    title = section_title(sec)
    if should_skip_section(title):
        return []

    current_path = parent_path + ([title] if title else [])
    records = paragraph_records_from_container(sec, metadata, current_path, min_chars)

    for child in sec:
        if local_name(child.tag) == "sec":
            records.extend(walk_sections(child, metadata, current_path, min_chars))

    return records


# ---------------------------------------------------------------------------
# Article parsing
# ---------------------------------------------------------------------------
# One PMC XML file can contain abstract and body sections. This function parses
# both, returns paragraph records for JSONL output, and leaves failures to the
# caller so a manifest can record which files need attention.


def parse_article_xml(path: Path, min_chars: int) -> list[dict[str, str | int]]:
    root = ET.parse(path).getroot()
    metadata = extract_metadata(root, path)
    records: list[dict[str, str | int]] = []

    for abstract in abstract_elements(root):
        if should_skip_section("Abstract"):
            continue
        records.extend(paragraph_records_from_container(abstract, metadata, ["Abstract"], min_chars))

    body = body_element(root)
    if body is None:
        return records

    for child in body:
        if local_name(child.tag) == "sec":
            records.extend(walk_sections(child, metadata, [], min_chars))
        elif local_name(child.tag) == "p":
            records.extend(paragraph_records_from_container(body, metadata, ["Body"], min_chars))
            break

    return records


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
# JSONL stores the parsed text for downstream chunking and embedding. Manifest
# CSV stores one row per XML file so failed or low-yield parses are easy to
# audit without opening every article.


def write_jsonl(records: list[dict[str, str | int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_manifest(rows: list[dict[str, str | int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_path",
        "pmcid",
        "pmid",
        "title",
        "status",
        "paragraph_count",
        "error",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_directory(
    xml_dir: Path,
    out_path: Path,
    manifest_path: Path,
    min_chars: int,
    limit: int | None,
) -> tuple[int, int]:
    xml_paths = sorted(xml_dir.glob("PMC*.xml"))
    if limit is not None:
        xml_paths = xml_paths[:limit]

    all_records: list[dict[str, str | int]] = []
    manifest_rows: list[dict[str, str | int]] = []
    failed = 0

    for index, path in enumerate(xml_paths, start=1):
        try:
            records = parse_article_xml(path, min_chars=min_chars)
            all_records.extend(records)
            metadata = records[0] if records else extract_metadata(ET.parse(path).getroot(), path)
            manifest_rows.append(
                {
                    "source_path": str(path),
                    "pmcid": metadata.get("pmcid", ""),
                    "pmid": metadata.get("pmid", ""),
                    "title": metadata.get("title", ""),
                    "status": "parsed",
                    "paragraph_count": len(records),
                    "error": "",
                }
            )
            print(f"[{index}/{len(xml_paths)}] {path.name}: parsed {len(records)} paragraphs")
        except Exception as exc:  # noqa: BLE001 - record per-file parse failures.
            failed += 1
            manifest_rows.append(
                {
                    "source_path": str(path),
                    "pmcid": path.stem,
                    "pmid": "",
                    "title": "",
                    "status": "failed",
                    "paragraph_count": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[{index}/{len(xml_paths)}] {path.name}: failed", file=sys.stderr)

    write_jsonl(all_records, out_path)
    write_manifest(manifest_rows, manifest_path)
    return len(all_records), failed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse PMC XML full text into paragraph-level JSONL.")
    parser.add_argument("--xml-dir", type=Path, default=Path("data/articles/raw/pmc_xml"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/articles/processed/article_sections.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/articles/processed/article_parse_manifest.csv"),
    )
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--limit", type=int, help="Parse only the first N XML files.")
    args = parser.parse_args(argv)

    records, failed = parse_directory(
        xml_dir=args.xml_dir,
        out_path=args.out,
        manifest_path=args.manifest,
        min_chars=args.min_chars,
        limit=args.limit,
    )
    print(f"records={records}")
    print(f"failed={failed}")
    print(f"out={args.out}")
    print(f"manifest={args.manifest}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
