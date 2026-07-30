# Second Hospital Mastitis Project

This repository contains the RAG knowledge base and the later multimodal modelling
work for the Second Hospital non-puerperal mastitis project.

The first-stage goal is not to predict patient outcomes directly. The goal is to
collect traceable external medical knowledge that can support:

- candidate feature design for hospital data collection;
- endpoint and treatment-response definition;
- structured extraction from ultrasound reports and clinical notes;
- evidence-grounded model explanations.

## Project Path

Expected server location:

```bash
/home/amax/E-FUN/second-hospital-mastitis
```

```bash
cd /home/amax/E-FUN/second-hospital-mastitis
```

## Environment

The intended conda environment is `hospital`.

```bash
conda activate hospital
python -m pip install -e .
```

Dependencies are listed in `pyproject.toml`.

The Chinese OCR pipeline additionally requires the system commands `pdfinfo`,
`pdftotext`, `pdftoppm` and Tesseract with `chi_sim` and `eng` language data.

## Code Layout

- `src/rag_medical/english/`: PubMed/PMC acquisition, XML parsing, English
  semantic chunking, strict medical filtering and animal-literature filtering.
- `src/rag_medical/chinese/`: CNKI PDF text extraction, Chinese OCR,
  section-aware semantic chunking and Chinese medical filtering.
- `src/rag_medical/common/`: language-independent embedding, FAISS indexing,
  corpus validation, retrieval, answer generation and answer evaluation.
- `scripts/english/`: entry points for the English literature pipeline.
- `scripts/chinese/`: entry points for the Chinese literature pipeline.
- `scripts/rag/`: shared embedding, retrieval, generation and evaluation entry points.
- `scripts/setup/`: server environment setup.
- `scripts/legacy/`: obsolete scripts retained only for historical reference.

## Data Layout

```text
data/
├── articles/
│   ├── raw/
│   │   ├── english/pmc_xml/
│   │   └── chinese/cnki_pdf/
│   └── processed/
│       ├── english/
│       ├── chinese/
│       └── combined/
├── registry/
│   ├── english/
│   └── chinese/
└── index/
    ├── english/{broad,strict}/
    ├── chinese/strict/
    └── combined/
```

Raw literature, processed text, embeddings and indexes remain local and are not
committed to Git.

## Day 1 Deliverables

- `RAG_SCOPE.md`: scope, inclusion/exclusion rules, evidence priorities.
- `configs/sources.yaml`: source registry.
- `configs/queries.yaml`: search terms.
- `configs/embedding.yaml`: planned embedding and index settings.
- `src/rag_medical/english/search_pubmed.py`: minimal PubMed metadata search script.
- `outputs/data_availability_checklist.md`: variable mapping and data availability checklist.
- `outputs/endpoint_definitions.md`: initial endpoint framework.

## Quick Start

Search PubMed metadata for every PubMed query defined in `configs/queries.yaml`:

```bash
conda activate hospital
bash scripts/english/run_pubmed_searches.sh 100
```

This creates short, stable CSV files by query type:

```text
data/registry/english/raw/core.csv
data/registry/english/raw/outcome.csv
data/registry/english/raw/ultrasound.csv
data/registry/english/raw/therapy.csv
```

You can inspect or edit the configured queries in `configs/queries.yaml`.

Merge and deduplicate the raw registry files:

```bash
bash scripts/english/merge_registry.sh
```

This creates:

```text
data/registry/english/processed/literature_registry.csv
data/registry/english/processed/literature_registry_summary.md
```

Download PMC XML full text for records with PMCID:

```bash
bash scripts/english/fetch_pmc_articles.sh
```

For a small smoke test:

```bash
bash scripts/english/fetch_pmc_articles.sh 3
```

This creates:

```text
data/articles/raw/english/pmc_xml/PMC*.xml
data/articles/processed/english/pmc_download_manifest.csv
```

Parse downloaded PMC XML into paragraph-level article sections:

```bash
bash scripts/english/parse_pmc_xml.sh
```

For a small smoke test:

```bash
bash scripts/english/parse_pmc_xml.sh 3
```

This creates:

```text
data/articles/processed/english/article_sections.jsonl
data/articles/processed/english/article_parse_manifest.csv
```

## Chinese Literature

Parse CNKI PDF pages, using Chinese OCR only when a page has no usable text layer:

```bash
bash scripts/chinese/parse_pdfs.sh
```

The parser processes eight PDFs concurrently by default. On a smaller machine,
reduce the load with `CNKI_PARSE_WORKERS=2 bash scripts/chinese/parse_pdfs.sh`.

Build section-aware semantic chunks with the local BGE-M3 model:

```bash
bash scripts/chinese/semantic_chunk.sh
```

Apply the auditable Chinese medical strict/review/excluded filter:

```bash
bash scripts/chinese/filter_corpus.sh
```

Check that the strict Chinese chunks are ready for the shared embedding module,
without generating embeddings or an index:

```bash
bash scripts/rag/check_corpus.sh
```

## Recommended Workflow

1. Use `rag_medical.english.search_pubmed` to create the English literature registry.
2. Manually screen titles/abstracts for disease relevance and treatment relevance.
3. Prioritize guidelines, expert consensus, systematic reviews, meta-analyses,
   and clinical cohorts.
4. Store only legally accessible abstracts or open-access full text.
5. Use the screened literature to refine hospital data fields and endpoint
   definitions before receiving patient-level data.
