"""Minimal in-memory tasks for same-document evidence completion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..coverage_models import CoverageInputPackage

if TYPE_CHECKING:
    from ..assertion_extract import AssertionExtractionResult


@dataclass(frozen=True)
class DocumentCompletionTask:
    """One source document, not a guarantee of a single unmixed study."""

    query_id: str
    question_text: str
    document_id: str | None  # None keeps unresolved evidence separate.
    evidence_packages: list[CoverageInputPackage]
    assertion_results: dict[str, AssertionExtractionResult]  # Audit hints only.
    input_warnings: list[str]


@dataclass(frozen=True)
class DocumentChunk:
    """An unchanged source chunk with its original metadata and seed links."""

    document_id: str | None
    chunk_id: str
    section: str
    text: str
    source_metadata: dict[str, Any]
    metadata_path: Path | None  # None means only the existing hit was available.
    seed_evidence_ids: list[str]  # Includes existing R1 source-chunk links.


@dataclass(frozen=True)
class DocumentMaterials:
    """Source-level material collection; study attribution is not verified."""

    task: DocumentCompletionTask
    chunks: list[DocumentChunk]
    ordering_note: str
    warnings: list[str]


@dataclass(frozen=True)
class SelectionParameters:
    """Complete per-chunk chat budgets; document token counts sum its calls."""

    max_input_tokens: int
    max_new_tokens: int = 2048


@dataclass(frozen=True)
class SelectedPassage:
    """One selectable original sentence/fragment; offsets are [start, end)."""

    passage_id: str
    chunk_id: str
    start: int
    end: int
    text: str
    is_initial: bool
    source_fragment: bool


class ContentType(str, Enum):
    DESIGN = "DESIGN"
    METHOD = "METHOD"
    RESULT = "RESULT"
    CONCLUSION = "CONCLUSION"
    OTHER = "OTHER"


@dataclass(frozen=True)
class ClassificationCall:
    chunk_id: str
    raw_output: str
    input_token_count: int
    output_token_count: int


@dataclass(frozen=True)
class DocumentSelection:
    """Four content classes are included; OTHER remains an auditable reserve."""

    task: DocumentCompletionTask
    supplements: list[SelectedPassage]
    reserve: list[SelectedPassage]
    selected_initial_ids: list[str]
    passage_types: dict[str, ContentType]
    classification_calls: list[ClassificationCall]
    input_token_count: int
    output_token_count: int
    prompt_version: str
    warnings: list[str]
