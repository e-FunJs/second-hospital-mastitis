from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class LiteratureRecord:
    pmid: str = ""
    pmcid: str = ""
    title: str = ""
    year: str = ""
    journal: str = ""
    doi: str = ""
    abstract: str = ""
    source_url: str = ""
    disease_type: str = ""
    topic: str = ""
    study_type: str = ""
    evidence_level: str = ""
    has_full_text: str = ""

    @classmethod
    def csv_header(cls) -> list[str]:
        return [field.name for field in fields(cls)]

    def as_row(self) -> list[str]:
        return [getattr(self, name) for name in self.csv_header()]

