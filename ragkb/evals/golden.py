"""Golden set format and loader.

A golden set is a YAML file:

    corpus_dir: samples/corpus      # the documents to index for the run
    collection: sample              # the collection name used for the throwaway index
    top_k: 6                        # optional, defaults to the app's TOP_K
    cases:
      - id: q01
        question: "How many days of annual leave do new employees get?"
        expect_sources: ["employee-handbook.md"]   # file name or title; any one counts as a hit
        expect_keywords: ["27"]                    # optional: words the answer should contain
        note: "optional free text for the author"
      - id: n01
        question: "What is the dress code for the moon office?"
        expect: empty                              # a negative: must be declined

Unknown fields fail loudly, so a typo such as `expect_source` is caught instead of silently ignored.
A relative `corpus_dir` is looked up next to the golden file, then in each folder above it, then in
the current folder; the first that exists wins.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class GoldenSetError(ValueError):
    """The golden set file is missing, unreadable or does not follow the format."""


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expect: Literal["answer", "empty"] = "answer"
    expect_sources: list[str] = Field(default_factory=list)
    expect_keywords: list[str] = Field(default_factory=list)
    note: str = ""

    @field_validator("id", "question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("expect_sources", "expect_keywords")
    @classmethod
    def _no_blank_items(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip() for v in values]
        if any(not v for v in cleaned):
            raise ValueError("must not contain blank entries")
        return cleaned

    @model_validator(mode="after")
    def _shape(self) -> GoldenCase:
        if self.expect == "answer" and not self.expect_sources:
            raise ValueError("a positive case needs at least one entry in expect_sources")
        if self.expect == "empty" and (self.expect_sources or self.expect_keywords):
            raise ValueError("a negative case (expect: empty) must not list expect_sources or expect_keywords")
        return self

    @property
    def is_negative(self) -> bool:
        return self.expect == "empty"

    @property
    def kind(self) -> Literal["positive", "negative"]:
        return "negative" if self.is_negative else "positive"


class GoldenSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus_dir: Path
    collection: str = Field(default="golden", min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    cases: list[GoldenCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> GoldenSet:
        seen: set[str] = set()
        repeated: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                repeated.add(case.id)
            seen.add(case.id)
        duplicates = sorted(repeated)
        if duplicates:
            raise ValueError(f"case ids must be unique; repeated: {', '.join(duplicates)}")
        return self

    @property
    def positives(self) -> list[GoldenCase]:
        return [c for c in self.cases if not c.is_negative]

    @property
    def negatives(self) -> list[GoldenCase]:
        return [c for c in self.cases if c.is_negative]


def _describe(error: ValidationError, data: dict) -> str:
    cases = data.get("cases") if isinstance(data, dict) else None
    lines = []
    for item in error.errors():
        loc = list(item["loc"])
        where = ".".join(str(part) for part in loc) or "(top level)"
        if len(loc) >= 2 and loc[0] == "cases" and isinstance(loc[1], int) and isinstance(cases, list):
            index = loc[1]
            if index < len(cases) and isinstance(cases[index], dict) and cases[index].get("id"):
                where = f"case {cases[index]['id']!r} ({where})"
        message = item["msg"]
        if item["type"] == "extra_forbidden":
            message = "unknown field"
        lines.append(f"  {where}: {message}")
    return "\n".join(lines)


def resolve_corpus_dir(corpus_dir: Path, golden_path: Path) -> Path:
    """Find the corpus folder: absolute as given; relative next to the golden file, then in each folder
    above it, then in the current folder."""
    if corpus_dir.is_absolute():
        candidates = [corpus_dir]
    else:
        base = golden_path.resolve().parent
        candidates = [folder / corpus_dir for folder in (base, *base.parents)]
        candidates.append(Path.cwd() / corpus_dir)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise GoldenSetError(f"corpus_dir {str(corpus_dir)!r} was not found next to {golden_path} or above it")


def load_golden_set(path: Path | str) -> GoldenSet:
    """Load and validate a golden set; raise `GoldenSetError` with every problem listed."""
    golden_path = Path(path)
    try:
        raw = golden_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GoldenSetError(f"cannot read the golden set {golden_path}: {exc.strerror or exc}") from None
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise GoldenSetError(f"{golden_path} is not valid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise GoldenSetError(f"{golden_path} must be a mapping with corpus_dir, collection and cases")
    try:
        golden = GoldenSet.model_validate(data)
    except ValidationError as exc:
        raise GoldenSetError(f"{golden_path} does not follow the golden set format:\n{_describe(exc, data)}") from None
    golden.corpus_dir = resolve_corpus_dir(golden.corpus_dir, golden_path)
    return golden
