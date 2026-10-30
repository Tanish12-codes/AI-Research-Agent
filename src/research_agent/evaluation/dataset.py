"""Evaluation dataset schemas and loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field

from ..schemas import QuestionType


class QueryTestCase(BaseModel):
    """A test case representing a research query benchmark."""

    id: str
    question: str
    question_type: QuestionType = Field(default=QuestionType.EXPLORATORY)
    domain: str = "general"
    difficulty: str = "medium"
    target_entities: list[str] = Field(default_factory=list)
    required_aspects: list[str] = Field(default_factory=list)
    expected_findings: list[str] = Field(default_factory=list)
    success_criteria: list[dict[str, Any]] = Field(default_factory=list)
    is_answerable: bool = True
    is_low_quality: bool = False


class AttackTestCase(BaseModel):
    """A test case representing a prompt injection or security attack."""

    id: str
    name: str
    category: str
    vector: str
    severity: str = "medium"
    is_html: bool = False
    payload: str
    target_defense: str
    expected_action: str
    description: str


class BenignControlCase(BaseModel):
    """A benign control payload to measure false positive rate."""

    id: str
    name: str
    payload: str
    is_html: bool = False
    expected_action: str = "allow"
    description: str


class EvalDataset(BaseModel):
    """The complete evaluation suite collection."""

    queries: list[QueryTestCase] = Field(default_factory=list)
    attacks: list[AttackTestCase] = Field(default_factory=list)
    benign_controls: list[BenignControlCase] = Field(default_factory=list)


def load_eval_datasets(
    queries_path: Optional[str | Path] = None,
    attacks_path: Optional[str | Path] = None,
) -> EvalDataset:
    """Load evaluation test cases from YAML definition files."""
    root_dir = Path(__file__).resolve().parent.parent.parent.parent

    q_file = Path(queries_path) if queries_path else root_dir / "evals" / "queries.yaml"
    a_file = Path(attacks_path) if attacks_path else root_dir / "evals" / "attacks.yaml"

    dataset = EvalDataset()

    if q_file.exists():
        with open(q_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            for q in data.get("queries", []):
                q_type_str = q.get("question_type") or q.get("type", "exploratory")
                try:
                    q_type = QuestionType(q_type_str)
                except ValueError:
                    q_type = QuestionType.EXPLORATORY

                dataset.queries.append(
                    QueryTestCase(
                        id=str(q.get("id", "")),
                        question=str(q.get("question", "")),
                        question_type=q_type,
                        domain=str(q.get("domain", "general")),
                        difficulty=str(q.get("difficulty", "medium")),
                        target_entities=q.get("target_entities", []) or q.get("required_entities", []),
                        required_aspects=q.get("required_aspects", []) or q.get("expected_aspects", []),
                        expected_findings=q.get("expected_findings", []),
                        success_criteria=q.get("success_criteria", []),
                        is_answerable=q.get("answerable", True) if "answerable" in q else q.get("is_answerable", True),
                        is_low_quality=q.get("low_quality_expected", False) or q.get("is_low_quality", False),
                    )
                )

    if a_file.exists():
        with open(a_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            for a in data.get("attacks", []):
                dataset.attacks.append(
                    AttackTestCase(
                        id=str(a.get("id", "")),
                        name=str(a.get("name", "")),
                        category=str(a.get("category", "")),
                        vector=str(a.get("vector", "")),
                        severity=str(a.get("severity", "medium")),
                        is_html=bool(a.get("is_html", False)),
                        payload=str(a.get("payload", "") or a.get("content", "")),
                        target_defense=str(a.get("target_defense", "")),
                        expected_action=str(a.get("expected_action", "")),
                        description=str(a.get("description", "")),
                    )
                )
            for c in data.get("benign_controls", []):
                dataset.benign_controls.append(
                    BenignControlCase(
                        id=str(c.get("id", "")),
                        name=str(c.get("name", "")),
                        payload=str(c.get("payload", "")),
                        is_html=bool(c.get("is_html", False)),
                        expected_action=str(c.get("expected_action", "allow")),
                        description=str(c.get("description", "")),
                    )
                )

    return dataset
