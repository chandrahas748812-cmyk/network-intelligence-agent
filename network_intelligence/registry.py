"""Model registry: versioned model configs + promotion.

A tiny registry (JSON file) tracking which model version is staging vs
production, its serving config, and its eval scores. Promoting a model
requires a recorded eval — a version with no eval result cannot become
production. This is the "model versioning" piece of the JD, kept
deliberately small: a real deployment would use MLflow or a model
registry service; the shape (version, config hash, eval gate, status)
is the same.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelVersion:
    version: str
    model: str  # base model id / path
    serving_config: dict
    status: str = "staging"  # staging | production | retired
    eval_root_cause_accuracy: float | None = None
    eval_evidence_score: float | None = None
    notes: str = ""

    def config_hash(self) -> str:
        raw = json.dumps(self.serving_config, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["config_hash"] = self.config_hash()
        return d


class ModelRegistry:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else Path(__file__).resolve().parent.parent / "data" / "model_registry.json"
        self.versions: dict[str, ModelVersion] = {}
        if self.path.exists():
            for item in json.loads(self.path.read_text()):
                self.versions[item["version"]] = ModelVersion(**{k: v for k, v in item.items() if k != "config_hash"})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([v.to_dict() for v in self.versions.values()], indent=2))

    def register(self, version: ModelVersion) -> None:
        self.versions[version.version] = version
        self.save()

    def get(self, version: str) -> ModelVersion | None:
        return self.versions.get(version)

    def production(self) -> ModelVersion | None:
        for v in self.versions.values():
            if v.status == "production":
                return v
        return None

    def record_eval(self, version: str, accuracy: float, evidence: float) -> None:
        v = self.versions[version]
        v.eval_root_cause_accuracy = accuracy
        v.eval_evidence_score = evidence
        self.save()

    def promote(self, version: str, min_accuracy: float = 0.75) -> ModelVersion:
        """Promote to production. Requires a recorded eval at or above
        ``min_accuracy`` — no eval, no promotion."""
        v = self.versions[version]
        if v.eval_root_cause_accuracy is None:
            raise ValueError(f"{version} has no recorded eval — run evals first")
        if v.eval_root_cause_accuracy < min_accuracy:
            raise ValueError(
                f"{version} accuracy {v.eval_root_cause_accuracy} < gate {min_accuracy}"
            )
        for other in self.versions.values():
            if other.status == "production":
                other.status = "retired"
        v.status = "production"
        self.save()
        return v
