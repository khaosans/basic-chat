import json
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "evaluator.config.json"

class LLMJudgeConsistency:
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = config_path
        self._config = None
        self._load_config()

    def _load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = json.load(f)

    @property
    def version(self):
        return self._config.get("version", "0.0.1")

    @property
    def threshold(self):
        return self._config.get("threshold", 7.0)

    @property
    def evaluation_categories(self):
        return self._config.get("evaluation_categories", {})

    def get_rubric(self, category):
        cat = self.evaluation_categories.get(category, {})
        return cat.get("rubric", {})

    def get_weight(self, category):
        cat = self.evaluation_categories.get(category, {})
        return cat.get("weight", 1.0)

    def get_description(self, category):
        cat = self.evaluation_categories.get(category, {})
        return cat.get("description", "")

    def all_categories(self):
        return list(self.evaluation_categories.keys())

    def rubric_text(self):
        """Return a markdown string of all rubrics for prompt injection."""
        lines = [f"## LLM Judge Rubric v{self.version}"]
        for cat, data in self.evaluation_categories.items():
            lines.append(f"### {cat.replace('_', ' ').title()} ({data.get('weight', 1.0)})")
            lines.append(f"- {data.get('description', '')}")
            rubric = data.get('rubric', {})
            for score, desc in rubric.items():
                lines.append(f"  - **{score}:** {desc}")
            lines.append("")
        return "\n".join(lines) 