
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class PromptConfig:
    version: str
    description: str
    model: str
    temperature: float
    system_prompt: str


def load_prompt_config(file_path: str) -> PromptConfig:
    with open(file_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return PromptConfig(**data)
