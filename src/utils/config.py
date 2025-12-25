import yaml
from src.model.transformer import GPTConfig
from src.dataset import DataConfig

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    model: GPTConfig
    data: DataConfig

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            raw_yaml = yaml.safe_load(f)

        return cls(
            model=GPTConfig(**raw_yaml["model"]), data=DataConfig(**raw_yaml["data"])
        )
