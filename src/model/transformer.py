import yaml
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024  # Maximum sequence length (Context windows)
    vocab_size: int = 50257  # Number of tokens in GPT-2's vocabulary
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Number of attention heads
    n_embd: int = 768  # Embedding dimention (hidden size)
    dropout: float = 0.1  # Dropout probability
    bias: bool = True  # whether to use bias in Linear layers ( True for GPT-2)

    @classmethod
    def from_yaml(cls, config_path: str):
        """Loads config from a YAML file and returns a GPTConfig instance."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # We look for the 'model' key in our YAML
        return cls(**config_dict["model"])
