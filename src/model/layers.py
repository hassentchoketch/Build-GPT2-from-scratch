# MLP and LayerNorm components
import torch
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # First linear layer: 768 -> 3072
        self.c_fc = nn.Linear(config.model.n_embd, 4 * config.model.n_embd)

        # GELU activation: GPT-2 uses the 'tanh'approximation
        # approximate="tanh" is used to match the original OpenAI weights exactly
        self.gelu = nn.GELU(approximate="tanh")

        # Second linear layer :3072 ->768
        self.c_proj = nn.Linear(4 * config.model.n_embd, config.model.n_embd)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.model.dropout)

    def forward(self, x):
        # 1. Expand
        x = self.c_fc(x)
        # 2. Activate
        x = self.gelu(x)
        # 3. Project back
        x = self.c_proj(x)
        # 4. Dropout
        x = self.dropout(x)
        return x
