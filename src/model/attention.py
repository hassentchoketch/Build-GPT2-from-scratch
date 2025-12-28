import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.model.n_embd % config.model.n_head == 0
        # Key, Query, Value projection for all heads but in a batch
        self.c_attn = nn.Linear(config.model.n_embd, 3 * config.model.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.model.n_embd, config.model.n_embd)

        # Regularization
        self.n_head = config.model.n_head
        self.n_embd = config.model.n_embd

        # Causal mask: ensures that attention can only move to the left
        # We register it as a 'bafer' so it's moved to GPU automatically but not trained
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(config.model.block_size, config.model.block_size)
            ).view(1, 1, config.model.block_size, config.model.block_size),
        )

    def forward(self, x):
        B, T, C = x.size()
        # batch size, sequence length (block size), embedding dimensionality (n_embd)
        print(B, T, C)
        # Calculate query, key, values for all heads in batch
        # nh is "number of heads", hs is "head size", and C = nh * hs
        qkv = self.c_attn(x)
        # print(qkv.size())
        q, k, v = qkv.split(self.n_embd, dim=2)
        # print("q\n", q.shape, "k\n", k.shape, "v\n", v.shape)

        # Reshape to (B,nh,T,hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # print("q\n", q.shape, "k\n", k.shape, "v\n", v.shape)
        # Causal self-attantion: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        # Apply attention to values
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assenble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.c_proj(y)

        return y
