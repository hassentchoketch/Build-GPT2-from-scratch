import yaml
import torch
import torch.nn as nn
from dataclasses import dataclass
from src.model.attention import CausalSelfAttention
from src.model.layers import MLP
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # Maximum sequence length (Context windows)
    vocab_size: int = 50257  # Number of tokens in GPT-2's vocabulary
    n_layer: int = 12  # Number of transformer blocks
    n_head: int = 12  # Number of attention heads
    n_embd: int = 768  # Embedding dimention (hidden size)
    dropout: float = 0.1  # Dropout probability
    bias: bool = True  # whether to use bias in Linear layers ( True for GPT-2)
    max_new_token: int = 20
    temperature: float = 0.8
    top_k: int = 4

    @classmethod
    def from_yaml(cls, config_path: str):
        """Loads config from a YAML file and returns a GPTConfig instance."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        # We look for the 'model' key in our YAML
        return cls(**config_dict["model"])


class BLock(nn.Module):
    def __init__(self, config):
        super().__init__()

        # LayerNorm 1: Before Attention
        self.ln_1 = nn.LayerNorm(config.model.n_embd)
        self.attn = CausalSelfAttention(config)

        # LayerNorm 2: Before MLP
        self.ln_2 = nn.LayerNorm(config.model.n_embd)
        self.mpl = MLP(config)

    def forward(self, x):

        # 1. Attention path with Residual Connection
        # x = x + Attention(layerNorm(x))
        x = x + self.attn(self.ln_1(x))

        # 2. MLP path with Residual Connection
        # x = x + MLP(LayerNOrm(x))
        x = x + self.mpl(self.ln_2(x))

        return x


class GPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # The ModuleDict keeps thing organized and visible to PyTorch
        self.transformer = nn.ModuleDict(
            dict(
                # wte = weight Token Embedding (vocab Size --> Hidden Dim)
                wte=nn.Embedding(
                    self.config.model.vocab_size, self.config.model.n_embd
                ),
                # wpe = weight Positional Embedding(Block Size --> Hidden Dim)
                wpe=nn.Embedding(
                    self.config.model.block_size, self.config.model.n_embd
                ),
                # h = The hidden layers ( we will add these next!)
                h=nn.ModuleList(
                    [nn.Identity() for _ in range(self.config.model.n_layer)]
                ),
                # ln_f = Final LayerNorm before the output head
                ln_f=nn.LayerNorm(self.config.model.n_embd),
            )
        )
        # The Output Head (maps hidden dim back to vocabulary)
        self.lm_head = nn.Linear(
            self.config.model.n_embd,
            self.config.model.vocab_size,
            self.config.model.bias,
        )

        # Weight Sharing (Weight Tying)
        # GPT-2 Share weights between the embeding and the output head
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        # 1 Create position indicies: [0, 1, ....,t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(
            0
        )  # shape (1, t)

        # 2 get embeddings
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(
            pos
        )  # position embeddings of shape (1, t, n_embd)

        # 3. Combine them!
        # (The pos_emb will be broadcasted across the batch dimension 'b')
        x = tok_emb + pos_emb

        # Pass through all transformer blocks
        for block in self.transformer.h:
            x = block(x)

        # Final LayerNorm and Projection to Vocab
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Calculate loss if targer are provided
        loss = None
        if targets is not None:

            # flaten logit and target for CrossEntropy Loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx):
        """Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        for _ in range(self.config.model.max_new_token):
            # if the sequence context is growing too leng we must crop it at block size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.model.block_size
                else idx[:, -self.config.model.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / self.config.model.temperature
            # optionally crop the logits to only the top k options
            if self.config.model.top_k is not None:
                v, _ = torch.topk(logits, min(self.config.model.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
