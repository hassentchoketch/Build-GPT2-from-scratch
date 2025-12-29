import torch
import os
import time
import yaml
from src.utils.checkpoint import save_checkpoint
from dataclasses import dataclass


@dataclass
class TrainConfig:
    device: str
    learning_rate: float
    weight_decay: float

    @classmethod
    def from_yaml(cls, config_path: str):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # We only oput the data section for this class
        return cls(**config_dict["train"])


class Trainer:
    def __init__(self, model, optimizer, config):
        self.model = model
        self.optimizer = optimizer
        # self.config = config
        # self.data_config = data_config
        # self.device = "cpu"
        self.model.to(config.train.device)

    def train(self, train_loader, max_steps=100, accum_steps=8):
        self.model.train()
        start_time = time.time()

        for step, (x, y) in enumerate(train_loader):
            if step >= max_steps:
                break

            # Forward pass
            logits, loss = self.model(x, y)

            # Scale loss for gradient accumulation
            loss = loss / accum_steps
            loss.backward()

            # Only update weights every 'accum_steps'
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                dt = time.time() - start_time
                print(
                    f"Step {step} | Loss: {loss.item() * accum_steps:.4f} | Time: {dt:.2f}s"
                )
                start_time = time.time()

            # Save periodically
            if step > 0 and step % 10 == 0:
                save_checkpoint(self.model, self.optimizer, step, loss.item())
