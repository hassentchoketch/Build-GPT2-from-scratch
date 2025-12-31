import torch
import os
import time
import yaml
from src.utils.checkpoint import save_checkpoint
from dataclasses import dataclass
import tiktoken


@dataclass
class TrainConfig:
    device: str
    learning_rate: float
    weight_decay: float
    max_steps: int
    accum_steps: int

    @classmethod
    def from_yaml(cls, config_path: str):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # We only oput the data section for this class
        return cls(**config_dict["train"])


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        config,
        train_loader,
        val_loader,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model.to(config.train.device)

        # Initialize with infinity
        self.best_val_loss = float("inf")

    def train(self):
        self.model.train()
        start_time = time.time()

        for step, (x, y) in enumerate(self.train_loader):
            if step >= self.config.train.max_steps:
                break

            # Forward pass
            logits, train_loss = self.model(x, y)

            # Scale loss for gradient accumulation
            train_loss = train_loss / self.config.train.accum_steps
            train_loss.backward()
            val_loss = self.evaluate()
            val_loss = val_loss / self.config.train.accum_steps

            # Only update weights every 'accum_steps'
            if (step + 1) % self.config.train.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                dt = time.time() - start_time
                print(
                    f"Step {step} | Train Loss: {train_loss.item() * self.config.train.accum_steps:.4f} | Val Loss: {val_loss:.4f} | Time: {dt:.2f}s"
                )
                start_time = time.time()

            # Check if this is the new 'best'
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            # Save (always updates latest, updates best if is_best is True)
            save_checkpoint(
                self.model,
                self.optimizer,
                step,
                val_loss,
                checkpoints_dir="checkpoints",
                is_best=is_best,
            )

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        steps = 0

        for x, y in self.val_loader:
            x, y = x.to(self.config.train.device), y.to(self.config.train.device)
            logits, loss = self.model(x, y)
            total_loss += loss.item()
            steps += 1
            # We don't need to run through the whole validation set if it's huge
            if steps >= self.config.train.max_steps:
                break

        loss = total_loss / self.config.train.accum_steps
        self.model.train()  # Set back to training mode
        return loss
