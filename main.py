import torch
import tiktoken
from src.utils.config import ExperimentConfig
from src.dataset import create_dataloader
from src.model.transformer import GPT2Model
from src.trainer import Trainer


config = ExperimentConfig.load("configs/gpt2_small.yaml")


def main():
    # I- load train and valdatin dataset
    train_data_loader = create_dataloader(
        config.data.train_path, config.data.batch_size, config.model.block_size
    )
    val_data_loader = create_dataloader(
        config.data.val_path, config.data.batch_size, config.model.block_size
    )

    # II - Inastaintiat the model
    model = GPT2Model(config)

    # III- Train the Model
    # 1. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    # 2.train
    trainer = Trainer(
        model,
        optimizer,
        config,
        train_data_loader,
        val_data_loader,
    )
    trainer.train()

    # IV - Generate Text
    enc = tiktoken.get_encoding("gpt2")
    # 1. Prompting
    prompt = "Gisburn was rich; and it was immediately"
    x = torch.tensor(
        enc.encode(prompt),
        dtype=torch.long,
        device=config.train.device,
    ).unsqueeze(0)

    # 4. Generate!
    y = model.generate(x)
    print(f"Generated text: {enc.decode(y[0].tolist())}")


if __name__ == "__main__":
    main()
