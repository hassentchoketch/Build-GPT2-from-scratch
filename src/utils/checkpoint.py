import os
import torch


def save_checkpoint(
    model, optimizer, step, val_loss, checkpoints_dir="checkpoints", is_best=False
):
    """
    Saves the model state. Overwrites 'latest' always,
    and 'best' only if is_best is True.
    """
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict,
        "loss": val_loss,
    }
    # 1. Always save as the 'latest' for recovery
    latest_path = os.path.join("checkpoints", "latest_ckpt.pt")
    torch.save(checkpoint, latest_path)

    # 2. Only overwrite 'best_model' if it's actually the best
    if is_best:
        best_path = os.path.join("checkpoints", "best_model.pt")
        torch.save(checkpoint, best_path)
        print(f"🏆 New best model saved! (Validation Loss: {val_loss:.4f})")


def load_checkpoint(path, model, optimizer):
    # map_location = "cpu" is essential since you are on CPU
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"], checkpoint["loss"]
