import os
import torch


def save_checkpoint(model, optimizer, step, loss, checkpoints_dir="checkpoints"):
    if not os.path.exists(checkpoints_dir):
        os.chdir(checkpoints_dir)

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict,
        "loss": loss,
    }
    path = os.path.join(checkpoints_dir, f"chpt_step_{step}.pt")
    torch.save(checkpoint, path)
    print(f"Checkpoint saves at step {step}")


def load_checkpoint(path, model, optimizer):
    # map_location = "cpu" is essential since you are on CPU
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["step"], checkpoint["loss"]
