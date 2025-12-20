import os

base_path = "."

# Define the directory hierarchy
directories = [
    "configs",
    "data/raw",
    "data/processed",
    "src/model",
    "src/utils",
    "tests",
    "checkpoints",  # Added to store model weights
    "logs",  # Added for TensorBoard/WandB logs
]

# Files to initialize
files = {
    "src/model/__init__.py": "",
    "src/model/attention.py": "# Causal Self-Attention implementation\n",
    "src/model/layers.py": "# MLP and LayerNorm components\n",
    "src/model/transformer.py": "# Main GPT-2 architecture\n",
    "src/utils/__init__.py": "",
    "src/utils/checkpoint.py": "# Saving/Loading logic\n",
    "src/utils/logging.py": "# Metric tracking logic\n",
    "src/dataset.py": "# PyTorch Dataset and DataLoader\n",
    "src/trainer.py": "# Training engine class\n",
    "configs/gpt2_small.yaml": "n_layer: 12\nn_head: 12\nn_embd: 768\n",
    "main.py": "if __name__ == '__main__':\n    print('GPT-2 Project Initialized!')",
    "requirements.txt": "torch\ntiktoken\npyyaml\ntqdm\n",
    "README.md": "# GPT-2 From Scratch\nImplementation of the GPT-2 architecture.",
}


def create_project_structure(base_path, directories, files):
    # Create directories
    for dir in directories:
        os.makedirs(os.path.join(base_path, dir), exist_ok=True)
        # Create an __init__.py ind directories that need to be packeges
        if dir.startswith("src"):
            with open(os.path.join(dir, "__init__.py"), "w") as f:
                pass

    # Create files
    for file_path, content in files.items():
        full_path = os.path.join(base_path, file_path)
        if not os.path.exists(full_path):
            with open(full_path, "w") as f:
                f.write(content)
            print(f"Created {file_path}")

        else:
            print(f"Skipped (exists): {file_path}")

    print("Project structure created successfully!")


if __name__ == "__main__":
    create_project_structure(base_path, directories, files)
