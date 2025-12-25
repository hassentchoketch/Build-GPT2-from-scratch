from utils.config import ExperimentConfig


config = ExperimentConfig.load("configs/gpt2_small.yaml")

if __name__ == "__main__":
    print("GPT-2 Project Initialized!")
