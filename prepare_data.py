import os


def split_text_file(input_path, train_ratio=0.9):
    # read the raw data
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()

    n = len(data)
    train_data = data[: int(n * train_ratio)]
    val_data = data[int(n * train_ratio) :]

    # Define paths based on your project structure
    train_path = os.path.join("data", "raw", "train.txt")
    val_path = os.path.join("data", "raw", "val.txt")

    # Save the splits
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_data)

    with open(val_path, "w", encoding="utf-8") as f:
        f.write(val_data)

    print(f"Data split complete!")
    print(f"All text set: {len(data):,} characters")
    print(f"Train set: {len(train_data):,} characters saved to {train_path}")
    print(f"Validation set: {len(val_data):,} characters saved to {val_path}")


if __name__ == "__main__":
    text_path = "data/raw/the-verdict.txt"
    if os.path.exists(text_path):
        split_text_file(text_path)
    else:
        print(f"Please put a text file at {text_path} fiest!")
