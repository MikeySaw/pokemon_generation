import json
import os
import random
import shutil


def split_data(data: dict, split_ratios: tuple = (0.8, 0.1, 0.1)):
    """
    Based on the json file created in add_data_description.py this function splits data into train, val and test
    """
    random.seed(123)
    n = len(data)
    random.shuffle(data)

    train_ratio, val_ratio, test_ratio = split_ratios
    if train_ratio + val_ratio + test_ratio != 1:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1")

    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def move_images(data: dict, destination: str = "train"):
    with open(os.path.join("data/interim", destination, "metadata.jsonl"), "w") as f:
        for line in data:
            json.dump(line, f)
            f.write("\n")

    filenames = [data[i]["file_name"] for i in range(len(data))]
    for file in filenames:
        src_path = os.path.join("data/raw", file)
        dest_folder = os.path.join("data/interim", destination)
        dest_path = os.path.join(dest_folder, file)
        shutil.copy(src_path, dest_path)
    print(f"Moved all {destination} images to {os.path.join('data/interim', destination)}")


def main():
    train_data, val_data, test_data = split_data(data=data)
    move_images(train_data, destination="train")
    move_images(val_data, destination="val")
    move_images(test_data, destination="test")


if __name__ == "__main__":
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")

    paths = ["train", "val", "test"]

    for path in paths:
        if not os.path.exists(os.path.join("data/interim", path)):
            os.makedirs(os.path.join("data/interim", path))

    file_path = "data/pokemon_data.json"

    with open(file_path, "r") as file:
        data = json.load(file)

    main()
