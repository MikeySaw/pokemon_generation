import webdataset as wds 
import os 
import torch
import json
from make_dataset import PokemonDataset
from torchvision import transforms

def write_data_to_tar(dataset, type="train"):
    with wds.ShardWriter(f"data/processed/{type}-%06d.tar", maxcount=1000) as sink:
        for i, (image, caption) in enumerate(dataset):
            transform = transforms.ToPILImage()
            image = transform(image) 
            caption = json.dumps(caption)
            sink.write({
                "__key__": f"{i:06d}",
                "jpg": image,
                "json": caption
            })

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor()
        ])

    train_file_path = "data/interim/train/metadata.jsonl"
    val_file_path = "data/interim/val/metadata.jsonl"
    test_file_path = "data/interim/test/metadata.jsonl"

    train_data = []
    val_data = []
    test_data = []

    with open(train_file_path, "r") as file:
        for line in file:
            train_data.append(json.loads(line))

    with open(val_file_path, "r") as file:
        for line in file:
            val_data.append(json.loads(line))

    with open(test_file_path, "r") as file:
        for line in file:
            test_data.append(json.loads(line))

    train_dataset = PokemonDataset(data=train_data, image_folder="data/interim/train", transform=transform)
    val_dataset = PokemonDataset(data=val_data, image_folder="data/interim/val", transform=transform)
    test_dataset = PokemonDataset(data=test_data, image_folder="data/interim/test", transform=transform)

    write_data_to_tar(train_dataset, type="train")
    write_data_to_tar(val_dataset, type="val")
    write_data_to_tar(test_dataset, type="test")
    