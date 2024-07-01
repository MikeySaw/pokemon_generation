import os
import argparse
import json
import random
from datasets import load_dataset
from loguru import logger

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms 


class PokemonDataset(Dataset):
    def __init__(
            self, 
            data, 
            image_folder: str, 
            transform=None
    ):
        self.data = data
        self.image_folder = image_folder
        self.transform = transform 

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data[idx]['file_name'])
        image = Image.open(img_name).convert('RGB')
        caption = self.data[idx]['text']

        if self.transform:
            image = self.transform(image)

        return image, caption
    

def pokemon_huggingface():
    dataset = load_dataset("imagefolder", data_dir="data/interim")
    return dataset

def main():
    logger.info(f"Loading images...")
    
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])
    
    # check if processed datafolder exists, otherwise create it
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")

    train_file_path = "data/interim/train/metadata.jsonl"
    val_file_path = "data/interim/val/metadata.jsonl"
    test_file_path = "data/interim/test/metadata.jsonl"
    
    train_data = []
    val_data = []
    test_data = []

    with open(train_file_path, 'r') as file:
        for line in file:
            train_data.append(json.loads(line))

    with open(val_file_path, 'r') as file:
        for line in file:
            val_data.append(json.loads(line))

    with open(test_file_path, 'r') as file:
        for line in file:
            test_data.append(json.loads(line))

    train_dataset = PokemonDataset(data=train_data, image_folder="data/interim/train", transform=transform)
    torch.save(train_dataset, os.path.join("data/processed", 'pokemon_train.pth'))
    logger.info(f"Created dataset and saved at {os.path.join('data/processed', 'pokemon_train.pth')}")

    val_dataset = PokemonDataset(data=val_data, image_folder="data/interim/val", transform=transform)
    torch.save(val_dataset, os.path.join("data/processed", 'pokemon_val.pth'))
    logger.info(f"Created dataset and saved at {os.path.join('data/processed', 'pokemon_val.pth')}")

    test_dataset = PokemonDataset(data=test_data, image_folder="data/interim/test", transform=transform)
    torch.save(test_dataset, os.path.join("data/processed", 'pokemon_test.pth'))
    logger.info(f"Created dataset and saved at {os.path.join('data/processed', 'pokemon_test.pth')}")

    
if __name__ == "__main__":
    main()
    