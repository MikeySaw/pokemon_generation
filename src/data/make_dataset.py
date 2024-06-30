import os
import argparse
import json
import random
from datasets import load_dataset
from omegaconf import OmegaConf
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
            split: str = 'train', 
            split_ratios: tuple = (0.8, 0.1, 0.1), 
            transform=None
    ):
        random.seed(123)
        n = len(data)
        self.data = data
        random.shuffle(data)

        train_ratio, val_ratio, test_ratio = split_ratios

        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("train_ratio + val_ratio + test_ratio must be 1")
        
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)

        if split == 'train':
            self.data = self.data[:train_end]
        elif split == 'val':
            self.data = self.data[train_end:val_end]
        elif split == 'test':
            self.data = self.data[val_end:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
        
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

def main(framework: str):
    logger.info(f"Loading images...")
    
    if framework == torch:
        transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])
        
        # check if processed datafolder exists, otherwise create it
        if not os.path.exists("data/processed"):
            os.makedirs("data/processed")

        file_path = "data/interim/pokemon_data.json"

        with open(file_path, 'r') as file:
            data = json.load(file)

        dataset = PokemonDataset(data=data, image_folder="data/raw", transform=transform)
        torch.save(dataset, os.path.join("data/processed", 'pokemon.pth'))

        logger.info(f"Created dataset and saved at {os.path.join('data/processed', 'pokemon.pth')}")

    
    else:
        dataset = pokemon_huggingface()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a Pokemon dataset')
    parser.add_argument('--framework', choices=['torch', 'huggingface'],
                        help='Choose between torch or huggingface', default='huggingface')

    args = parser.parse_args() 
    main(args.framework)
    