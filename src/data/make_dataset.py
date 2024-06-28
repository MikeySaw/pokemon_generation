import os
import json
from omegaconf import OmegaConf
from loguru import logger

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms 


class PokemonDataset(Dataset):
    def __init__(self, data, image_folder, transform=None):
        self.data = data
        self.image_folder = image_folder
        self.transform = transform 
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data[idx]['image'])
        image = Image.open(img_name).convert('RGB')
        caption = self.data[idx]['caption']

        if self.transform:
            image = self.transform(image)

        return image, caption

def main():
    logger.info(f"Loading images...")
    
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])
    
    # check if processed datafolder exists, otherwise create it
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")

    file_path = "data/processed/pokemon_data.json"

    with open(file_path, 'r') as file:
        data = json.load(file)

    dataset = PokemonDataset(data=data, image_folder="data/raw", transform=transform)
    torch.save(dataset, os.path.join("data/processed", 'pokemon.pth'))
    
    logger.info(f"Created dataset and saved at {os.path.join('data/processed', 'pokemon.pth')}")

    
if __name__ == "__main__": 
    main()
    
