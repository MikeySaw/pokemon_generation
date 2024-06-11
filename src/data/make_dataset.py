import os
from loguru import logger

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms 


class PokemonDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform 
        self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image 


def main():
    logger.info(f"Loading images from {"data/raw"}...")
    
    transform = transforms.Compose([
        transforms.Resize((TENSOR_SIZE)),
        transforms.ToTensor()
    ])

    # check if processed datafolder exists, otherwise create it
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    dataset = PokemonDataset(image_folder=RAW_DATA_DIR, transform=transform)
    torch.save(dataset, os.path.join(PROCESSED_DATA_DIR, 'pokemon.pth'))
    
    logger.info(f"Created dataset and saved at {os.path.join(PROCESSED_DATA_DIR, 'pokemon.pth')}")

    
if __name__ == "__main__": 
    TENSOR_SIZE = (128, 128) # you can change the size of the images here (TODO: Move this to hydra config)
    PROCESSED_DATA_DIR = "data/processed"
    RAW_DATA_DIR = "data/raw"
    main()
    
