import os
import hydra 
from omegaconf import OmegaConf
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


@hydra.main(config_path='../config', config_name='default_config.yaml')
def main(config):
    logger.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    logger.info(f"Loading images...")

    data_params = config.dataset
    
    transform = transforms.Compose([
        transforms.Resize(data_params['image_size']),
        transforms.ToTensor()
    ])

    print(data_params['image_size'])
    # that line is here because hydra changes the current workind directory
    os.chdir('../../..')
    # check if processed datafolder exists, otherwise create it
    if not os.path.exists(data_params['processed_path']):
        os.makedirs(data_params['processed_path'])

    dataset = PokemonDataset(image_folder=data_params['image_path'], transform=transform)
    torch.save(dataset, os.path.join(data_params['processed_path'], 'pokemon.pth'))
    
    logger.info(f"Created dataset and saved at {os.path.join(data_params['processed_path'], 'pokemon.pth')}")

    
if __name__ == "__main__": 
    main()
    
