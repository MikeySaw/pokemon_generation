import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader
from make_dataset import PokemonDataset

def imshow(img, ax, caption):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    ax.set_title(caption, fontsize=8)
    ax.axis('off')

def show_batch(dataloader):
    batch = next(iter(dataloader))
    images, captions = batch
    batch_size = len(images)
    
    fig, axs = plt.subplots(1, batch_size, figsize=(15, 5))
    if batch_size == 1:
        axs = [axs]
    
    for i in range(batch_size):
        imshow(images[i], axs[i], captions[i])
    
    plt.show()
    plt.savefig("reports/figures/pokemon_images.png")


def dataset_statistics():
    dataset = torch.load("data/processed/pokemon.pth")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    show_batch(dataloader)

    

if __name__ == "__main__":
    dataset_statistics()