import torch
import torchvision
from torch.utils.data import DataLoader
from data.make_dataset import PokemonDataset # noqa: F401

import matplotlib.pyplot as plt
import numpy as np 


def visualize_batch(batch, nrow=4, ncol=2):
    grid = torchvision.utils.make_grid(batch, nrow, ncol)

    plt.figure(figsize=(50,50))
    plt.imshow(np.transpose(grid, (1,2,0)))
    plt.savefig("reports/figures/batch.png")

def main():
    dataset = torch.load("data/processed/pokemon.pth")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(dataloader))
    visualize_batch(batch, 4, 4)
    print(batch.shape)


if __name__ == "__main__":
    main()