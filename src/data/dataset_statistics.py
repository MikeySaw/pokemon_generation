import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from make_dataset import PokemonDataset
from mpl_toolkits.axes_grid1 import ImageGrid

def show_image_and_caption(images: torch.Tensor, caption: str) -> None:
    """Plot images and their captions in a grid."""
    row_col = int(len(images) ** 0.5)
    fig = plt.figure(figsize=(10.0, 10.0))
    grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
    for ax, im, cap in zip(grid, images, caption):
        ax.imshow(im.squeeze(), cmap="gray")
        ax.set_title(f"Caption: {cap}")
        ax.axis("off")
    plt.show()



def dataset_statistics():
    transform = transforms.Compose(
        [transforms.Resize((128, 128)),
         transforms.ToTensor()])
    data = PokemonDataset(image_folder="data/raw",
                          transform=transform)
    

if __name__ == "__main__":
    dataset_statistics()