import os.path
import json

import pytest
from torchvision import transforms

from src.data.make_dataset import PokemonDataset
from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA),
                    reason='Data files not found')
def test_data():
    transform = transforms.Compose(
        [transforms.Resize((128, 128)),
         transforms.ToTensor()])
    file_path = "data/processed/pokemon_data.json"

    with open(file_path, 'r') as file:
        data = json.load(file)
    data_set = PokemonDataset(data, image_folder=os.path.join(_PATH_DATA, 'raw'),
                          transform=transform)
    assert len(data_set) == 7357
    for x, _ in data_set:
        assert x.shape == (3, 128, 128)
