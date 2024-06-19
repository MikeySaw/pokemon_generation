import os.path

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
    data = PokemonDataset(image_folder=os.path.join(_PATH_DATA, 'raw'),
                          transform=transform)
    assert len(data) == 7357
    for x in data:
        assert x.shape == (3, 128, 128)
