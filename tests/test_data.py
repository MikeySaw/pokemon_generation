# import json
# import os.path

# import numpy as np
# import pytest
# from torchvision import transforms

# from src.data.make_dataset import PokemonDataset, pokemon_huggingface
# from tests import _PATH_DATA


# @pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
# def test_torch_data():
#     transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

#     train_file_path = "data/interim/train/metadata.jsonl"
#     val_file_path = "data/interim/val/metadata.jsonl"
#     test_file_path = "data/interim/test/metadata.jsonl"

#     train_data = []
#     val_data = []
#     test_data = []

#     with open(train_file_path, "r") as file:
#         for line in file:
#             train_data.append(json.loads(line))

#     with open(val_file_path, "r") as file:
#         for line in file:
#             val_data.append(json.loads(line))

#     with open(test_file_path, "r") as file:
#         for line in file:
#             test_data.append(json.loads(line))

#     train_dataset = PokemonDataset(data=train_data, image_folder="data/interim/train", transform=transform)
#     val_dataset = PokemonDataset(data=val_data, image_folder="data/interim/val", transform=transform)
#     test_dataset = PokemonDataset(data=test_data, image_folder="data/interim/test", transform=transform)

#     assert len(train_dataset) + len(val_dataset) + len(test_dataset) == 7357
#     for x, _ in train_dataset:
#         assert x.shape == (3, 128, 128)


# @pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
# def test_huggingface_data():
#     data = pokemon_huggingface()
#     assert len(data["train"]) + len(data["validation"]) + len(data["test"]) == 7357
#     assert np.mean([tuple(data["train"][i].keys()) == ("image", "text") for i in range(len(data["train"]))]) == 1
#     assert (
#         np.mean([tuple(data["validation"][i].keys()) == ("image", "text") for i in range(len(data["validation"]))]) == 1
#     )
#     assert np.mean([tuple(data["test"][i].keys()) == ("image", "text") for i in range(len(data["test"]))]) == 1
