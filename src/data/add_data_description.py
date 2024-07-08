import json
import os
import re

from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration


def extract_numeric_value(dictionary):
    match = re.search(r"\d+", dictionary["image"])
    return int(match.group()) if match else 0


def add_data_description(image_folder, output_json, processor, model, test=False):
    # Set the model to evaluation mode
    model.eval()

    # Create a list to store the data
    data = []

    # Iterate through the images
    i = 0
    for filename in tqdm(os.listdir(image_folder)):
        # Load the image
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)

        # Generate a caption for the image
        inputs = processor(images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, num_beams=5, early_stopping=True)
        caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Add the image path and caption to the data list
        data.append({"file_name": filename, "text": caption})

        if test:
            i += 1

            if i == 30:
                break

    # Sort the list of dictionaries based on the numeric value extracted from the 'image' key
    sorted_data = sorted(data, key=extract_numeric_value)

    # Save the data to a JSON file
    with open(output_json, "w") as f:
        json.dump(sorted_data, f)

    print(f"Data saved to {output_json}")


if __name__ == "__main__":
    image_folder = "data/raw"
    output_json = "data/pokemon_data.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the Blip2 model and processor
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
    model.to(device)

    add_data_description(image_folder=image_folder, output_json=output_json, processor=processor, model=model)
