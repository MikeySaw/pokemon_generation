import os 
import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# Set paths
image_folder = "data/raw"
output_json = "data/processed/pokemon_data.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the Blip2 model and processor
processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
model.to(device)

# Set the model to evaluation mode
model.eval()

# Create a list to store the data
data = []

# Iterate through the images
for filename in tqdm(os.listdir(image_folder)):
    # Load the image
    image_path = os.path.join(image_folder, filename)
    image = Image.open(image_path)

    # Generate a caption for the image
    inputs = processor(images=image, return_tensors="pt")
    generated_ids = model.generate(**inputs)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Add the image path and caption to the data list
    data.append({"image_path": image_path, "caption": caption})

# Save the data to a JSON file
with open(output_json, "w") as f:
    json.dump(data, f)

print(f"Data saved to {output_json}")