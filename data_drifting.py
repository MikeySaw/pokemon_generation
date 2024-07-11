import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA

# Load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Load the pretrained CLIP model for feature extraction
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

def extract_features(image):
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = clip_model.get_image_features(inputs['pixel_values'])
    return features.cpu().numpy().flatten()

# Generate multiple images and extract features
def generate_dataset(prompts, n_samples):
    features_list = []
    for _ in range(n_samples):
        for prompt in prompts:
            image = generate_image(prompt)
            features = extract_features(image)
            features_list.append(features)
    return np.array(features_list)

# Generate datasets
correct_prompts = ["A happy puppy", "A playful kitten", "A smiling child"]
incorrect_prompts = ["A pup", "A kitty", "A smelling child"]
n_samples = 5  # Increased to see a better shift result!

correct_features = generate_dataset(correct_prompts, n_samples)
incorrect_features = generate_dataset(incorrect_prompts, n_samples)

# Apply PCA for dimensionality reduction
n_samples, n_features = correct_features.shape
max_components = min(n_samples, n_features, 50)
pca = PCA(n_components=max_components)  

correct_features_pca = pca.fit_transform(correct_features)
incorrect_features_pca = pca.transform(incorrect_features)

# Create dataframes for Evidently
correct_df = pd.DataFrame(correct_features_pca, columns=[f'feature_{i}' for i in range(correct_features_pca.shape[1])])
incorrect_df = pd.DataFrame(incorrect_features_pca, columns=[f'feature_{i}' for i in range(incorrect_features_pca.shape[1])])

# Create and run Evidently report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=correct_df, current_data=incorrect_df)
report.save_html('image_drift_report.html')

print("Report generated and saved as 'image_drift_report.html'")