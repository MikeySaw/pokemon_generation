import requests
import json
import base64
from PIL import Image
from io import BytesIO

url = "http://localhost:8080/predictions/latent_diffusion"
data = {"data": "A cute pokemon"}

response = requests.post(url, data=json.dumps(data))

if response.status_code == 200:
    result = response.json()
    img_data = base64.b64decode(result['generated_image'])
    img = Image.open(BytesIO(img_data))
    img.save("generated_pokemon.png")
    print("Image generated and saved as 'generated_pokemon.png'")
else:
    print(f"Error: {response.status_code}, {response.text}")
