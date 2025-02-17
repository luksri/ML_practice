"""
run this command in terminal to run the stable diffusion model

python3.10 launch.py --no-half --skip-torch-cuda-test

python3 launch.py --ckpt ~/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned.ckpt --no-half --skip-torch-cuda-test --disable-safe-unpickle

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import json
from PIL import Image
from io import BytesIO

# **Step 1: Load BioGPT Model**
model_name = "microsoft/biogpt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# **Step 2: Generate Medical Text with BioGPT**
def generate_medical_prompt(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**inputs, max_length=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example Query
bio_prompt = generate_medical_prompt("what is the drug under study?")

print("Generated Medical Prompt:\n", bio_prompt)

# **Step 3: Send Prompt to Stable Diffusion API**
stable_diffusion_url = "http://127.0.0.1:7860/sdapi/v1/txt2img"  # Update if using an external API

def generate_image(prompt):
    payload = {
        "prompt": prompt,
        "steps": 30,  # Number of inference steps
        "cfg_scale": 7.5,  # Guidance scale for quality
        "width": 512,
        "height": 512,
        "sampler_name": "Euler a"
    }
    response = requests.post(stable_diffusion_url, json=payload)
    if response.status_code == 200:
        img_data = response.json()['images'][0]
        image = Image.open(BytesIO(bytes.fromhex(img_data)))
        image.save("./generated_medical_image.png")
        print("✅ Image saved as generated_medical_image.png")
    else:
        print("❌ Error in generating image:", response.text)

# **Step 4: Generate Image from BioGPT Output**
generate_image(bio_prompt)
