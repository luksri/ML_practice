from openai import OpenAI
import os
from dotenv import load_dotenv
import requests

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
client = OpenAI()
OpenAI.api_key=os.environ['OPENAI_API_KEY']

def generate_infographic(prompt, output_filename="infographic.png"):
    """
    Generates an infographic using OpenAI's DALL·E model.
    :param prompt: Text description of the infographic.
    :param output_filename: Name of the output image file.
    """
    try:
        
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            response_format="b64_json",  # Get Base64 instead of a URL
            size="1024x1024"
        )

        # Decode base64 and save as an image
        import base64

        image_data = response.data[0].b64_json
        with open(output_filename, "wb") as file:
            file.write(base64.b64decode(image_data))

        print(f"✅ Image saved as {output_filename}")
    except Exception as e:
        print(f"Error in creating image: {e}")

# if __name__ == "__main__":

#     prompt = input("Enter a description for the infographic: ")
#     generate_infographic(prompt)
