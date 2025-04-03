import openai
from dotenv import load_dotenv
import os
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']

# Replace with your actual API key
openai.api_key = OPENAI_API_KEY

try:
    response = openai.Model.list()  # Fetch available models
    print("✅ API Key is working. Available models:", [model["id"] for model in response["data"]])
except openai.error.AuthenticationError:
    print("❌ Invalid API Key! Check if it's correct.")
except openai.error.RateLimitError:
    print("⚠️ You have hit a usage limit. Check your billing settings.")
except openai.error.OpenAIError as e:
    print(f"🚨 OpenAI Error: {e}")
