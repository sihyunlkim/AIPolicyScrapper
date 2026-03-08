from google import genai
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(m.name)