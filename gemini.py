import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GENAI_API_KEY")
client=genai.Client(api_key=api_key)    


response = client.models.generate_content(
    model= "gemini-3.1-flash-lite",
    contents="Explain AI in 5 simple sentences.",
)

print("\nGemini Response:\n")
print(response.text)

