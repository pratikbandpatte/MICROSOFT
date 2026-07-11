import os
from dotenv import load_dotenv
from google import genai
from src.vectorization import Vectorizer
from src.retriever import mainretriever

vc= Vectorizer()
vector_db= vc.hfembedder()
mnr= mainretriever()

load_dotenv()
api_key = os.getenv("GENAI_API_KEY")
client=genai.Client(api_key=api_key)   

while True:
    question= input("\nAsk Any Question: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Stopped.")
        break
    context= mnr.Topkretriever(question, vector_db, 3) 
    sysprompt= f'''
    You are a helpful assistant. Use the context provided to answer the user's question.
    You are made by Pratik, your creator name is Pratik. Pratik is a Super Senior Pro Max AI Developer.
    Context: {context}
    User Question: {question}
    you have to give response in small general explanation with key important details.
    '''

    response = client.models.generate_content(
    model= "gemini-3.1-flash-lite",
    contents=sysprompt,
    )

    print("\nGemini Response:\n")
    print(response.text)



