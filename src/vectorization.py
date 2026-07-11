from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from pathlib import Path


CHROMA_DIR= "chroma_db"
EMBEDDING_MODEL= "sentence-transformers/all-MiniLM-L6-v2"

class Vectorizer:
    def __init__(self):
        pass

    def hfembedder(self, chunks= None):
        embedding_model= HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        if Path(CHROMA_DIR).exists():
            print('Loading existing Chroma DB')
            vector_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
        else:
            print('Creating new Chroma DB')
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_model,
                persist_directory=CHROMA_DIR,
            )
        print("Chroma DB is created")
        return vector_db