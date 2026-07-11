import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "DATA"
CHROMA_DIR = "chroma"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QA_MODEL = "deepset/roberta-base-squad2"
TOP_K = 3

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file.")

documents = []

documents += DirectoryLoader(
    DATA_DIR,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
).load()

documents += DirectoryLoader(
    DATA_DIR,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
).load()

documents += DirectoryLoader(
    DATA_DIR,
    glob="**/*.docx",
    loader_cls=Docx2txtLoader
).load()

print(f"Loaded documents/pages: {len(documents)}")

if not documents:
    raise ValueError("No documents found in data folder.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Split into chunks: {len(chunks)}")

if not chunks:
    raise ValueError("No chunks created from documents.")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

rebuild_db = False

if Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir()):
    print("Loading existing Chroma DB...")

    vector_db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    try:
        db_count = vector_db._collection.count()
        print(f"Existing Chroma chunks: {db_count}")

        if db_count == 0:
            rebuild_db = True

    except Exception:
        rebuild_db = True

else:
    rebuild_db = True

if rebuild_db:
    print("Creating new Chroma DB...")

    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"New Chroma chunks: {vector_db._collection.count()}")

retriever = vector_db.as_retriever(
    search_kwargs={"k": TOP_K}
)

client = InferenceClient(token=HF_TOKEN)

while True:
    question = input("Question: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Stopped.")
        break

    if not question:
        print("Please enter a valid question.")
        continue

    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        print("No relevant documents found for this question.")
        continue

    context = "\n\n".join(
        doc.page_content.strip()
        for doc in retrieved_docs
        if doc.page_content and doc.page_content.strip()
    )

    if not context:
        print("Context is empty. Rebuild your Chroma DB by deleting the chroma folder.")
        continue

    try:
        answer = client.question_answering(
            model=QA_MODEL,
            question=question,
            context=context
        )

        answer_text = answer.get("answer", answer) if isinstance(answer, dict) else getattr(answer, "answer", answer)

        print("\nAnswer:", answer_text)

        print("\nSources used:")
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "unknown source")
            page = doc.metadata.get("page", "")
            print(f"{i}. {source} page {page}")

    except Exception as e:
        print("\nError while answering:", e)

    print("-" * 60)

