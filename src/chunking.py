from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(self):
        pass
    def recursive_overlap(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into chunks: {len(chunks)}")
        return chunks