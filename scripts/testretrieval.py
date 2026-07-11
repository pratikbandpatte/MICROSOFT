from src.docloader import MainLoader
from src.chunking import Chunker
from src.vectorization import Vectorizer
from src.retriever import mainretriever


mn= MainLoader()
ch= Chunker()
vc=Vectorizer()
mnr= mainretriever()

documents= mn.allfileloader()
print("test1")

chunked_docs= ch.recursive_overlap(documents)
print("test2")

vector_db= vc.hfembedder(chunked_docs)
fncontext= mnr.Topkretriever("Who is Nadela?", vector_db, 3)
print(fncontext)

