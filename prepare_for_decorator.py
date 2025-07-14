from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import time
from pathlib import Path

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"⏱️ '{func.__name__}' 실행 시간: {time.time() - start:.4f}초")
        return result
    return wrapper

file_path = Path("~/test/langchain-test/rag_sample.txt").expanduser()

# 문서 로드
start = time.time()
loader = TextLoader(str(file_path), encoding="utf-8")

@timeit
def load_docs():
    return loader.load()

@timeit
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

@timeit
def load_embed():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@timeit
def create_vectorstore(splits, embedding):
    return Chroma.from_documents(splits, embedding, persist_directory="rag_db")

# 실행
document = load_docs()
splits = split_docs(document)
embedding = load_embed()
vectorstore = create_vectorstore(splits, embedding)