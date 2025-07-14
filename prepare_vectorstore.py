from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import time
from pathlib import Path

file_path = Path("~/test/langchain-test/rag_sample.txt").expanduser()

# 문서 로드
start = time.time()
loader = TextLoader(str(file_path), encoding="utf-8")
documents = loader.load()
print(f"📄 문서 로딩 완료: {time.time() - start:.4f}초")

# 문서 쪼개기
start = time.time()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(documents)
print(f"✅ 문서 분할 완료: {time.time() - start:.4f}초")

start = time.time()
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print(f"🧠 임베딩 모델 로드 완료: {time.time() - start:.4f}초")

# ✅ 최초 1회만 실행
start = time.time()
vectorstore = Chroma.from_documents(splits, embedding, persist_directory="rag_db")
print(f"📦 벡터 저장소 생성 완료: {time.time() - start:.4f}초")