import json
from pathlib import Path
from langchain.schema import Document
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import time


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"⏱️ '{func.__name__}' 실행 시간: {time.time() - start:.4f}초")
        return result
    return wrapper

# 파일 경로 설정
file_path = Path("raw_report.json")
txt_output_path = Path("processed_report.txt")
chroma_path = "chroma_report_db"

# JSON 파일 읽기
with file_path.open(encoding="utf-8") as f:
    raw_data = json.load(f)

# JSON -> LangChain Document 변환
documents = []
for row in raw_data:
    _time = row.get("_time")
    cloud = row.get("cloudName")
    provider = row.get("provider")
    cpu = round(row.get("cpuUsage", 0.0), 2)
    mem = round(row.get("memoryUsage", 0.0), 2)
    disk = round(row.get("diskUsage", 0.0), 2)

    content = (
        f"[{_time}]\n"
        f"Cloud: {cloud} (Provider: {provider})\n"
        f"CPU Usage: {cpu}%\n"
        f"Memory Usage: {mem}%\n"
        f"Disk Usage: {disk}%"
    )
    documents.append(Document(page_content=content, metadata=row))

# 전처리된 텍스트를 파일로 저장
with txt_output_path.open("w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc.page_content + "\n\n")

print(f"✅ 전처리 텍스트 저장 완료: {txt_output_path}")

# 문서 로드
start = time.time()
# loader = TextLoader(str(file_path), encoding="utf-8")

@timeit
def load_docs():
    loader = JSONLoader(
        file_path=str(file_path),
        jq_schema=".[]", # 배열 내 각 객체를 하나의 문서로
        text_content=False
    )
    return loader.load()

@timeit
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=256, chunk_overlap=32)
    return splitter.split_documents(docs)

@timeit
def load_embed():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@timeit
def create_vectorstore(splits, embedding):
    return Chroma.from_documents(splits, embedding, persist_directory="report.db")

# 실행
# document = load_docs()
splits = split_docs(documents) # load_docs() 대신 가공된 documents 사용
embedding = load_embed()
vectorstore = create_vectorstore(splits, embedding)