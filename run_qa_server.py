import os
from dotenv import load_dotenv
from langchain_community.vectorstores import vectara
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from pydantic import SecretStr
from functools import lru_cache
import time

from rag_test import embedding


# .env 파일 로드
start = time.time()
_ = load_dotenv()
print(f"✅ 문서 로드 완료: {time.time() - start:.4f}초")

@lru_cache(maxsize=1)
def get_embedding():
    print("🔄 임베딩 모델 로딩 중... (최초 1회)")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

start = time.time()
embedding = get_embedding()

# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="rag_db", embedding_function=embedding)



# Groq API 기반 LLM
llm = ChatOpenAI(
    model="gemma2-9b-it",
    temperature=0,
        api_key=SecretStr(os.getenv("OPENAI_API_KEY") or "") if os.getenv("OPENAI_API_KEY") else None,
        base_url=os.getenv("OPENAI_API_BASE")
)

# RAG 체인 생성
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())


# ✅ 사용자 입력 받아서 실행

# question = "SPERO의 구조는 무엇인가요?"
# question = "SPERO의 구조에서 Backend를 구성하는 요소들에 대해서 설명해주세요."
while True:
    query = input("질문을 입력하세요 (exit 입력 시 종료): ").strip()
    
    if not query:
        print("질문을 입력해주세요.")
        continue
    
    if query.lower() == "exit":
        break

    try:
        response = qa_chain.invoke({"query": query})
        print("\n💬 응답:", response["result"], "\n")
    except Exception as e:
        print(f"에러 발생: {e}")