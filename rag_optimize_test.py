import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma # LangChain 1.0부터 사라질 예정. 더 이상 권장되지 않음
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from pydantic import SecretStr
import time



# .env 파일 로드
start = time.time()
_ = load_dotenv()
print(f"✅ 문서 로드 완료: {time.time() - start:.2f}초")

# 문서 로드
loader = TextLoader("rag_sample.txt", encoding="utf-8")
documents = loader.load()

# 문서 쪼개기
start = time.time()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(documents)
print(f"✅ 문서 분할 완료: {time.time() - start:.2f}초")

# 텍스트를 벡터로 변환 (로컬 HuggingFace 모델 사용)
# all-MiniLM-L6-v2 : 일반적이고 정확도 좋음, paraphrase-MiniLM-L3-v2 : 훨씬 작고 빠름(속도가 중요할 때 추천)
start = time.time()
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 최적화 부분
if os.path.exists("rag_db"):
    print("💬 기존 벡터 DB 로드..")
    vectorstore = Chroma(persist_directory="rag_db", embedding_function=embedding)
else:
    print("💬 새로운 벡터 DB 생성..")
    vectorstore = Chroma.from_documents(splits, embedding, persist_directory="rag_db")
    # vectorstore.persist() -> deprecated!! 자동 저장되어 호출 필요 없음
    print("✅ 저장 완료!")
    print(f"✅ 벡터 DB 생성 완료: {time.time() - start:.2f}초")


# Groq API 기반 LLM
llm = ChatOpenAI(
    model="gemma2-9b-it",
    temperature=0,
        api_key=SecretStr(os.getenv("OPENAI_API_KEY") or "") if os.getenv("OPENAI_API_KEY") else None,
        base_url=os.getenv("OPENAI_API_BASE")
)

# RAG 체인 생성
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

# 질문 실행
# question = "SPERO의 구조는 무엇인가요?"
# question = "SPERO의 구조에서 Backend를 구성하는 요소들에 대해서 설명해주세요."
question = "SPERO의 구조에서 Frontend에 대해 설명해주세요."
response = qa_chain.invoke(question)

print("💬 응답(질문):\n", response.get("query"))
print("💬 응답(답변):\n", response.get("result"))