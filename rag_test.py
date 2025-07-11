import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings (deprecated)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from pydantic import SecretStr

# .env 파일 로드
_ = load_dotenv()

# 문서 로드
loader = TextLoader("rag_sample.txt", encoding="utf-8")
documents = loader.load()

# 문서 쪼개기
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(documents)

# 텍스트를 벡터로 변환 (로컬 HuggingFace 모델 사용)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 벡터 DB 생성
vectorstore = Chroma.from_documents(splits, embedding)

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