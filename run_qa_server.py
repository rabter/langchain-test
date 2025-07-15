import os
from dotenv import load_dotenv
from langchain_community.vectorstores import vectara
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import retriever
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from pydantic import SecretStr
from functools import lru_cache
import time
import unicodedata
import sys

def normalize_text(text: str) -> str:
    """
    사용자 입력 쿼리에서 유니코드 정규화 및 깨진 문자 제거
    - NFC 정규화: 유니코드 조합형 문제 해결
    - UTF-8 디코딩 오류 방지를 위해 surrogate 문자 제거
    """

    # 유니코드 정규화 (NFC: 권장방식)
    text = unicodedata.normalize("NFC", text)

    # 깨진 문자(잘린 유니코드 등) 제거
    return text.encode("utf-8", errors="ignore").decode("utf-8")

def safe_input(prompt: str = "") -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        line = sys.stdin.buffer.readline()
        return line.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        print(f"입력 오류: {e}")
        return ""
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
# 유사도 기반 검색을 수행하면서 임계값 조건도 추가하는 방식(similarity_score_threshold)
retriever = vectorstore.as_retriever()    


qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# ✅ 일반 지식 응답 생성 함수
def answer_with_llm_only(llm, question: str) -> str:
    prompt = PromptTemplate.from_template("다음 질문에 일반 지식을 기반으로 답변해줘:\n\n{question}")

    return llm.invoke(prompt.format(question=question)).content

# ✅ RAG 여부 판단 + 분기 처리
def hybrid_qa(query: str) -> str:
    print(f"[DEBUG] query value: {query}")

    if not isinstance(query, str):
        raise ValueError("쿼리는 String이어야 합니다.")
    
    # 유사도 + 문서 가져오기
    try:
        docs_with_score = vectorstore.similarity_search_with_score(query, k=3)
    except Exception as e:
        print(f"문서 검색 중 오류: {e}")
        return answer_with_llm_only(llm, query)

    # 문서가 아예 없을 경우
    if not docs_with_score:
        print("❌ 문서가 아예 검색되지 않음")
        return answer_with_llm_only(llm, query)
    
    # 가장 유사한 문서의 유사도 점수 확인
    top_doc, top_score = docs_with_score[0]
    print(f"최고 유사도 점수: {top_score:.4f}")

    if top_score < 0.5:
        print(" 유사도가 낮아 일반 지식 기반 응답으로 전환")
        return answer_with_llm_only(llm, query)

    print("관련 문서가 충분히 유사하여 RAG 기반으로 응답합니다.")
    return qa_chain.invoke({"query": query})["result"]

# ✅ 사용자 입력 받아서 실행

# question = "SPERO의 구조는 무엇인가요?"
# question = "SPERO의 구조에서 Backend를 구성하는 요소들에 대해서 설명해주세요."
while True:
    query = safe_input("질문을 입력하세요 (exit 입력 시 종료): ")
    query_normalized = normalize_text(query)
    
    if not query_normalized:
        print("⚠️ 질문을 입력해주세요.")
        continue
    
    if query_normalized.lower() == "exit":
        print("👋 종료합니다.")
        break

    try:
        print("🧠 질문 분석 중...")
        result = hybrid_qa(query_normalized)
        print("\n💬 응답:", result, "\n")
    except Exception as e:
        print(f"에러 발생: {e}")