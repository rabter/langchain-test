import os
from dotenv import load_dotenv
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

# RAG 판별 기준 상수: 이보다 크면 유사하지 않다고 간주
MAX_DISTANCE_FOR_RAG = 1.1 # Chroma는 거리 기반이므로, 0에 가까울수록 유사, 너무 멀면 그냥 fallback(예외 방지용 최대값)
SCORE_GAP_THRESHOLD = 0.25   # 문서 1~2위 거리 차이가 작으면 유사한 문서로 판단

# .env 파일 로드
start = time.time()
_ = load_dotenv()
print(f"✅ 문서 로드 완료: {time.time() - start:.4f}초")

@lru_cache(maxsize=1)
def get_embedding():
    print("🔄 임베딩 모델 로딩 중... (최초 1회)")
    vector_model = "sentence-transformers/all-MiniLM-L6-v2" # 빠르지만 정확도 낮음
    # vector_model = "sentence-transformers/all-mpnet-base-v2" # 고성능
    # vector_model = "intfloat/e5-large-v2" # 성능 매우 우수
    return HuggingFaceEmbeddings(model_name=vector_model)

start = time.time()
embedding = get_embedding()

# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="../rag_db", embedding_function=embedding)
print(f"🔎 벡터 DB 내 문서 수: {vectorstore._collection.count()}개")


# Groq API 기반 LLM
llm = ChatOpenAI(
    model="gemma2-9b-it",
    temperature=0, # temperature=0.3~0.7 더 창의 적인 답변
        api_key=SecretStr(os.getenv("OPENAI_API_KEY") or "") if os.getenv("OPENAI_API_KEY") else None,
        base_url=os.getenv("OPENAI_API_BASE")
)


# RAG 체인 생성
# 유사도 기반 검색을 수행하면서 임계값 조건도 추가하는 방식(similarity_score_threshold)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # 기본 4~5 추천

qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# ✅ 일반 지식 응답 생성 함수
def answer_with_llm_only(llm, question: str) -> str:
    prompt = PromptTemplate.from_template("다음 질문에 일반 지식을 기반으로 답변해줘:\n\n{question}")

    return llm.invoke(prompt.format(question=question)).content

# ✅ RAG 여부 판단 + 분기 처리
def get_rag_answer(query: str) -> str:
    print(f"[DEBUG] query value: {query}")

    if not isinstance(query, str):
        raise ValueError("쿼리는 String이어야 합니다.")
    
    # 유사도 + 문서 가져오기
    try:
        docs_with_score = vectorstore.similarity_search_with_score(query, k=3)
        for i, (doc, score) in enumerate(docs_with_score, 1):
            print(f"[#{i}] 거리 점수: {score:.4f}")
            print(doc.page_content[:200])
            print("-" * 40)        
    except Exception as e:
        print(f"문서 검색 중 오류: {e}")
        return answer_with_llm_only(llm, query)

    # 문서가 아예 없을 경우
    if not docs_with_score:
        print("❌ 문서가 아예 검색되지 않음")
        return answer_with_llm_only(llm, query)
    
    # 거리 점수 추출 및 로그
    scores = [score for _, score in docs_with_score]

    # 가장 유사한 문서의 유사도 점수 확인
    top_doc, top_score = docs_with_score[0]

    print(f"\n 유사도 거리 점수:")
    for i, score in enumerate(scores, 1):
        print(f" - #{i} 문서 거리 점수: {score:.4f}")
    print(f"➡ 최고 거리 점수: {top_score:.4f}")

    score_gap = scores[1] - scores[0] if len(scores) > 1 else 1.0
    is_relevant = (top_score < MAX_DISTANCE_FOR_RAG and score_gap < SCORE_GAP_THRESHOLD)

    print(f"📐 거리 기준 판단 → top_score < {MAX_DISTANCE_FOR_RAG} = {top_score < MAX_DISTANCE_FOR_RAG}")
    print(f"📐 거리 차 판단 → gap < {SCORE_GAP_THRESHOLD} = {score_gap < SCORE_GAP_THRESHOLD}")
    print(f"📌 최종 유사 여부 판단 결과: {is_relevant}")

    # 응답 분기
    if is_relevant:
        print("✅ 관련 문서가 충분히 유사하여 RAG 기반 응답 수행")

        # retriever가 top-k 벡터 유사 문서를 찾아서 LLM에게 context로 넘겨줌
        # 내부적으로 vectorstore.similarity_search(query)를 수행함. 검색된 문서(top-k)만 LLM에게 전달
        return qa_chain.invoke({"query": query})["result"]
    else:
        print("🌐 관련 문서의 유사도가 낮아 일반 지식 기반 응답으로 fallback")
        return answer_with_llm_only(llm, query)
