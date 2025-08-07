import os
import time
from functools import lru_cache
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

# .env 환경변수 로드
load_dotenv()

# 벡터 거리 기준 설정
MAX_DISTANCE_FOR_RAG = 2.0
SCORE_GAP_THRESHOLD = 0.5

@lru_cache(maxsize=1)
def get_embedding():
    print("🔄 임베딩 모델 로딩 중...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding = get_embedding()

# 벡터스토어 연결
vectorstore = Chroma(persist_directory="../report.db", embedding_function=embedding)
print(f"📦 Report 벡터 DB 문서 수: {vectorstore._collection.count()}개")

# Groq 기반 LLM 정의
llm = ChatOpenAI(
    model="gemma2-9b-it",
    temperature=0.3,
    api_key=SecretStr(os.getenv("OPENAI_API_KEY") or "") if os.getenv("OPENAI_API_KEY") else None,
    base_url=os.getenv("OPENAI_API_BASE")
)

# 리포팅 전용 프롬프트
custom_prompt = PromptTemplate.from_template(
    """
    다음은 클라우드 자원의 사용량 리포트입니다:

    {context}

    위 리포트 데이터를 기반으로 아래 질문에 대해 **정확하고 구체적으로** 한국어로 답변하세요.
    - 제목을 붙이고, 마크다운 헤더 (`##`) 형식을 사용하세요.
    - 사용량 변화(%)를 요약하고, 자원별로 항목화하세요.
    - 가장 많이 사용된 자원을 강조하여 분석하세요.
    - 사용량 수치를 가능한 한 자연어로 요약하세요.
    - CPU, 메모리, 디스크 중 어떤 자원이 많이 사용되고 있는지 분석해 주세요.
    - 숫자만 나열하지 말고, 비율의 높고 낮음을 기준으로 상태를 판단해 주세요.
    - 존재하지 않는 정보는 언급하지 마세요.
    - **문서에 존재하지 않는 정보는 절대 추측하지 마세요.**

    질문: {question}

    답변:
    """
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True,
    verbose=True
)

def answer_with_llm_only(llm, question: str) -> str:
    fallback_prompt = PromptTemplate.from_template("다음 질문에 일반 지식을 기반으로 답변해줘:\n\n{question}")
    return llm.invoke(fallback_prompt.format(question=question)).content

def get_rag_answer(query: str) -> str:
    if not isinstance(query, str):
        raise ValueError("쿼리는 문자열이어야 합니다.")

    try:
        # context를 제대로 가져오는지 확인
        docs = vectorstore.similarity_search(query, k=5)
        print(f"📄 RAG 검색된 문서 수: {len(docs)}")
        for i, doc in enumerate(docs):
            print(f"[{i}] {doc.page_content[:200]}...")  # 첫 200자 정도


        docs_with_score = vectorstore.similarity_search_with_score(query, k=3)
        print(f"docs_with_score: {docs_with_score}")

    except Exception as e:
        print(f"문서 검색 오류: {e}")
        return answer_with_llm_only(llm, query)

    if not docs_with_score:
        return answer_with_llm_only(llm, query)

    scores = [score for _, score in docs_with_score]
    top_doc, top_score = docs_with_score[0]
    score_gap = scores[1] - scores[0] if len(scores) > 1 else 1.0

    print(f"top_score: {top_score}, score_gap: {score_gap}")


    is_relevant = (top_score < MAX_DISTANCE_FOR_RAG and score_gap < SCORE_GAP_THRESHOLD)

    if is_relevant:
        retrieved_docs = retriever.get_relevant_documents(query)  # Document 객체 리스트
        # 각 문서의 텍스트만 뽑아서 "\n\n"으로 연결
        retrieved_docs_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

        formatted_prompt = custom_prompt.format(context=retrieved_docs_text, question=query)
        print("---- LLM에게 전달되는 프롬프트 ----")
        print(formatted_prompt)
        print("----------------------------")
        return qa_chain.invoke({"query": query})["result"]
    else:
        return answer_with_llm_only(llm, query)
