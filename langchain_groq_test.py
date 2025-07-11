import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr
from dotenv import load_dotenv
from typing import cast

# .env 파일 로딩
_ = load_dotenv()

# LLM 인스턴스 생성
llm = ChatOpenAI(
        # model="llama3-8b-8192",
        model="gemma2-9b-it",
        temperature=0.7,
        api_key=SecretStr(os.getenv("OPENAI_API_KEY") or "") if os.getenv("OPENAI_API_KEY") else None,
        base_url=os.getenv("OPENAI_API_BASE")
)

# 1. 기본 메시지 전송 및 응답 출력
# response = llm.invoke([
#     HumanMessage(content="LangChain은 어떤 도구이고, 어떤 용도로 쓰이나요?")
# ])

# print("💬 응답:\n", cast(str, response.content))

# 2. 프롬프트 템플릿 예제
prompt = ChatPromptTemplate.from_template(
    "너는 전문적인 기술 설명가야. 다음 질문에 한국어로 자세히 답변해줘: \n\n질문: {question}"
)

prompt_template = prompt | llm

response = prompt_template.invoke({"question": "RAG는 무엇인가요?"})
print("💬 응답:\n", response.content)
