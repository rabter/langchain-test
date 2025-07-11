import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# .env 파일 로딩
load_dotenv()

# LLM 인스턴스 생성
llm = ChatOpenAI(
        model="llama3-8b-8192",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
)

# 메시지 전송 및 응답 출력
response = llm.invoke([
    HumanMessage(content="LangChain은 어떤 도구이고, 어떤 용도로 쓰이나요?")
])

print("💬 응답:\n", response.content)
