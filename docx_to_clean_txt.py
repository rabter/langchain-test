import os
import re
from docx import Document

def clean_paragraph(paragraph: str) -> str:
    # 줄바꿈 문자 정리
    text = paragraph.strip()

    # 머릿말/꼬릿말/페이지 정보처럼 불필요한 표현 제거
    patterns_to_remove = [
        r"^Page \d+ of \d+",     # 페이지 번호
        r"^Confidential.*$",     # 기밀 문구
        r"^Document Version.*$", # 버전 정보
        r"^Table of Contents.*$" # 목차
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # 공백이거나 너무 짧은 문장 제거
    if len(text.strip()) < 2:
        return ""

    return text

def docx_to_clean_txt(input_path: str, output_path: str):
    doc = Document(input_path)
    cleaned_paragraphs = []

    for para in doc.paragraphs:
        cleaned = clean_paragraph(para.text)
        if cleaned:
            cleaned_paragraphs.append(cleaned)

    # 문단 사이에 한 줄 공백 넣기
    cleaned_text = '\n\n'.join(cleaned_paragraphs)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    print(f"✅ 변환 완료: {output_path} (문단 수: {len(cleaned_paragraphs)})")

if __name__ == "__main__":
    # ✅ 사용 예시
    input_file = "rag_spero_manual.docx"       # 변환할 워드 파일 경로
    output_file = "user_manual_clean.txt" # 저장할 텍스트 파일 경로

    docx_to_clean_txt(input_file, output_file)
