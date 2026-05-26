#  여러개의 wav를 배치 처리 하는 방법 습득
# 파일마다 load_model()을 호출하면 매번 모델 로딩하는데 수십초가 걸림.

import os
import time
import whisper
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import numpy as np

load_dotenv()

def get_embedding(text : str) -> np.array :
    """
    OpenAI text-embedding-3-small 모델로 텍스트를 벡터로 변환합니다.
    반환값: 1536차원 숫자 배열 (numpy array)

    text-embedding-3-small: 빠르고 저렴한 임베딩 모델
    text-embedding-3-large: 더 정확하지만 느리고 비쌈 (3072차원)
    """
    
    response = client.embeddings.create(
      model='text-embedding-3-small',
      input=text
    )
    
    # response.data[0].embedding -> 숫자 리스트 -> numpy 배열로 변환
    return np.array(response.data[0].embedding)
    

def batch_transcribe(audio_dir:str, model_name:str="base") -> list :
    """
    폴더 안의 오디오 파일을 모두 변환합니다.
    모델을 한 번만 로드하고 재사용합니다.

    사용 예시:
        results = batch_transcribe("./audio_files", model_name="base")
        for r in results:
            print(r["file"], "→", r["text"][:50])

    반환값 구조:
        [
            {
                "file":     "radio_001.wav",
                "text":     "전체 변환 텍스트",
                "language": "ko",
                "segments": [...]
            },
            ...
        ]
    """
    # 모델을 함수 밖에서 한 번만 로드
    model = whisper.load_model(model_name)
    results = []

    # WAV, MP3, MP4, M4A, FLAC 파일만 필터링
    audio_files = [
        f for f in os.listdir(audio_dir)
        if f.lower().endswith((".wav", ".mp3", ".mp4", ".m4a", ".flac"))
    ]

    for filename in sorted(audio_files):
        filepath = os.path.join(audio_dir, filename)
        print(f"변환 중: {filename}")

        result = model.transcribe(
            filepath,
            fp16=False
        )

        results.append({
            "file":     filename,
            "text":     result["text"],
            "language": result["language"],
            "segments": result["segments"],
        })
        print(f"  → {result['text'][:60]}")

    return results
  

# OpenAIEmbeddingFunction : chromadb가 문서 저장/검색 시 자동으로
# OpenAI API를 호출하여 벡터로 변환해준다.
embedding_fn = OpenAIEmbeddingFunction(
        model_name='text-embedding-3-small'
)

# 인메모리 클라이언트 (실습용 — 프로그램 종료 시 데이터 사라짐)
# 영구 저장하려면: client = chromadb.PersistentClient(path="./chroma_db")
client = chromadb.Client()
# client = chromadb.PersistentClient(path="./chroma_db")

# Collection : Vector DB 안의 데이터를 저장하는 단위 (RDB에서의 테이블 개념과 유사)
collection = client.create_collection(
        name="cctv_detection_logs",
        embedding_function=embedding_fn
)

trc_list = batch_transcribe('./waves', model_name='base')


# 프롬프트 템플릿  
prompt = PromptTemplate.from_template(
    """당신은 CCTV 보안 분석 전문가입니다.
아래 과거 대화를 참고하여 중요 정보를 누락 하지 않고 한국어로 요약된 문장을 출력하세요.
negative prompt : 주어지지 않은 정보에 대한 추론 금지.

[실제 통신 내용]
{context}

"""
)

# LLM load
llm = ChatOpenAI(model="gpt-4o", temperature=0)

records = []

for idx, trc in enumerate(trc_list, 1):
    text = trc["text"]

    rag_chain = prompt | llm | StrOutputParser()
    summary = rag_chain.invoke({"context": text})

    file_stem = os.path.splitext(trc["file"])[0]

    records.append({
        "id": f"audio_{idx}_{file_stem}",
        "document": summary,
        "metadata": {
            "source_file": "./waves/" + trc["file"],
            "language": trc["language"],
            "segment_count": len(trc["segments"]),
            "original_text_length": len(text),
            "summary_text_length": len(summary),
        }
    })
    
collection.add(
    documents=[r["document"] for r in records],
    ids=[r["id"] for r in records],
    metadatas=[r["metadata"] for r in records],
)

query = "제한구역 인근에서 시도된 침입에 의하여 모든 요원들이 공범으로 추정되는 인물을 포함한 탐지된 모든 용의자를 추격중인 자료를 출력"

# query = """B2 주차구역에서 30대 중반 남성이 제한구역 침입을 시도했으며, 
# 무단 출입카드 사용을 3회 시도했다. 
# 공범으로 추정되는 제2 인물도 탐지되었고, 
# 보안 인력들이 두 대상을 포위·추적 중이다. 주요 용의자는 현재 도주 속도를 높이고 있는 자료 출력"""

results = collection.query(
        query_texts=[query],
        n_results=1    # 상위 1개 반환
)

print("📋 유사 자료 ")
for i, (doc, meta) in enumerate(zip(
    results["documents"][0],
    results["metadatas"][0]
), 1):
    print(f"  📅 {meta['source_file']}")
    print(f"  📄 {doc}")
    

