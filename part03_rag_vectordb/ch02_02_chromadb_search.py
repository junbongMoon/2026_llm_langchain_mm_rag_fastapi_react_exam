# ChromaDB에 탐지 로그를 저장하고 유사 상황을 검색해 봅시다.

import chromadb
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()

# OpenAIEmbeddingFunction : chromadb가 문서 저장/검색 시 자동으로
# OpenAI API를 호출하여 벡터로 변환해준다.
embedding_fn = OpenAIEmbeddingFunction(
        model='text-embedding-3-small'
)

# 인메모리 클라이언트 (실습용 — 프로그램 종료 시 데이터 사라짐)
# 영구 저장하려면: client = chromadb.PersistentClient(path="./chroma_db")
client = chromadb.Client()


