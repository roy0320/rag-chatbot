from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── 常數 ──────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "documents"


class VectorStore:
    """
    LangChain 版向量資料庫。

    改用 langchain_chroma.Chroma 取代直接操作 chromadb，
    這樣可以直接取得 LangChain 原生的 Retriever 物件，
    讓 LangGraph 的 retrieve 節點可以直接呼叫。

    Embedding 改用 langchain_huggingface.HuggingFaceEmbeddings，
    與 LangChain 生態系整合，embedding / 搜尋都統一由 Chroma wrapper 處理。
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self._persist_dir = persist_dir

        # LangChain 的 HuggingFaceEmbeddings：
        # 包裝 sentence-transformers，讓 Chroma 可以自動在
        # add_documents / similarity_search 時呼叫它算向量。
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # 讓餘弦相似度落在 0~1
        )

        # langchain_chroma.Chroma 是 chromadb 的 LangChain 包裝器，
        # 負責把 Document 物件轉成向量並存入 ChromaDB。
        self.db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, chunks: list, source_name: str) -> int:
        """
        將切好的段落加入向量資料庫。

        LangChain 的 Chroma.add_documents() 會自動：
        1. 呼叫 embedding_function 把每段文字轉成向量
        2. 存入 ChromaDB（含文字內容 + metadata）
        不需要手動呼叫 model.encode()。
        """
        for chunk in chunks:
            chunk.metadata["source"] = source_name  # 把檔名寫進 metadata
        self.db.add_documents(chunks)
        return len(chunks)

    def search_with_scores(self, query: str, k: int = 5) -> list[tuple]:
        """
        語意搜尋，回傳 (Document, relevance_score) 的 list。

        使用 LangChain 的 similarity_search_with_relevance_scores()：
        - relevance_score 範圍 0~1，越高越相關（已正規化）
        - 底層呼叫 chromadb 的 cosine 距離，再換算成相關度
        """
        return self.db.similarity_search_with_relevance_scores(query, k=k)

    def get_document_count(self) -> int:
        """回傳目前資料庫中的段落總數（供 UI 顯示用）。"""
        return self.db._collection.count()

    def list_sources(self) -> list[str]:
        """回傳所有已上傳文件的檔名清單（去重複、排序）。"""
        result = self.db._collection.get()
        sources = {
            meta.get("source", "Unknown")
            for meta in (result.get("metadatas") or [])
            if meta
        }
        return sorted(sources)

    def clear(self):
        """清空所有段落（刪除所有 ID）。"""
        all_ids = self.db._collection.get()["ids"]
        if all_ids:
            self.db._collection.delete(ids=all_ids)
