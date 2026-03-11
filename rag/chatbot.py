import ollama as _ollama
from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, START, END

from rag.vector_store import VectorStore

# ── 常數 ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "llama3.2"
DEFAULT_PROVIDER = "ollama"   # 預設使用本地 Ollama
MAX_HISTORY_TURNS = 10        # 保留最近幾輪對話送給 LLM
MAX_RETRIES = 1               # Query Rewrite 最多重試幾次


# ── LangGraph 狀態定義 ────────────────────────────────────────────────────────
class GraphState(TypedDict):
    """
    LangGraph 的狀態物件，在每個節點之間傳遞。
    每個節點接收整個 state dict，回傳要更新的部分 dict，
    LangGraph 自動合併（覆蓋同名 key）。
    """
    question: str        # 當前問題（可能被 rewrite_query 節點改寫）
    documents: list      # 撈到的 Document 物件 list（grade 後可能變少）
    generation: str      # 最終生成的回答
    retries: int         # 已重試次數（防止無限迴圈）
    history: list        # 對話歷史 [{"role": ..., "content": ...}]


# ── 工具函式 ──────────────────────────────────────────────────────────────────
def list_local_models() -> list[str]:
    """呼叫 Ollama API 取得本機已下載的模型清單，供 UI 下拉選單使用。"""
    try:
        result = _ollama.list()
        return [m.model for m in result.models]
    except Exception:
        return []


def build_llm(provider: str, model: str, api_key: str = "") -> BaseChatModel:
    """
    根據選擇的 provider 建立對應的 LLM 物件。

    這裡展示 LangChain 最大的優勢：
    不管用哪個模型供應商，對外介面都一樣（BaseChatModel），
    LangGraph 的節點完全不需要修改。

    支援：
    - ollama：本地模型，資料不外傳，免費
    - openai：OpenAI API（GPT-4o 等），需要 API 金鑰
    - azure：Azure OpenAI Service，企業級部署
    """
    if provider == "ollama":
        return ChatOllama(model=model, temperature=0)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0,
        )

    elif provider == "azure":
        from langchain_openai import AzureChatOpenAI
        import os
        return AzureChatOpenAI(
            azure_deployment=model,
            api_key=api_key,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version="2024-02-01",
            temperature=0,
        )

    else:
        raise ValueError(f"不支援的 provider: {provider}，請選擇 ollama / openai / azure")


# ── RAGChatbot ────────────────────────────────────────────────────────────────
class RAGChatbot:
    """
    使用 LangGraph 實作的 Corrective RAG 聊天機器人。
    支援多種 LLM 供應商：Ollama（本地）、OpenAI、Azure OpenAI。

    圖結構：
        START
          ↓
        retrieve          ← 從向量資料庫撈相關段落
          ↓
        grade_documents   ← 用 LLM 逐一判斷段落是否真的相關
          ↓ (conditional)
        ┌─── [有相關段落，或已重試過] ──→ generate → END
        └─── [沒有相關段落，且未重試] ──→ rewrite_query
                                              ↓
                                           retrieve （再試一次）
    """

    def __init__(
        self,
        vector_store: VectorStore,
        model: str = DEFAULT_MODEL,
        provider: str = DEFAULT_PROVIDER,
        api_key: str = "",
    ):
        self.vector_store = vector_store
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self._graph = None  # 延遲建立，第一次 chat() 才組裝

    # ── 建立 LangGraph 圖 ────────────────────────────────────────────────────
    def _build_graph(self):
        """
        組裝 LangGraph 的有向圖（StateGraph）。

        使用 LangChain Expression Language（LCEL）：
            prompt | llm | output_parser
        這個 pipe 語法讓 prompt → llm → parser 串成一條 chain，
        呼叫 chain.invoke({...}) 就會依序執行每個步驟。
        """

        # 根據 provider 建立對應的 LLM
        # 這裡換 provider 只需要改這一行，底下所有 chain 完全不用動
        llm = build_llm(self.provider, self.model, self.api_key)
        vs = self.vector_store

        # ── Chain 1：文件相關性評分器 ────────────────────────────────────────
        grader = (
            ChatPromptTemplate.from_messages([
                ("system",
                 "你是評估文件與問題相關性的評分員。\n"
                 "只回答 'yes'（相關）或 'no'（不相關），不要有其他文字。"),
                ("human", "問題：{question}\n\n文件內容：{document}"),
            ])
            | llm
            | StrOutputParser()
        )

        # ── Chain 2：查詢改寫器 ──────────────────────────────────────────────
        rewriter = (
            ChatPromptTemplate.from_messages([
                ("system",
                 "你是查詢優化專家。請將問題改寫為更適合語意搜尋的查詢語句。"
                 "只輸出改寫後的查詢，不要任何說明。"),
                ("human", "原始問題：{question}"),
            ])
            | llm
            | StrOutputParser()
        )

        # ── Chain 3：回答生成器 ──────────────────────────────────────────────
        generator = (
            ChatPromptTemplate.from_messages([
                ("system",
                 "你是一個根據知識庫內容回答問題的 AI 助手。\n\n"
                 "相關文件段落：\n{context}\n\n"
                 "回答規則：\n"
                 "1. 優先根據上方段落回答，並在末尾標注來源檔名。\n"
                 "2. 若段落不足，可補充一般知識，但需說明。\n"
                 "3. 若完全無相關資訊，誠實告知使用者。\n"
                 "4. 使用繁體中文回答，語氣清晰有條理。\n\n"
                 "歷史對話：\n{history}"),
                ("human", "{question}"),
            ])
            | llm
            | StrOutputParser()
        )

        # ── 節點定義 ─────────────────────────────────────────────────────────
        def retrieve(state: GraphState) -> dict:
            """【節點 1】向量搜尋"""
            results = vs.search_with_scores(state["question"], k=5)
            docs = []
            for doc, score in results:
                doc.metadata["_relevance"] = round(score, 4)
                docs.append(doc)
            return {"documents": docs}

        def grade_documents(state: GraphState) -> dict:
            """【節點 2】LLM 相關性評分"""
            relevant = []
            for doc in state["documents"]:
                verdict = grader.invoke({
                    "question": state["question"],
                    "document": doc.page_content,
                })
                if "yes" in verdict.strip().lower():
                    relevant.append(doc)
            return {"documents": relevant}

        def rewrite_query(state: GraphState) -> dict:
            """【節點 3】查詢改寫"""
            new_question = rewriter.invoke({"question": state["question"]})
            return {
                "question": new_question.strip(),
                "retries": state["retries"] + 1,
            }

        def generate(state: GraphState) -> dict:
            """【節點 4】生成最終回答"""
            context = "\n\n---\n\n".join(
                f"[來源: {d.metadata.get('source', 'Unknown')} | "
                f"相關度: {d.metadata.get('_relevance', 0):.1%}]\n{d.page_content}"
                for d in state["documents"]
            ) or "（知識庫中無相關資料）"

            history_text = "\n".join(
                f"{'使用者' if m['role'] == 'user' else '助手'}：{m['content']}"
                for m in state.get("history", [])
            ) or "（無歷史對話）"

            answer = generator.invoke({
                "context": context,
                "history": history_text,
                "question": state["question"],
            })
            return {"generation": answer}

        # ── 條件路由函式 ─────────────────────────────────────────────────────
        def decide_after_grading(state: GraphState) -> str:
            if state["documents"] or state["retries"] >= MAX_RETRIES:
                return "generate"
            return "rewrite_query"

        # ── 組裝圖 ───────────────────────────────────────────────────────────
        g = StateGraph(GraphState)

        g.add_node("retrieve", retrieve)
        g.add_node("grade_documents", grade_documents)
        g.add_node("rewrite_query", rewrite_query)
        g.add_node("generate", generate)

        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade_documents")
        g.add_edge("rewrite_query", "retrieve")
        g.add_edge("generate", END)

        g.add_conditional_edges(
            "grade_documents",
            decide_after_grading,
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query",
            },
        )

        return g.compile()

    # ── 對外介面 ────────────────────────────────────────────────────────────
    def chat(self, history: list[dict], user_query: str) -> tuple[str, list[dict]]:
        """執行 LangGraph 圖，取得回答與參考段落。"""
        if self._graph is None:
            self._graph = self._build_graph()

        recent_history = history[-(MAX_HISTORY_TURNS * 2):]

        result = self._graph.invoke({
            "question": user_query,
            "documents": [],
            "generation": "",
            "retries": 0,
            "history": recent_history,
        })

        retrieved_docs = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "distance": 1 - doc.metadata.get("_relevance", 0.5),
            }
            for doc in result["documents"]
        ]

        return result["generation"], retrieved_docs