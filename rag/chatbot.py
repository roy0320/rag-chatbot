import ollama as _ollama
from typing import TypedDict

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

from rag.vector_store import VectorStore

# ── 常數 ──────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "llama3.2"
MAX_HISTORY_TURNS = 10   # 保留最近幾輪對話送給 LLM
MAX_RETRIES = 1          # Query Rewrite 最多重試幾次


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


# ── RAGChatbot ────────────────────────────────────────────────────────────────
class RAGChatbot:
    """
    使用 LangGraph 實作的 Corrective RAG 聊天機器人。

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

    def __init__(self, vector_store: VectorStore, model: str = DEFAULT_MODEL):
        self.vector_store = vector_store
        self.model = model
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

        # ChatOllama：LangChain 包裝的 Ollama 對話模型
        # temperature=0 讓 yes/no 評分更穩定
        llm = ChatOllama(model=self.model, temperature=0)
        vs = self.vector_store

        # ── Chain 1：文件相關性評分器 ────────────────────────────────────────
        # 用途：判斷撈回來的段落是否真的與問題有關
        # 輸出：只回答 "yes" 或 "no"
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
        # 用途：當撈不到相關段落時，把問題改寫成更適合向量搜尋的語句
        # 例如：「公司假期怎麼算？」→「員工年假計算規則」
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
        # 用途：根據篩選後的相關段落 + 對話歷史，生成最終回答
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
        # 每個節點：接收完整 state，回傳要更新的 key-value dict

        def retrieve(state: GraphState) -> dict:
            """
            【節點 1】向量搜尋

            呼叫 VectorStore.search_with_scores() 做語意搜尋，
            把相關度分數存進每個 Document 的 metadata，
            方便後續 UI 顯示相似度百分比。
            """
            results = vs.search_with_scores(state["question"], k=5)
            docs = []
            for doc, score in results:
                doc.metadata["_relevance"] = round(score, 4)
                docs.append(doc)
            return {"documents": docs}

        def grade_documents(state: GraphState) -> dict:
            """
            【節點 2】LLM 相關性評分

            對每個撈回的段落，讓 LLM 判斷「這段文字對回答問題有沒有幫助」。
            過濾掉回答 'no' 的段落，只保留真正相關的段落。

            這步驟解決向量搜尋的假陽性問題：
            向量相似不代表語意真的相關，讓 LLM 再做一層把關。
            """
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
            """
            【節點 3】查詢改寫（Corrective RAG 的核心）

            當 grade_documents 發現撈回的段落都不相關時，
            代表原始問題的「向量表示」可能不夠好，
            讓 LLM 把問題改寫成更具體的搜尋語句，然後再試一次。

            retries +1 防止無限迴圈（超過 MAX_RETRIES 就強制進 generate）。
            """
            new_question = rewriter.invoke({"question": state["question"]})
            return {
                "question": new_question.strip(),
                "retries": state["retries"] + 1,
            }

        def generate(state: GraphState) -> dict:
            """
            【節點 4】生成最終回答

            把篩選後的相關段落組合成 context，
            加上對話歷史一起送給 LLM 生成回答。
            若 documents 是空的（知識庫無相關資料），
            context 會標示「無相關資料」讓 LLM 誠實回答。
            """
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
            """
            grade_documents 完成後決定下一步：

            - 如果還有相關段落 → 直接 generate
            - 如果完全沒相關段落，且還有重試次數 → rewrite_query
            - 如果完全沒相關段落，但已超過重試上限 → 仍然 generate
              （會用空 context 讓 LLM 誠實回答沒有相關資料）
            """
            if state["documents"] or state["retries"] >= MAX_RETRIES:
                return "generate"
            return "rewrite_query"

        # ── 組裝圖 ───────────────────────────────────────────────────────────
        g = StateGraph(GraphState)

        # 加入節點
        g.add_node("retrieve", retrieve)
        g.add_node("grade_documents", grade_documents)
        g.add_node("rewrite_query", rewrite_query)
        g.add_node("generate", generate)

        # 加入固定邊（必定走這條路）
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade_documents")
        g.add_edge("rewrite_query", "retrieve")   # 改寫後重新撈
        g.add_edge("generate", END)

        # 加入條件邊（根據 decide_after_grading 的回傳值決定去哪）
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
        """
        執行 LangGraph 圖，取得回答與參考段落。

        第一次呼叫時才建立 graph（延遲初始化），
        避免模型未安裝時在 import 階段就報錯。
        """
        if self._graph is None:
            self._graph = self._build_graph()

        recent_history = history[-(MAX_HISTORY_TURNS * 2):]

        # invoke() 啟動整個圖，從 START 一路跑到 END
        result = self._graph.invoke({
            "question": user_query,
            "documents": [],
            "generation": "",
            "retries": 0,
            "history": recent_history,
        })

        # 把 LangChain Document 物件轉成 UI 用的 dict 格式
        retrieved_docs = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "distance": 1 - doc.metadata.get("_relevance", 0.5),
            }
            for doc in result["documents"]
        ]

        return result["generation"], retrieved_docs
