import sys
import os
sys.path.insert(0, "/Users/qlai/Desktop/chatbot")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("RAG 知識庫助手")

# 延遲載入，不在啟動時初始化
_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        from rag.vector_store import VectorStore
        _vector_store = VectorStore(persist_dir="/Users/qlai/Desktop/chatbot/chroma_db")
    return _vector_store


@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """
    搜尋知識庫，找出與問題最相關的段落。
    當使用者詢問知識庫相關問題時使用這個工具。
    """
    vs = get_vector_store()
    results = vs.search_with_scores(query, k=3)
    
    if not results:
        return "知識庫中沒有找到相關資料。"
    
    output = []
    for doc, score in results:
        source = doc.metadata.get("source", "Unknown")
        output.append(
            f"來源：{source}（相似度 {score:.1%}）\n"
            f"內容：{doc.page_content}\n"
        )
    
    return "\n---\n".join(output)


@mcp.tool()
def get_status() -> str:
    """
    查詢知識庫目前的狀態。
    當使用者詢問知識庫有哪些資料時使用這個工具。
    """
    vs = get_vector_store()
    chunk_count = vs.get_document_count()
    sources = vs.list_sources()
    
    if chunk_count == 0:
        return "知識庫目前是空的，請先上傳文件。"
    
    sources_text = "\n".join(f"- {s}" for s in sources)
    return (
        f"知識庫狀態：\n"
        f"總段落數：{chunk_count}\n"
        f"文件數：{len(sources)}\n"
        f"已載入文件：\n{sources_text}"
    )


@mcp.tool()
def list_sources() -> str:
    """
    列出知識庫中所有已上傳的文件名稱。
    當使用者詢問知識庫有哪些文件時使用這個工具。
    """
    vs = get_vector_store()
    sources = vs.list_sources()
    
    if not sources:
        return "知識庫目前沒有任何文件。"
    
    sources_text = "\n".join(f"- {s}" for s in sources)
    return f"知識庫中共有 {len(sources)} 份文件：\n{sources_text}"


if __name__ == "__main__":
    mcp.run()