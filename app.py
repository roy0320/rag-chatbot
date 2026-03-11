import streamlit as st
from rag.document_loader import process_uploaded_file
from rag.vector_store import VectorStore
from rag.chatbot import RAGChatbot, list_local_models, DEFAULT_MODEL

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # {role, content, sources?}
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL
if "chatbot" not in st.session_state:
    st.session_state.chatbot = RAGChatbot(
        st.session_state.vector_store, model=st.session_state.selected_model
    )
if "session_uploads" not in st.session_state:
    st.session_state.session_uploads = set()  # files uploaded this session


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 設定")

    # Model selection
    available_models = list_local_models()
    if available_models:
        selected = st.selectbox(
            "Ollama 模型",
            options=available_models,
            index=available_models.index(st.session_state.selected_model)
            if st.session_state.selected_model in available_models
            else 0,
        )
    else:
        selected = st.text_input(
            "Ollama 模型名稱",
            value=st.session_state.selected_model,
            placeholder="llama3.2",
            help="請先確認 Ollama 已啟動（ollama serve）",
        )
        st.caption("⚠️ 無法連線到 Ollama，請確認服務已啟動。")

    if selected and selected != st.session_state.selected_model:
        st.session_state.selected_model = selected
        st.session_state.chatbot = RAGChatbot(
            st.session_state.vector_store, model=selected
        )
        st.success(f"已切換至 {selected}")

    st.divider()

    # ── Document Upload ───────────────────────────────────────────────────────
    st.subheader("📁 上傳知識庫文件")
    uploaded_files = st.file_uploader(
        "支援 PDF、TXT、DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state.session_uploads:
                with st.spinner(f"處理中：{f.name} ..."):
                    try:
                        chunks = process_uploaded_file(f)
                        n = st.session_state.vector_store.add_documents(chunks, f.name)
                        st.session_state.session_uploads.add(f.name)
                        st.success(f"✅ {f.name}（{n} 個段落）")
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")

    st.divider()

    # ── Knowledge Base Status ─────────────────────────────────────────────────
    st.subheader("📊 知識庫狀態")
    chunk_count = st.session_state.vector_store.get_document_count()
    sources = st.session_state.vector_store.list_sources()

    col1, col2 = st.columns(2)
    col1.metric("段落數", chunk_count)
    col2.metric("文件數", len(sources))

    if sources:
        with st.expander("已載入文件", expanded=True):
            for s in sources:
                st.write(f"• {s}")

    if chunk_count > 0:
        if st.button("🗑️ 清空知識庫", type="secondary", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.session_uploads.clear()
            st.rerun()

    st.divider()

    # ── Clear Chat ────────────────────────────────────────────────────────────
    if st.button("🧹 清除對話紀錄", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Chat Area ────────────────────────────────────────────────────────────
st.title("🤖 RAG Chatbot")
st.caption("上傳文件建立知識庫，再向機器人提問 | Powered by local LLMs & vector search")

if not st.session_state.vector_store.get_document_count():
    st.info(
        "👈 請先在左側側邊欄上傳文件，建立知識庫後即可開始提問。\n\n"
        "你也可以直接提問，機器人 將根據自身訓練知識回答。",
        icon="💡",
    )

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show retrieved sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📖 參考段落"):
                for doc in msg["sources"]:
                    similarity = 1 - doc["distance"]
                    st.markdown(f"**{doc['source']}** — 相似度 `{similarity:.1%}`")
                    preview = doc["content"][:300]
                    if len(doc["content"]) > 300:
                        preview += "…"
                    st.caption(preview)

# Chat input
if prompt := st.chat_input("輸入問題…"):
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build history for API (exclude metadata fields)
    api_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # exclude the just-added user msg
    ]

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            try:
                response_text, retrieved_docs = st.session_state.chatbot.chat(
                    api_history, prompt
                )
                st.markdown(response_text)

                if retrieved_docs:
                    with st.expander("📖 參考段落"):
                        for doc in retrieved_docs:
                            similarity = 1 - doc["distance"]
                            st.markdown(f"**{doc['source']}** — 相似度 `{similarity:.1%}`")
                            preview = doc["content"][:300]
                            if len(doc["content"]) > 300:
                                preview += "…"
                            st.caption(preview)

            except Exception as e:
                response_text = f"發生錯誤：{e}"
                retrieved_docs = []
                st.error(response_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text, "sources": retrieved_docs}
    )
