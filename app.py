import streamlit as st
from rag.document_loader import process_uploaded_file
from rag.vector_store import VectorStore
from rag.chatbot import RAGChatbot, list_local_models, DEFAULT_MODEL, DEFAULT_PROVIDER

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()
if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL
if "selected_provider" not in st.session_state:
    st.session_state.selected_provider = DEFAULT_PROVIDER
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "chatbot" not in st.session_state:
    st.session_state.chatbot = RAGChatbot(
        st.session_state.vector_store,
        model=st.session_state.selected_model,
        provider=st.session_state.selected_provider,
    )
if "session_uploads" not in st.session_state:
    st.session_state.session_uploads = set()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 設定")

    # ── Provider 選擇 ─────────────────────────────────────────────────────────
    st.subheader("🔌 模型供應商")
    provider = st.radio(
        "選擇模型來源",
        options=["ollama", "openai", "azure"],
        format_func=lambda x: {
            "ollama": "🦙 Ollama（本地，免費）",
            "openai": "🌐 OpenAI API",
            "azure": "☁️ Azure OpenAI",
        }[x],
        index=["ollama", "openai", "azure"].index(st.session_state.selected_provider),
    )

    # ── 根據 provider 顯示不同的設定 ─────────────────────────────────────────
    api_key = ""

    if provider == "ollama":
        # 本地 Ollama：列出已安裝的模型
        available_models = list_local_models()
        if available_models:
            selected_model = st.selectbox(
                "Ollama 模型",
                options=available_models,
                index=available_models.index(st.session_state.selected_model)
                if st.session_state.selected_model in available_models
                else 0,
            )
        else:
            selected_model = st.text_input(
                "Ollama 模型名稱",
                value=st.session_state.selected_model,
                placeholder="llama3.2",
            )
            st.caption("⚠️ 無法連線到 Ollama，請確認 ollama serve 已啟動。")

    elif provider == "openai":
        # OpenAI API：輸入 API 金鑰 + 選擇模型
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="從 https://platform.openai.com 取得",
        )
        selected_model = st.selectbox(
            "OpenAI 模型",
            options=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        )
        if not api_key:
            st.caption("⚠️ 請輸入 OpenAI API Key 才能使用。")

    elif provider == "azure":
        # Azure OpenAI：輸入 API 金鑰 + Deployment 名稱
        api_key = st.text_input(
            "Azure OpenAI API Key",
            type="password",
            placeholder="你的 Azure API Key",
        )
        selected_model = st.text_input(
            "Deployment 名稱",
            placeholder="gpt-4o",
            help="Azure OpenAI Studio 裡的 Deployment 名稱",
        )
        if not api_key:
            st.caption("⚠️ 請輸入 Azure API Key 才能使用。")

    # ── 偵測設定是否有變更，重新建立 chatbot ──────────────────────────────────
    config_changed = (
        provider != st.session_state.selected_provider
        or selected_model != st.session_state.selected_model
        or api_key != st.session_state.api_key
    )

    if config_changed and selected_model:
        st.session_state.selected_provider = provider
        st.session_state.selected_model = selected_model
        st.session_state.api_key = api_key
        st.session_state.chatbot = RAGChatbot(
            st.session_state.vector_store,
            model=selected_model,
            provider=provider,
            api_key=api_key,
        )
        provider_label = {
            "ollama": "Ollama",
            "openai": "OpenAI",
            "azure": "Azure OpenAI",
        }[provider]
        st.success(f"已切換至 {provider_label} / {selected_model}")

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

# 顯示目前使用的模型
provider_emoji = {"ollama": "🦙", "openai": "🌐", "azure": "☁️"}
st.caption(
    f"上傳文件建立知識庫，再向 AI 提問 | "
    f"{provider_emoji.get(st.session_state.selected_provider, '🤖')} "
    f"{st.session_state.selected_provider.upper()} / {st.session_state.selected_model}"
)

if not st.session_state.vector_store.get_document_count():
    st.info(
        "👈 請先在左側側邊欄上傳文件，建立知識庫後即可開始提問。\n\n"
        "你也可以直接提問，AI 將根據自身訓練知識回答。",
        icon="💡",
    )

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
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
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    api_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("AI 思考中…"):
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