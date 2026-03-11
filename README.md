# 🤖 RAG Chatbot

基於 **Retrieval-Augmented Generation（RAG）** 技術的智能問答系統。上傳你的文件，讓本地 Llama 模型根據你的知識庫內容精確回答問題，**完全離線、無需付費 API**。

---

## 功能特色

- 📄 **多格式支援**：PDF、TXT、DOCX
- 🦙 **本地推論**：透過 Ollama 執行 Llama 等開源模型，資料不離開本機
- 🔍 **語意搜尋**：多語言 Embedding 模型，支援中英文混合查詢
- 💾 **持久化知識庫**：重啟程式後文件仍保留
- 📖 **來源標注**：每則回答附上參考段落與相似度分數
- 🔄 **即時切換模型**：側邊欄下拉選單，隨時換用不同本地模型
- 🧹 **一鍵清空**：支援清除知識庫或對話紀錄

---

## 系統需求

- Python 3.10 以上
- [Ollama](https://ollama.com)（本地 LLM 執行環境）
- 至少 8 GB RAM（建議 16 GB 以上以獲得較好效能）

---

## 安裝步驟

### 1. 安裝並啟動 Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

啟動 Ollama 服務：

```bash
ollama serve
```

> Ollama 需保持在背景執行，預設監聽 `http://localhost:11434`。

### 2. 下載語言模型

```bash
ollama pull llama3.2        # 2 GB，速度與品質平衡（推薦入門）
ollama pull llama3.1:8b     # 4.7 GB，回答品質較強
ollama pull gemma3          # Google 出品，中文表現不錯
```

已下載的模型可透過 `ollama list` 查看，也會自動出現在應用程式的下拉選單中。

### 3. 建立虛擬環境（建議）

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 4. 安裝相依套件

```bash
pip install -r requirements.txt
```

> 首次執行時會自動下載 Embedding 模型（約 300 MB），請確保網路連線正常。

---

## 啟動應用程式

```bash
streamlit run app.py
```

瀏覽器會自動開啟 `http://localhost:8501`。

---

## 使用說明

### Step 1｜選擇模型

側邊欄會自動列出本機已安裝的 Ollama 模型，從下拉選單選擇即可。若無法連線到 Ollama，會顯示文字輸入框讓你手動填寫模型名稱。

### Step 2｜上傳文件

點擊側邊欄的 **「上傳知識庫文件」** 區塊，選擇一或多個檔案（PDF / TXT / DOCX）。

上傳完成後，側邊欄會顯示：
- 已載入的段落數與文件數
- 每份文件的名稱清單

### Step 3｜開始提問

在底部輸入框輸入問題，按下 Enter 送出。

模型會根據知識庫內容回答，並在回答下方附上 **「📖 參考段落」** 展開區塊，顯示每個引用段落的來源檔案與相似度分數。

### Step 4｜管理知識庫與對話

| 按鈕 | 說明 |
|------|------|
| 🗑️ 清空知識庫 | 刪除所有已上傳文件（向量資料庫清空） |
| 🧹 清除對話紀錄 | 清除畫面上的對話，不影響知識庫 |

---

## 專案結構

```
chatbot/
├── app.py                   # Streamlit 主程式（前端介面）
├── requirements.txt         # Python 相依套件清單
├── .gitignore
└── rag/
    ├── __init__.py
    ├── document_loader.py   # 文件載入、解析、段落切割
    ├── vector_store.py      # ChromaDB 向量資料庫管理
    └── chatbot.py           # Ollama 整合與 RAG 邏輯
```

---

## 技術架構

```
使用者上傳文件
      │
      ▼
 document_loader.py
 ├─ 解析文件內容（PDF / TXT / DOCX）
 └─ 切割為 500 字元段落（重疊 80 字元）
      │
      ▼
 vector_store.py
 ├─ 使用 paraphrase-multilingual-MiniLM-L12-v2 計算 Embedding
 └─ 存入 ChromaDB（持久化於 ./chroma_db/）
      │
      ▼
 使用者提問
      │
      ▼
 vector_store.query()
 └─ 語意搜尋，找出最相關的 5 個段落
      │
      ▼
 chatbot.py
 ├─ 將相關段落注入 System Prompt
 └─ 透過 Ollama 呼叫本地 Llama 模型生成回答
      │
      ▼
 Streamlit 顯示回答 + 參考來源
```

---

## 常見問題

**Q：首次執行很慢？**
首次啟動會下載 Embedding 模型（約 300 MB），之後會從快取載入，速度正常。模型推論速度取決於你的 CPU / GPU 規格。

**Q：側邊欄顯示「無法連線到 Ollama」？**
請確認 `ollama serve` 已在背景執行，且沒有被防火牆阻擋。可執行 `curl http://localhost:11434` 確認服務正常。

**Q：重新啟動後文件還在嗎？**
是的，知識庫存於 `./chroma_db/` 資料夾，重啟後自動載入。若要重置，刪除該資料夾或點擊「🗑️ 清空知識庫」。

**Q：可以上傳中文文件嗎？**
完全支援。Embedding 模型為多語言模型，中英文皆可正確查詢。建議選用中文表現較佳的模型（如 `gemma3` 或 `llama3.1:8b`）。

**Q：知識庫太大會有問題嗎？**
ChromaDB 可處理大量文件，但查詢效能會隨資料量增加略有下降。建議依主題分開管理知識庫。

---

## 相依套件

| 套件 | 用途 |
|------|------|
| `ollama` | 本地 LLM 推論（Ollama Python client） |
| `streamlit` | Web 前端介面 |
| `langchain-community` | 文件載入器（PDF / DOCX） |
| `langchain-text-splitters` | 文字切割 |
| `chromadb` | 向量資料庫 |
| `sentence-transformers` | 文字 Embedding |
| `pypdf` | PDF 解析 |
| `docx2txt` | DOCX 解析 |
| `python-dotenv` | 讀取 .env 設定檔 |
