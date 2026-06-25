#  FinDoc RAG — Financial Report Intelligence

A production-style **Retrieval-Augmented Generation (RAG)** system that lets you ask natural-language questions over any financial PDF annual reports, SEC 10-K filings, earnings releases.

<img src="https://github.com/d-payel/gifs_/blob/main/working_gif_git.gif" width="40" height="40" align="center" /> Built with **LangChain · ChromaDB · HuggingFace Embeddings · LLaMA · Streamlit**.

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                           USER QUERY                            │
└─────────────────────────┬───────────────────────────────────────┘
                          │
          ┌───────────────v───────────────┐
          │    Streamlit Frontend (UI)    │
          └───────────────┬───────────────┘
                          │
          ┌───────────────v───────────────────────────────────┐
          │           RAG Pipeline (LangChain LCEL)           │
          │                                                   │
          │  ┌───────────┐    ┌───────────┐    ┌───────────┐  │
          │  │    PDF    │    │   Text    │    │ Embedding │  │
          │  │  Parser   │───>│  Chunker  │───>│    (HF    │  │
          │  │ (PyMuPDF) │    │  800tok   │    │ MiniLM)   │  │
          │  └───────────┘    └───────────┘    └─────┬─────┘  │
          │                                          │        │
          │  ┌───────────────────────────────────────v─────┐  │
          │  │          ChromaDB Vector Store              │  │
          │  │  (Cosine Similarity · MMR · top 5 chunks)   │  │
          │  └─────────────────────────────────────┬───────┘  │
          │                                        │          │
          │  ┌─────────────────────────────────────v───────┐  │
          │  │         LCEL Chain (Groq LLaMA 3)           │  │
          │  │   RunnablePassthrough · StrOutputParser     │  │
          │  │      + Manual Memory (last 5 turns)         │  │
          │  └─────────────────────────────────────┬───────┘  │
          │                                        │          │
          └────────────────────────────────────────┼──────────┘
                                                   │
          ┌────────────────────────────────────────v──────────┐
          │   Answer + Source Citations (page, score, text)   │
          └───────────────────────────────────────────────────┘
```

---

##  Features

| Feature | Details |
|---|---|
| **Multi-turn chat** | Conversation memory (last 5 turns via `ConversationBufferWindowMemory`) |
| **Source citations** | Every answer shows source page, relevance score, and passage snippet |
| **MMR Retrieval** | Max Marginal Relevance ensures diverse, non-redundant context chunks |
| **Finance-tuned prompt** | System prompt engineered for financial precision, no hallucinated figures |
| **Sample questions** | One-click chip UI for quick demo |
| **Live stats** | Pages, chunks, and tokens embedded shown after ingestion |

---

##  Project Structure

```
financial-rag/
├── app.py                  # Streamlit UI
├── src/
│   ├── rag_pipeline.py     # Core RAG: PDF parsing, embedding, retrieval, chain
│   └── utils.py            # Helper functions
├── requirements.txt
└── README.md
```

---

##  Key Technical Decisions

**Chunking**: 800-token chunks with 120-token overlap using `RecursiveCharacterTextSplitter` — splits on paragraph breaks first, then sentences, never mid-word. Balances context richness with retrieval precision for dense financial prose.

**MMR Retrieval**: Chosen over pure cosine similarity to reduce redundancy when the same figure appears repeatedly (e.g., AUM mentioned 50+ times in an annual report). Fetches 20 candidates, returns the 5 most diverse and relevant.

**Cosine Similarity**: ChromaDB configured with `hnsw:space: cosine` instead of the default L2 distance — cosine is more appropriate for semantic embeddings where directional similarity matters more than magnitude.

**HuggingFace Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` runs via HuggingFace Inference API — no rate limits from API providers, no regional restrictions, no cost.

**Groq LLaMA 3**: Free inference with sub-5s latency on the free tier. Swap to `gpt-4o` for production-grade financial reasoning accuracy.

**ChromaDB in-memory**: Zero setup, no external services needed for prototyping. For production, swap to persistent ChromaDB with `persist_directory` or a managed vector store like Pinecone — eliminates re-embedding on every session.

**LCEL Pipeline**: Built using LangChain Expression Language (`|` pipe syntax) instead of deprecated chain classes — modern, composable, and easier to extend with additional steps like guardrails or reranking.

---
##  Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourusername/financial-rag
cd financial-rag
pip install -r requirements.txt

# 2. Run
streamlit run app.py
```

Enter your **OpenAI API key** in the sidebar, upload a PDF, click **Build Knowledge Base**, and start asking.

---

##  Where to Get Financial PDFs

### Free Official Sources

| Source | What you get | URL |
|---|---|---|
| **SEC EDGAR** | All US public company 10-K, 10-Q filings | https://www.sec.gov/cgi-bin/browse-edgar |
| **BlackRock IR** | BlackRock annual reports & earnings | https://ir.blackrock.com |
| **Annual Reports (.com)** | Curated annual reports from 8,000+ companies | https://www.annualreports.com |
| **Macrotrends** | Financial data + linked filings | https://www.macrotrends.net |

### Recommended Starter Documents

- **BlackRock 2023 Annual Report** — 10-K from EDGAR (search: `BLK 10-K 2023`)
- **JPMorgan Chase 2023 Annual Report** — good complexity
- **Apple 10-K 2023** — clean formatting, great for testing

### How to Download from SEC EDGAR

1. Go to https://efts.sec.gov/LATEST/search-index?q=%22BlackRock%22&dateRange=custom&startdt=2024-01-01&enddt=2024-12-31&forms=10-K
2. Click the filing → **Documents** tab → download the `.htm` or `.pdf` file

---

##  Potential Extensions

- [ ] **Hybrid search** — combine vector + BM25 keyword search (LangChain `EnsembleRetriever`)
- [ ] **Multi-document** — compare two company annual reports side by side
- [ ] **RAGAS evaluation** — automated faithfulness + answer relevancy scoring
- [ ] **Table extraction** — use `pdfplumber` to parse financial tables as structured data
- [ ] **Streaming responses** — LangChain streaming callbacks for faster UX
