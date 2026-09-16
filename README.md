# LLM Document Q&A & Summarization Assistant

End-to-end Retrieval-Augmented Generation (RAG) system for document question-answering and summarization using open-source models from Hugging Face.

**Live demo:** [https://llama-nlp-assistant-4hyjezelnvzdrndrsxqjts.streamlit.app](https://llama-nlp-assistant-4hyjezelnvzdrndrsxqjts.streamlit.app)

> First load may take 1–2 minutes while the model downloads. Subsequent use is faster.

**Author:** Oguma Eluanatein Odo

---

## Features

- **Document upload** — PDF and TXT
- **Intelligent chunking** — recursive character splitting with overlap
- **Semantic embeddings** — sentence-transformers
- **Vector retrieval** — FAISS similarity search
- **Open-source LLM** — default `HuggingFaceTB/SmolLM2-360M-Instruct` (fits Streamlit Cloud free tier)
- **Question answering** — answers grounded in uploaded documents
- **Summarization** — concise document summaries
- **Simple evaluation** — basic relevance / response quality helpers
- **Clean Streamlit UI**

---

## How It Works (RAG Pipeline)

1. **Load** — extract text from PDF/TXT  
2. **Chunk** — split into overlapping segments  
3. **Embed** — convert chunks to dense vectors  
4. **Index** — store in FAISS  
5. **Retrieve** — top-k relevant chunks for a query  
6. **Generate** — LLM answers using retrieved context  

---

## Project Structure

```text
llama-nlp-assistant/
├── data/               # Sample data & uploads
├── src/
│   ├── data_loader.py  # PDF/TXT loading
│   ├── chunking.py     # Text chunking
│   ├── embeddings.py   # Embedding model
│   ├── retriever.py    # FAISS + retrieval
│   ├── llm_pipeline.py # Generation pipeline
│   └── evaluation.py   # Simple metrics
├── notebooks/          # Experiments
├── app.py              # Streamlit application
├── requirements.txt
└── README.md
```

---

## Quick Start (local)

```bash
git clone https://github.com/eluan216/llama-nlp-assistant.git
cd llama-nlp-assistant
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Deploy `eluan216/llama-nlp-assistant`
4. Set **Main file path** to `app.py`
5. Deploy

After new commits, use **Manage app → Reboot** so changes are picked up.

---

## Tech Stack

- Python 3.10+
- PyTorch + Hugging Face Transformers
- sentence-transformers
- FAISS
- Streamlit
- pypdf

---

## Author

**Oguma Eluanatein Odo**  
[LinkedIn](https://linkedin.com/in/eluanatein-oguma-5552571b6) · [GitHub](https://github.com/eluan216)

---

## License

MIT

---

Built as a portfolio project demonstrating practical LLM + RAG engineering skills.
