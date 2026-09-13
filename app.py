"""Streamlit frontend for the Document Q&A and Summarization system.

Features: model selector, multi-turn chat history, evaluation, RAG.
Optimized for Streamlit Cloud free tier.
"""

from __future__ import annotations

import streamlit as st

from src.data_loader import load_from_bytes
from src.chunking import chunk_text, get_chunk_stats
from src.embeddings import EmbeddingModel
from src.retriever import VectorStore
from src.llm_pipeline import LLMPipeline
from src.evaluation import evaluate_answer, evaluate_chat_history

# Models ranked roughly by size (smallest first — best for free cloud)
MODEL_OPTIONS = {
    "SmolLM2-360M (recommended for free tier)": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-3-mini-4k (heavier)": "microsoft/Phi-3-mini-4k-instruct",
}

st.set_page_config(
    page_title="LLM Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 LLM-Powered Document Q&A & Summarization")
st.caption("RAG pipeline with model selector, chat history, and evaluation metrics")

# Sidebar
with st.sidebar:
    st.header("Settings")

    model_label = st.selectbox("LLM model", list(MODEL_OPTIONS.keys()), index=0)
    selected_model = MODEL_OPTIONS[model_label]

    chunk_size = st.slider("Chunk size", 200, 800, 400, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 100, 40, 10)
    top_k = st.slider("Top-k retrieved chunks", 1, 6, 3)

    st.divider()
    st.subheader("Generation")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max new tokens", 64, 400, 180, 16)

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

    st.caption(f"Active model:\n`{selected_model}`")

# Session state
for key, default in [
    ("vector_store", None),
    ("chunks", []),
    ("raw_text", ""),
    ("llm", None),
    ("llm_name", None),
    ("embedding_model", None),
    ("chat_history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Upload
st.subheader("1. Upload Document")
uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    # Re-index when a new file is uploaded (by name)
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.get("file_id") != file_id:
        with st.spinner("Loading and processing document..."):
            try:
                raw_text = load_from_bytes(uploaded_file.getvalue(), uploaded_file.name)
                st.session_state.raw_text = raw_text
                st.session_state.file_id = file_id
                st.session_state.chat_history = []

                chunks = chunk_text(
                    raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                st.session_state.chunks = chunks

                if st.session_state.embedding_model is None:
                    with st.spinner("Loading embedding model..."):
                        st.session_state.embedding_model = EmbeddingModel()

                vs = VectorStore(embedding_model=st.session_state.embedding_model)
                vs.build(chunks)
                st.session_state.vector_store = vs

                stats = get_chunk_stats(chunks)
                st.success(
                    f"Document processed! {stats['count']} chunks "
                    f"(avg length {stats['avg_len']:.0f} chars)"
                )
            except Exception as e:
                st.error(f"Failed to process document: {e}")
    else:
        stats = get_chunk_stats(st.session_state.chunks)
        st.success(
            f"Document ready: {uploaded_file.name} — {stats['count']} chunks"
        )

def get_llm(model_name: str) -> LLMPipeline:
    # Reload if model selection changed
    if st.session_state.llm is None or st.session_state.llm_name != model_name:
        with st.spinner(f"Loading {model_name} (first time may take a few minutes)..."):
            try:
                st.session_state.llm = LLMPipeline(model_name=model_name)
                st.session_state.llm_name = model_name
            except Exception as e:
                st.error(f"Could not load model `{model_name}`.\n\n{e}")
                st.stop()
    return st.session_state.llm

if st.session_state.vector_store is not None:
    tab_chat, tab_sum, tab_eval = st.tabs(["Chat Q&A", "Summarization", "Evaluation"])

    with tab_chat:
        st.subheader("2. Chat with your document")

        # Render history
        for turn in st.session_state.chat_history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["content"])
                if turn.get("metrics"):
                    with st.expander("Turn metrics"):
                        st.json(turn["metrics"])

        prompt = st.chat_input("Ask a question about the document...")
        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    vs: VectorStore = st.session_state.vector_store
                    results = vs.search(prompt, top_k=top_k)
                    context = "\n\n---\n\n".join(c for c, _ in results)
                    llm = get_llm(selected_model)
                    answer = llm.answer(
                        prompt,
                        context,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                    )
                    metrics = evaluate_answer(prompt, answer, context, results)
                    st.markdown(answer)
                    with st.expander("Retrieved context & metrics"):
                        st.json(metrics)
                        for i, (chunk, score) in enumerate(results, 1):
                            st.markdown(f"**Chunk {i}** (score: {score:.3f})")
                            st.text(chunk[:400] + ("..." if len(chunk) > 400 else ""))

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "metrics": metrics}
            )

    with tab_sum:
        st.subheader("Summarize Document")
        if st.button("Generate Summary", type="primary", key="sum_btn"):
            with st.spinner("Generating summary..."):
                llm = get_llm(selected_model)
                summary = llm.summarize(
                    st.session_state.raw_text,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                st.markdown("### Summary")
                st.write(summary)

    with tab_eval:
        st.subheader("Session evaluation")
        summary_metrics = evaluate_chat_history(st.session_state.chat_history)
        if summary_metrics.get("turns", 0) == 0:
            st.info("Chat with the document first to populate evaluation metrics.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("QA turns", summary_metrics["turns"])
            c2.metric("Avg grounding", f"{summary_metrics['avg_grounding']:.3f}")
            c3.metric("Avg retrieval", f"{summary_metrics['avg_retrieval_score']:.3f}")
            st.json(summary_metrics)

else:
    st.info("↑ Upload a PDF or TXT file to get started.")

st.divider()
st.caption(
    "Hugging Face Transformers • sentence-transformers • FAISS • Streamlit | "
    "Model selector + chat history + evaluation"
)
