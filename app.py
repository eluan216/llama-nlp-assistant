"""Streamlit frontend for the Document Q&A and Summarization system.

Optimized for Streamlit Cloud free tier.
"""

import streamlit as st

from src.data_loader import load_from_bytes
from src.chunking import chunk_text, get_chunk_stats
from src.embeddings import EmbeddingModel
from src.retriever import VectorStore
from src.llm_pipeline import LLMPipeline
from src.evaluation import evaluate_relevance, answer_overlap

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 LLM-Powered Document Q&A & Summarization")
st.caption(
    "Upload a PDF or TXT → Ask questions or generate a summary "
    "(RAG + open-source LLM • optimized for free Streamlit Cloud)"
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    chunk_size = st.slider("Chunk size", 200, 800, 400, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 100, 40, 10)
    top_k = st.slider("Top-k retrieved chunks", 1, 6, 3)

    st.divider()
    st.subheader("Generation")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max new tokens", 64, 400, 180, 16)

    st.divider()
    st.markdown("**Model**")
    st.info(
        "Default: `SmolLM2-360M-Instruct`\n\n"
        "Small & fast — works on Streamlit Cloud free tier.\n"
        "First load downloads ~700 MB."
    )

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""
if "llm" not in st.session_state:
    st.session_state.llm = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

# ---------------------------------------------------------------------------
# Document upload & indexing
# ---------------------------------------------------------------------------
st.subheader("1. Upload Document")
uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    with st.spinner("Loading and processing document..."):
        try:
            raw_text = load_from_bytes(uploaded_file.getvalue(), uploaded_file.name)
            st.session_state.raw_text = raw_text

            chunks = chunk_text(
                raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            st.session_state.chunks = chunks

            if st.session_state.embedding_model is None:
                with st.spinner("Loading embedding model (first time only)..."):
                    st.session_state.embedding_model = EmbeddingModel()

            vs = VectorStore(embedding_model=st.session_state.embedding_model)
            vs.build(chunks)
            st.session_state.vector_store = vs

            stats = get_chunk_stats(chunks)
            st.success(
                f"Document processed! {stats['count']} chunks "
                f"(avg length {stats['avg_len']:.0f} chars)"
            )

            with st.expander("Preview extracted text (first 1200 chars)"):
                st.text(raw_text[:1200] + ("..." if len(raw_text) > 1200 else ""))

        except Exception as e:
            st.error(f"Failed to process document: {e}")

# ---------------------------------------------------------------------------
# Lazy load LLM
# ---------------------------------------------------------------------------
def get_llm() -> LLMPipeline:
    if st.session_state.llm is None:
        with st.spinner(
            "Loading language model (SmolLM2-360M). "
            "This may take 1–2 minutes on first run..."
        ):
            try:
                st.session_state.llm = LLMPipeline()
            except Exception as e:
                st.error(
                    f"Could not load the model. Streamlit Cloud free tier "
                    f"may be out of memory.\n\nError: {e}"
                )
                st.stop()
    return st.session_state.llm

# ---------------------------------------------------------------------------
# Q&A / Summarization / Evaluation
# ---------------------------------------------------------------------------
if st.session_state.vector_store is not None:
    tab_qa, tab_sum, tab_eval = st.tabs(
        ["Question Answering", "Summarization", "Evaluation"]
    )

    with tab_qa:
        st.subheader("2. Ask a Question")
        question = st.text_input("Your question about the document:", key="qa_input")

        if st.button("Get Answer", type="primary", key="qa_btn") and question.strip():
            with st.spinner("Retrieving context and generating answer..."):
                vs: VectorStore = st.session_state.vector_store
                results = vs.search(question, top_k=top_k)
                context = "\n\n---\n\n".join(c for c, _ in results)

                llm = get_llm()
                answer = llm.answer(
                    question,
                    context,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )

                st.markdown("### Answer")
                st.write(answer)

                with st.expander("Retrieved context"):
                    for i, (chunk, score) in enumerate(results, 1):
                        st.markdown(f"**Chunk {i}** (score: {score:.3f})")
                        st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))
                        st.divider()

                st.session_state.last_qa = {
                    "question": question,
                    "answer": answer,
                    "context": context,
                    "results": results,
                }

    with tab_sum:
        st.subheader("2. Summarize Document")
        if st.button("Generate Summary", type="primary", key="sum_btn"):
            with st.spinner("Generating summary..."):
                llm = get_llm()
                summary = llm.summarize(
                    st.session_state.raw_text,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                st.markdown("### Summary")
                st.write(summary)

    with tab_eval:
        st.subheader("Simple Evaluation Metrics")
        if "last_qa" in st.session_state:
            data = st.session_state.last_qa
            rel = evaluate_relevance(data["question"], data["results"])
            overlap = answer_overlap(data["answer"], data["context"])

            col1, col2, col3 = st.columns(3)
            col1.metric("Avg retrieval score", f"{rel['avg_score']:.3f}")
            col2.metric("Max retrieval score", f"{rel['max_score']:.3f}")
            col3.metric("Answer-context overlap", f"{overlap:.3f}")

            st.json(rel)
        else:
            st.info("Run a question in the Q&A tab first to see evaluation metrics.")

else:
    st.info("↑ Upload a PDF or TXT file to get started.")

st.divider()
st.caption(
    "Built with Hugging Face Transformers • sentence-transformers • FAISS • Streamlit | "
    "Optimized for Streamlit Cloud free tier"
)
