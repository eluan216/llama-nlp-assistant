"""Streamlit frontend — calm editorial UI for document Q&A (RAG).

Patterns inspired by modern AI marketing sites: hierarchy, quiet chrome,
source cards under answers, and a clear empty state. Optimized for
Streamlit Cloud free tier.
"""

from __future__ import annotations

import streamlit as st

from src.data_loader import load_from_bytes
from src.chunking import chunk_text, get_chunk_stats
from src.embeddings import EmbeddingModel
from src.retriever import VectorStore
from src.llm_pipeline import LLMPipeline
from src.evaluation import evaluate_answer, evaluate_chat_history

MODEL_OPTIONS = {
    "SmolLM2-360M (free tier)": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-3-mini (heavier)": "microsoft/Phi-3-mini-4k-instruct",
}

SAMPLE_PROMPTS = [
    "What is this document about?",
    "List the key takeaways in 5 bullets.",
    "What should I do next based on this?",
]

st.set_page_config(
    page_title="Document Q&A",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — quiet chrome, cards, editorial spacing
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"]  {
    font-family: 'Inter', system-ui, sans-serif;
  }

  .block-container {
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    max-width: 920px;
  }

  h1, .hero-title {
    font-family: 'Instrument Serif', Georgia, serif !important;
    font-weight: 400 !important;
    letter-spacing: -0.02em;
  }

  .hero-title {
    font-size: 2.4rem;
    line-height: 1.15;
    margin-bottom: 0.35rem;
    color: #1a1a1a;
  }

  .hero-sub {
    color: #5c5c5c;
    font-size: 1.02rem;
    line-height: 1.5;
    margin-bottom: 1.75rem;
  }

  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: #ffffff;
    border: 1px solid #e8e6e1;
    color: #3d3d3d;
    font-size: 0.82rem;
    font-weight: 500;
    margin-bottom: 1.25rem;
  }

  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #2f9e44;
    display: inline-block;
  }

  .card {
    background: #ffffff;
    border: 1px solid #e8e6e1;
    border-radius: 16px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
  }

  .answer-card {
    background: #ffffff;
    border: 1px solid #e8e6e1;
    border-radius: 16px;
    padding: 1.25rem 1.35rem;
    margin-top: 0.35rem;
  }

  .source-card {
    background: #fafaf8;
    border: 1px solid #ebe8e2;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.55rem;
    font-size: 0.88rem;
    color: #3a3a3a;
  }

  .source-meta {
    font-size: 0.75rem;
    color: #7a7a7a;
    margin-bottom: 0.35rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
  }

  .empty-wrap {
    background: #ffffff;
    border: 1px solid #e8e6e1;
    border-radius: 20px;
    padding: 2rem 1.75rem;
    margin-top: 0.5rem;
  }

  div[data-testid="stChatMessage"] {
    background: transparent;
  }

  header[data-testid="stHeader"] {
    background: rgba(247,246,243,0.85);
    backdrop-filter: blur(8px);
  }

  .footer-note {
    color: #8a8a8a;
    font-size: 0.8rem;
    text-align: center;
    margin-top: 2.5rem;
  }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in [
    ("vector_store", None),
    ("chunks", []),
    ("raw_text", ""),
    ("llm", None),
    ("llm_name", None),
    ("embedding_model", None),
    ("chat_history", []),
    ("file_id", None),
    ("file_name", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Sidebar — quiet, collapsed by default
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("**Controls**")
    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
    selected_model = MODEL_OPTIONS[model_label]

    with st.expander("Retrieval", expanded=False):
        chunk_size = st.slider("Chunk size", 200, 800, 400, 50)
        chunk_overlap = st.slider("Overlap", 0, 100, 40, 10)
        top_k = st.slider("Top-k", 1, 6, 3)

    with st.expander("Generation", expanded=False):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        max_tokens = st.slider("Max tokens", 64, 400, 180, 16)

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.caption(selected_model)

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">Ask your documents</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Upload a PDF or text file, then chat with grounded answers '
    "and visible sources — a calm RAG workspace.</div>",
    unsafe_allow_html=True,
)

if st.session_state.vector_store is not None:
    stats = get_chunk_stats(st.session_state.chunks)
    name = st.session_state.file_name or "Document"
    st.markdown(
        f'<div class="status-pill"><span class="status-dot"></span>'
        f"{name} · {stats['count']} chunks · ready</div>",
        unsafe_allow_html=True,
    )

uploaded_file = st.file_uploader(
    "Drop a document", type=["pdf", "txt"], label_visibility="collapsed"
)

if uploaded_file is not None:
    file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if st.session_state.get("file_id") != file_id:
        with st.spinner("Indexing document…"):
            try:
                raw_text = load_from_bytes(uploaded_file.getvalue(), uploaded_file.name)
                st.session_state.raw_text = raw_text
                st.session_state.file_id = file_id
                st.session_state.file_name = uploaded_file.name
                st.session_state.chat_history = []

                chunks = chunk_text(
                    raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
                st.session_state.chunks = chunks

                if st.session_state.embedding_model is None:
                    st.session_state.embedding_model = EmbeddingModel()

                vs = VectorStore(embedding_model=st.session_state.embedding_model)
                vs.build(chunks)
                st.session_state.vector_store = vs
                st.rerun()
            except Exception as e:
                st.error(f"Could not read that file. {e}")


def get_llm(model_name: str) -> LLMPipeline:
    if st.session_state.llm is None or st.session_state.llm_name != model_name:
        with st.spinner("Loading model (first time may take a minute)…"):
            try:
                st.session_state.llm = LLMPipeline(model_name=model_name)
                st.session_state.llm_name = model_name
            except Exception as e:
                st.error(f"Model failed to load.\n\n{e}")
                st.stop()
    return st.session_state.llm


def render_sources(results, metrics=None):
    if not results and not metrics:
        return
    with st.expander("Sources & metrics", expanded=False):
        if metrics:
            g = metrics.get("grounding_jaccard")
            avg = metrics.get("retrieval", {}).get("avg_score")
            cols = st.columns(2)
            if avg is not None:
                cols[0].caption(f"Retrieval · {avg:.3f}")
            if g is not None:
                cols[1].caption(f"Grounding · {g:.3f}")
        for i, (chunk, score) in enumerate(results, 1):
            preview = chunk[:320] + ("…" if len(chunk) > 320 else "")
            st.markdown(
                f'<div class="source-card">'
                f'<div class="source-meta">Source {i} · score {score:.3f}</div>'
                f"{preview}</div>",
                unsafe_allow_html=True,
            )


if st.session_state.vector_store is None:
    st.markdown('<div class="empty-wrap">', unsafe_allow_html=True)
    st.markdown("**Start with a document**")
    st.caption(
        "PDF or TXT. We'll chunk, embed, and retrieve context for every answer."
    )
    st.markdown("")
    st.caption("Once uploaded, try questions like:")
    for s in SAMPLE_PROMPTS:
        st.markdown(f"- *{s}*")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    tab_chat, tab_sum, tab_eval = st.tabs(["Chat", "Summary", "Evaluation"])

    with tab_chat:
        if not st.session_state.chat_history:
            st.caption("Suggested")
            bcols = st.columns(len(SAMPLE_PROMPTS))
            for i, sample in enumerate(SAMPLE_PROMPTS):
                if bcols[i].button(sample, key=f"sample_{i}", use_container_width=True):
                    st.session_state._pending_prompt = sample
                    st.rerun()

        for turn in st.session_state.chat_history:
            with st.chat_message(turn["role"]):
                if turn["role"] == "assistant":
                    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
                    st.markdown(turn["content"])
                    st.markdown("</div>", unsafe_allow_html=True)
                    render_sources(turn.get("results") or [], turn.get("metrics"))
                else:
                    st.markdown(turn["content"])

        pending = st.session_state.pop("_pending_prompt", None)
        prompt = st.chat_input("Ask anything about this document…") or pending

        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})

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

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "metrics": metrics,
                    "results": results,
                }
            )
            st.rerun()

    with tab_sum:
        st.markdown("#### Document summary")
        st.caption("A concise pass over the indexed text.")
        if st.button("Generate summary", type="primary"):
            with st.spinner("Writing summary…"):
                llm = get_llm(selected_model)
                summary = llm.summarize(
                    st.session_state.raw_text,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                st.markdown('<div class="answer-card">', unsafe_allow_html=True)
                st.markdown(summary)
                st.markdown("</div>", unsafe_allow_html=True)

    with tab_eval:
        st.markdown("#### Session quality")
        summary_metrics = evaluate_chat_history(st.session_state.chat_history)
        if summary_metrics.get("turns", 0) == 0:
            st.caption("Chat first — metrics appear per answer and for the session.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Turns", summary_metrics["turns"])
            c2.metric("Avg grounding", f"{summary_metrics['avg_grounding']:.3f}")
            c3.metric("Avg retrieval", f"{summary_metrics['avg_retrieval_score']:.3f}")
            with st.expander("Raw metrics"):
                st.json(summary_metrics)

st.markdown(
    '<div class="footer-note">Transformers · sentence-transformers · FAISS · Streamlit</div>',
    unsafe_allow_html=True,
)
