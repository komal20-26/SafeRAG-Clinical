# app.py
# ----------------------------------
# End-to-end SafeRAG Clinical App
# ----------------------------------

import streamlit as st

from rag import load_resources, safe_rag_summarize
from safety import safety_filter

st.set_page_config(page_title="SafeRAG Clinical Summarizer", layout="wide")

st.title(" SafeRAG – Clinical Summarization System")
st.write("Evidence-bound summarization with hallucination control")

# Load models once
@st.cache_resource
def load_all():
    return load_resources()

index, chunks, embedder, tokenizer, model = load_all()

note = st.text_area(
    "Paste Clinical Note:",
    height=220,
    placeholder="Enter a clinical note here..."
)

if st.button("Generate Safe Summary") and note.strip():
    with st.spinner("Running SafeRAG pipeline..."):
        draft_summary, evidence = safe_rag_summarize(
            note, index, chunks, embedder, tokenizer, model
        )

        safe_summary, report, dropped = safety_filter(draft_summary, note)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(" Draft Summary (LLM Output)")
        st.write(draft_summary)

        st.subheader(" Retrieved Evidence")
        st.write(evidence)

    with col2:
        st.subheader(" Safe Summary (After Verification)")
        st.write(safe_summary)

        st.subheader(" Safety Report")
        st.json(report)

        if dropped:
            st.subheader(" Dropped Claims")
            for d in dropped:
                st.write("- ", d)
