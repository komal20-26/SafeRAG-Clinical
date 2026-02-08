# rag.py
# ----------------------------------
# Loads the FAISS index and guideline chunks,
# retrieves relevant medical evidence,
# and generates a Safe-RAG summary.
# ----------------------------------

import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


INDEX_PATH = "artifacts/guideline.index"
CHUNKS_PATH = "artifacts/chunks.pkl"

MODEL_NAME = "google/flan-t5-base"


def load_resources():
    # Load FAISS index
    index = faiss.read_index(INDEX_PATH)

    # Load text chunks
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    # Load embedder
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Load lightweight CPU model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    return index, chunks, embedder, tokenizer, model


def retrieve_evidence(note, index, chunks, embedder, k=3):
    query_vec = embedder.encode([note], convert_to_numpy=True)
    D, I = index.search(query_vec, k)

    evidence = []
    for i in I[0]:
        evidence.append(chunks[i])

    return "\n".join(evidence)


def safe_rag_summarize(note, index, chunks, embedder, tokenizer, model):
    evidence = retrieve_evidence(note, index, chunks, embedder, k=3)

    prompt = f"""
You are a medical assistant explaining the case to a patient.

Create a patient-friendly explanation in simple language.

Rules:
- Use simple everyday words.
- Avoid medical jargon.
- Explain medical terms if needed.
- Keep sentences short.
- Only include important facts.
- Do not invent medical information.

Return output in this format:

Patient Problem:
Treatment or Tests:
Advice for Patient:

NOTE:
{note}

EVIDENCE:
{evidence}

ANSWER:
"""


    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_new_tokens=150)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary, evidence


if __name__ == "__main__":
    index, chunks, embedder, tokenizer, model = load_resources()

    sample_note = "Patient is a 65-year-old male with diabetes and chest pain. ECG pending."

    summary, evidence = safe_rag_summarize(
        sample_note, index, chunks, embedder, tokenizer, model
    )

    print("=== NOTE ===")
    print(sample_note)
    print("\n=== EVIDENCE ===")
    print(evidence)
    print("\n=== SAFE-RAG SUMMARY ===")
    print(summary)
