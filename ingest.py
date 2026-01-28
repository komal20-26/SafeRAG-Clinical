# ingest.py
# -----------------------------
# Builds the medical knowledge base.
# - Defines guideline documents
# - Splits them into chunks
# - Creates embeddings
# - Stores them in a FAISS index
# - Saves everything to disk
# -----------------------------

import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss


GUIDELINES = [
    """Hypertension Management:
    Patients with blood pressure above 140/90 mmHg should be evaluated.
    First-line treatment includes ACE inhibitors, ARBs, calcium channel blockers.
    Monitor blood pressure and assess for organ damage.""",

    """Diabetes Mellitus Care:
    Patients require monitoring of blood glucose and HbA1c.
    Metformin is first-line therapy.
    Complications include nephropathy, neuropathy, retinopathy.""",

    """Chest Pain Evaluation:
    Chest pain should be assessed for myocardial infarction.
    ECG and troponin tests are essential.
    Risk factors include diabetes and hypertension.""",

    """Infection Management:
    Evaluate vitals, blood counts, cultures.
    Start empiric antibiotics when needed.
    Monitor for fever and hypotension.""",

    """Discharge Planning:
    Include diagnosis, hospital course, medications, follow-up.
    Educate patient about warning signs."""
]


def chunk_text(text, chunk_size=80):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


def build_index():
    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    all_chunks = []
    for doc in GUIDELINES:
        all_chunks.extend(chunk_text(doc))

    print(f"Total chunks: {len(all_chunks)}")

    print("Creating embeddings...")
    embeddings = embedder.encode(all_chunks, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs("artifacts", exist_ok=True)

    faiss.write_index(index, "artifacts/guideline.index")

    with open("artifacts/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("Index and chunks saved in /artifacts")


if __name__ == "__main__":
    build_index()
