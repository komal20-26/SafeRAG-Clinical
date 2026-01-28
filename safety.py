# safety.py
# ----------------------------------
# Verifies each sentence in a summary
# using Natural Language Inference (NLI).
# Drops unsupported / hallucinated claims.
# ----------------------------------

from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk

nltk.download("punkt")

# Load NLI model (CPU-safe)
nli = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    device=-1  # CPU
)


def is_entailed(source_text, claim):
    """
    Returns True if claim is entailed by source_text.
    """
    pair = f"{source_text} </s></s> {claim}"
    result = nli(pair, truncation=True)[0]
    return result["label"] == "ENTAILMENT"


def safety_filter(summary, source_note):
    """
    Keeps only those sentences in the summary
    that are supported by the source note.
    """
    sentences = sent_tokenize(summary)

    kept = []
    dropped = []

    for s in sentences:
        if is_entailed(source_note, s):
            kept.append(s)
        else:
            dropped.append(s)

    safe_summary = " ".join(kept)

    report = {
        "total_sentences": len(sentences),
        "kept": len(kept),
        "dropped": len(dropped),
        "supported_rate": len(kept) / max(1, len(sentences))
    }

    return safe_summary, report, dropped


if __name__ == "__main__":
    note = "Patient is a 65-year-old male with diabetes and chest pain. ECG pending."
    summary = (
        "The patient is a 65-year-old male. "
        "He has diabetes and chest pain. "
        "He is suffering from kidney failure."
    )

    safe, report, dropped = safety_filter(summary, note)

    print("ORIGINAL SUMMARY:\n", summary)
    print("\nSAFE SUMMARY:\n", safe)
    print("\nDROPPED SENTENCES:\n", dropped)
    print("\nREPORT:\n", report)
