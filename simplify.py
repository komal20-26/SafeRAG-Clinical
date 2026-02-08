REPLACEMENTS = {
    "hypertension": "high blood pressure",
    "hypotension": "low blood pressure",
    "myocardial infarction": "heart attack",
    "empiric antibiotics": "broad antibiotics started early",
    "HbA1c": "average blood sugar level",
}

def simplify_text(text):
    for k, v in REPLACEMENTS.items():
        text = text.replace(k, v)
    return text
