import numpy as np
import shap
import torch
from scipy.special import softmax
from typing import List
from app import sentiment_model as smodel

# Sentiment labels
LABELS = ["negative", "neutral", "positive"]

_SHAP_EXPLAINER = None

def get_shap_explainer():
    global _SHAP_EXPLAINER
    if _SHAP_EXPLAINER is None:
        masker = shap.maskers.Text(smodel.tokenizer)
        _SHAP_EXPLAINER = shap.Explainer(roberta_predict_probabilities, masker, output_names=LABELS)
    return _SHAP_EXPLAINER


# Named it RoBERTa to clarify that this is the model I've used in this project
def roberta_chunk_text(text: str, max_tokens_per_chunk: int = 450) -> List[str]:
    ids = smodel.tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(ids), max_tokens_per_chunk):
        chunk_ids = ids[i : i + max_tokens_per_chunk]
        chunks.append(smodel.tokenizer.decode(chunk_ids))
    return chunks

def roberta_predict_probabilities(texts):
    probabilities = []
    with torch.no_grad(): # Reduces overhead by avoiding graph building
        for row in texts:
            encoded_text = smodel.tokenizer(row, return_tensors="pt", truncation=True, max_length=512)
            output = smodel.model(**encoded_text)
            raw_scores = output.logits[0].cpu().numpy()
            probabilities.append(softmax(raw_scores))
    return np.vstack(probabilities)

def roberta_merge_subwords(tokens, contributions):
    merged = []
    current_word = ""
    current_word_score = 0.0

    for token, score in zip(tokens, contributions):
        if token in ("<s>", "</s>"):    # Tokens that RoBERTa uses for start/end of sentence
            continue
        score = float(score)

        if token.startswith("Ġ"): # New word
            if current_word:
                merged.append((current_word, current_word_score))
            current_word = token[1:] # Removes the Ġ prefix
            current_word_score = score
        else:  # Continuation of the previous word
            current_word += token
            current_word_score += score

    if current_word:
        merged.append((current_word, current_word_score))

    return merged

# Method for explaining predictions instead of just receiving sentiment values
def explain_prediction(text, num_of_top_words=15):
    encoded_text = smodel.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    probabilities = roberta_predict_probabilities([text])[0]
    predicted_index = int(np.argmax(probabilities))

    explainer = get_shap_explainer()
    shap_values = explainer([text])

    tokens = smodel.tokenizer.convert_ids_to_tokens(encoded_text["input_ids"][0])
    contributions = shap_values.values[0, :, predicted_index]

    merged = roberta_merge_subwords(tokens, contributions)
    merged.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "predicted_label": LABELS[predicted_index],
        "probabilities": {LABELS[i]: float(probabilities[i]) for i in range(3)},
        "top_word_contributions": merged[:num_of_top_words],
    }
