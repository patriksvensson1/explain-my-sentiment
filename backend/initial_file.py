#Initial file to test the sentiment analysis and explainability setup, before refactoring etc.
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import shap

# Read data
df = pd.read_csv("backend/Reviews.csv").head(500) # Top 500 to limit runtime, file name does not matter for now
example_review = df["Text"][50] # Example review for testing

# RoBERTa model from huggingface (instead of using e.g. VADER approach)
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Sentiment labels
LABELS = ["negative", "neutral", "positive"]

# Method for predicting probabilities
def predict_probabilities(texts):
    probabilities = []
    for row in texts:
        encoded_text = tokenizer(row, return_tensors="pt", truncation=True, max_length=512)
        output = model(**encoded_text)
        raw_scores = output.logits[0].detach().numpy()
        probabilities.append(softmax(raw_scores))
    return np.vstack(probabilities)

# Method for merging RoBERTa subwords and their contributions
def merge_roberta_subwords(tokens, contributions):
    merged = []
    current_word = ""
    current_word_score = 0.0

    for token, score in zip(tokens, contributions):
        if token in ("<s>", "</s>"):    # Tokens that RoBERTa uses for start/end of sentence
            continue
        score = float(score)

        if token.startswith("Ġ"):  # New word
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
    encoded_text = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    probabilities = predict_probabilities([text])[0]
    predicted_index = int(np.argmax(probabilities))

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict_probabilities, masker, output_names=LABELS)
    shap_values = explainer([text])

    tokens = tokenizer.convert_ids_to_tokens(encoded_text["input_ids"][0])
    contributions = shap_values.values[0, :, predicted_index]

    merged = merge_roberta_subwords(tokens, contributions)
    merged.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "predicted_label": LABELS[predicted_index],
        "probabilities": {LABELS[i]: float(probabilities[i]) for i in range(3)},
        "top_word_contributions": merged[:num_of_top_words],
    }

result = explain_prediction(example_review)
print(result["predicted_label"])
print(result["probabilities"])
print(result["top_word_contributions"])