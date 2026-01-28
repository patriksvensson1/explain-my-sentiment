from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Currently using RoBERTa model from huggingface
MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
