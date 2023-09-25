# Pre-trained Model for Sentiment and Logic Consistency
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to("cuda")

tokenizer = AutoTokenizer.from_pretrained("mujeensung/roberta-base_mnli_bc")
model = AutoModelForSequenceClassification.from_pretrained("mujeensung/roberta-base_mnli_bc", num_labels=3, ignore_mismatched_sizes=True).to("cuda")

print('test successfully!')