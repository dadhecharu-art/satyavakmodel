from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ✅ use your Hugging Face repo directly
model_path = "CharuDadhe/legalbert_finetuned"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Ask user for input
text = input("Enter a legal text to analyze: ")

# Tokenize and run model
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)

# Predicted class
predicted_class = torch.argmax(outputs.logits, dim=1).item()
print(f"\nPredicted class: {predicted_class}")
