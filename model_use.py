import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from itertools import groupby

# --- Configuration ---
MODEL_DIR = "./layoutlmv3-menu"
IMAGE_PATH = "canva-cream-and-black-minimalist-cafe-menu-PnXx0aP6HOI.jpg"  # Change this to your image file

LABELS = ["O", "CATEGORY", "ITEM", "DESCRIPTION", "PRICE"]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# Load processor and model
processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=True)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load and process image
image = Image.open(IMAGE_PATH).convert("RGB")
encoding = processor(image, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
for k in encoding:
    encoding[k] = encoding[k].to(device)

# Run inference
with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()

# Decode tokens and labels
tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().cpu().numpy())
labels = [ID2LABEL.get(int(pred), "O") for pred in predictions]

# Print results
print("Token\tLabel")
for token, label in zip(tokens, labels):
    if token not in processor.tokenizer.all_special_tokens:
        print(f"{token}\t{label}")

results = list(zip(tokens, labels))

structured = []
current = {"label": None, "text": ""}

for token, label in results:
    if token in processor.tokenizer.all_special_tokens:
        continue
    if label != current["label"]:
        if current["label"] is not None:
            structured.append(current)
        current = {"label": label, "text": token.replace("Ġ", " ").strip()}
    else:
        current["text"] += token.replace("Ġ", " ").strip()
if current["label"] is not None:
    structured.append(current)

# Print grouped results
for entry in structured:
    print(f"{entry['label']}: {entry['text']}")