import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import pytesseract

# --- Configuration ---
MODEL_DIR = "./layoutlmv3-menu"
IMAGE_PATH = "training_images/menukort1_page_3.png"  # Use forward slash for cross-platform
LABELS = ["O", "CATEGORY", "ITEM", "DESCRIPTION", "PRICE"]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# Load processor and model
processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load image
image = Image.open(IMAGE_PATH).convert("RGB")

# Extract words and boxes using pytesseract with Danish language
ocr_data = pytesseract.image_to_data(image, lang="dan", output_type=pytesseract.Output.DICT)
words, boxes = [], []
width, height = image.size
for i in range(len(ocr_data["text"])):
    word = ocr_data["text"][i].strip()
    if word == "":
        continue
    x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
    # Normalize boxes to 1000 scale as expected by LayoutLMv3
    box = [
        int(1000 * x / width),
        int(1000 * y / height),
        int(1000 * (x + w) / width),
        int(1000 * (y + h) / height),
    ]
    words.append(word)
    boxes.append(box)

# Prepare encoding for the model
encoding = processor(
    image,
    words,
    boxes=boxes,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=512,
)
for k in encoding:
    encoding[k] = encoding[k].to(device)

# Run inference
with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()

tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().cpu().numpy())
labels = [ID2LABEL.get(int(pred), "O") for pred in predictions]

# Print results
print("Token\tLabel")
for token, label in zip(tokens, labels):
    if token not in processor.tokenizer.all_special_tokens:
        print(f"{token}\t{label}")

# Group tokens by label
results = list(zip(tokens, labels))
structured = []
current = {"label": None, "text": ""}
for token, label in results:
    if token in processor.tokenizer.all_special_tokens:
        continue
    if label != current["label"]:
        if current["label"] is not None:
            structured.append(current)
        current = {"label": label, "text": token.lstrip("Ġ")}
    else:
        if token.startswith("Ġ"):
            current["text"] += " " + token.lstrip("Ġ")
        else:
            current["text"] += token.lstrip("Ġ")
if current["label"] is not None:
    structured.append(current)

# Print grouped results
for entry in structured:
    print(f"{entry['label']}: {entry['text']}")