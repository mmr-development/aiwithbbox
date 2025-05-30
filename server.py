from flask import Flask, request, jsonify
import os
import tempfile
from PIL import Image
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import pytesseract

# --- Model setup ---
MODEL_DIR = "./layoutlmv3-menu"
LABELS = ["O", "CATEGORY", "ITEM", "DESCRIPTION", "PRICE"]
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

processor = LayoutLMv3Processor.from_pretrained(MODEL_DIR, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_DIR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

app = Flask(__name__)

def extract_words_boxes(image, lang="dan"):
    ocr_data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
    words, boxes = [], []
    width, height = image.size
    for i in range(len(ocr_data["text"])):
        word = ocr_data["text"][i].strip()
        if word == "":
            continue
        x, y, w, h = ocr_data["left"][i], ocr_data["top"][i], ocr_data["width"][i], ocr_data["height"][i]
        box = [
            int(1000 * x / width),
            int(1000 * y / height),
            int(1000 * (x + w) / width),
            int(1000 * (y + h) / height),
        ]
        words.append(word)
        boxes.append(box)
    return words, boxes

def run_model(image):
    words, boxes = extract_words_boxes(image)
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
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
    tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().cpu().numpy())
    labels = [ID2LABEL.get(int(pred), "O") for pred in predictions]
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
    return structured

@app.route("/request/json/data", methods=["POST"])
def handle_request():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save file to temp and open as image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        image = Image.open(tmp_path).convert("RGB")
        result = run_model(image)
        os.remove(tmp_path)
        return jsonify({"results": result})
    except Exception as e:
        os.remove(tmp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)