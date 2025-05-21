from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import torch

# Load processor and model (replace with your fine-tuned model if available)
MODEL_NAME = "microsoft/layoutlmv3-base"
processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=True)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_NAME, num_labels=5)
LABELS = ["O", "CATEGORY", "ITEM", "DESCRIPTION", "PRICE"]

def predict_labels(image_paths):
    results = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        # Get OCR words and boxes
        ocr = processor(image, return_tensors="pt", return_attention_mask=True)
        # Only keep keys the model expects
        model_inputs = {k: v for k, v in ocr.items() if k in ["input_ids", "bbox", "attention_mask", "token_type_ids"]}
        words = ocr["words"] if "words" in ocr else ocr.get("ocr_words", [])
        # Forward pass
        with torch.no_grad():
            outputs = model(**model_inputs)
        predictions = outputs.logits.argmax(-1).squeeze().tolist()
        if isinstance(predictions, int):
            predictions = [predictions]
        if not words:
            words = processor.tokenizer.convert_ids_to_tokens(ocr["input_ids"].squeeze())
        for word, pred in zip(words, predictions):
            label = LABELS[pred]
            if label != "O":
                results.append({"text": word, "label": label})
    return results