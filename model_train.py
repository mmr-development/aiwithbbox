import os
import json
import logging
from typing import List, Dict, Any
from tqdm import tqdm
from PIL import Image
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
)
import torch
import time
import random
import numpy as np

# --- Configuration ---
TRAIN_IMG_DIR = "training_images"
TRAIN_ANN_DIR = "training_data"
OUTPUT_DIR = "./layoutlmv3-menu"
LABELS = ["O", "CATEGORY", "ITEM", "DESCRIPTION", "PRICE"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
SEED = 42

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

def validate_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    """Ensure bbox is within image bounds and has four integers."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"Invalid bbox: {bbox}")
    x1, y1, x2, y2 = bbox
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    return [x1, y1, x2, y2]

def load_examples() -> List[Dict[str, Any]]:
    """Load and preprocess annotated examples for training."""
    examples = []
    ann_files = [f for f in os.listdir(TRAIN_ANN_DIR) if f.endswith(".json")]
    if not ann_files:
        logging.error("No annotation files found in %s", TRAIN_ANN_DIR)
        return examples

    for fname in tqdm(ann_files, desc="Loading annotations"):
        img_name = fname.replace(".json", ".png")
        img_path = os.path.join(TRAIN_IMG_DIR, img_name)
        ann_path = os.path.join(TRAIN_ANN_DIR, fname)
        if not os.path.exists(img_path):
            logging.warning("Image file %s not found, skipping.", img_path)
            continue
        try:
            with open(ann_path, encoding="utf-8") as f:
                anns = json.load(f)
            if not isinstance(anns, list) or not anns:
                logging.warning("Empty or invalid annotation in %s, skipping.", ann_path)
                continue
            image = Image.open(img_path).convert("RGB")
            width, height = image.size
            words, boxes, labels = [], [], []
            for ann in anns:
                text = ann.get("name") or ann.get("price") or ann.get("description")
                if not text or not isinstance(text, str):
                    continue
                try:
                    bbox = validate_bbox(ann["bbox"], width, height)
                except Exception as e:
                    logging.warning("Invalid bbox in %s: %s", ann_path, e)
                    continue
                # Normalize bbox to 1000 scale
                norm_bbox = [
                    int(1000 * bbox[0] / width),
                    int(1000 * bbox[1] / height),
                    int(1000 * bbox[2] / width),
                    int(1000 * bbox[3] / height),
                ]
                label = ann.get("type", "O").upper()
                if label not in LABEL2ID:
                    label = "O"
                tokens = processor.tokenizer.tokenize(text)
                for token in tokens:
                    words.append(token)
                    boxes.append(norm_bbox)
                    labels.append(LABEL2ID[label])
            if words:
                examples.append({"image": image, "words": words, "boxes": boxes, "labels": labels})
        except Exception as e:
            logging.error("Failed to process %s: %s", ann_path, e)
    return examples

class MenuDataset(torch.utils.data.Dataset):
    """Custom Dataset for LayoutLMv3 menu annotation."""
    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = processor(
            ex["image"],
            text=ex["words"],       # <-- FIXED
            boxes=ex["boxes"],
            word_labels=ex["labels"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

if __name__ == "__main__":
    logging.info("Loading processor and model...")
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", local_files_only=False, apply_ocr=False,)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    logging.info("Loading and preprocessing data...")
    start_time = time.time()
    examples = load_examples()
    if not examples:
        logging.error("No valid training examples found. Exiting.")
        exit(1)
    dataset = MenuDataset(examples)

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=32,
        num_train_epochs=500,
        logging_steps=100,
        save_steps=1000,
        learning_rate=5e-5,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=processor.tokenizer,
    )

    logging.info("Starting training...")
    trainer.train()
    logging.info("Training complete.")
    logging.info("Saving model and processor to %s", OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    elapsed = time.time() - start_time
    logging.info("Total time: %.2f seconds", elapsed)