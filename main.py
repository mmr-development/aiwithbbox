from pdf_to_image import pdf_to_image_exportable
from textExtractor import extract_text_from_images
from preprocessing_of_text import preprocess_texts
from nn import predict_labels

def process_menu(pdf_path, output_dir):
    # 1. Convert PDF to images
    image_paths = pdf_to_image_exportable(pdf_path, output_dir)
    if not image_paths:
        print("Failed to convert PDF to images.")
        return None

    # 2. Predict labels for each word using the neural network (LayoutLMv3)
    labeled_words = predict_labels(image_paths)
    print("Labeled Words:")
    for i, entry in enumerate(labeled_words):
        print(f"{entry['text']} - Label: {entry['label']}")

    # 3. Structure the labeled words into JSON format
    menu_json = []
    current_category = None
    for entry in labeled_words:
        label = entry["label"]
        text = entry["text"]
        if label == "CATEGORY":
            current_category = {"category": text, "items": []}
            menu_json.append(current_category)
        elif label in ["ITEM", "DESCRIPTION", "PRICE"]:
            if current_category is None:
                current_category = {"category": "Uncategorized", "items": []}
                menu_json.append(current_category)
            # Find or create the last item
            if label == "ITEM" or not current_category["items"]:
                current_category["items"].append({"name": "", "description": "", "price": ""})
            item = current_category["items"][-1]
            if label == "ITEM":
                item["name"] = text
            elif label == "DESCRIPTION":
                item["description"] = text
            elif label == "PRICE":
                item["price"] = text

    return menu_json

if __name__ == "__main__":
    pdf_path = "C:/Users/Mindaugas/Desktop/Svendeprøve/restaurant-menu-ai/menukort1.pdf"
    output_dir = "C:/Users/Mindaugas/Desktop/Svendeprøve/restaurant-menu-ai/output_images"
    menu = process_menu(pdf_path, output_dir)
    print("Processed Menu:")
    if menu:
        import json
        print(json.dumps(menu, indent=2))