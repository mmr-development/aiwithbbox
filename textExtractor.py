import pytesseract
from PIL import Image

def extract_text_from_image(image_path):
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Use Tesseract to do OCR on the image
            text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_text_from_images(image_paths):
    texts = []
    for image_path in image_paths:
        text = extract_text_from_image(image_path)
        if text:
            texts.append(text)
    return texts
