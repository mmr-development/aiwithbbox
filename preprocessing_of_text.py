import re

def clean_ocr_line(line):
    # Remove unwanted characters
    line = re.sub(r'[^\x00-\x7F]+', '', line)  # Remove non-ASCII chars
    line = re.sub(r'\s+', ' ', line)           # Normalize whitespace
    line = line.strip()
    # Fix common OCR errors (example: replace '0' with 'O' if surrounded by letters)
    line = re.sub(r'(?<=\b[A-Za-z])0(?=[A-Za-z]\b)', 'O', line)
    # Add more corrections as needed
    return line

def preprocess_ocr_lines(ocr_lines):
    cleaned_lines = []
    for line in ocr_lines:
        cleaned = clean_ocr_line(line)
        if cleaned:  # Skip empty lines
            cleaned_lines.append(cleaned)
    return cleaned_lines

def preprocess_text(text):
    # Split text into lines
    lines = text.split('\n')
    # Preprocess each line
    cleaned_lines = preprocess_ocr_lines(lines)
    # Join cleaned lines back into a single string
    cleaned_text = '\n'.join(cleaned_lines)
    return cleaned_text

def preprocess_texts(texts):
    cleaned_texts = []
    for text in texts:
        cleaned = preprocess_text(text)
        if cleaned:  # Skip empty texts
            cleaned_texts.append(cleaned)
    return cleaned_texts