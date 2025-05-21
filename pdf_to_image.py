from pdf2image import convert_from_path
import os
import sys
from PIL import Image
import io
import base64
import tempfile
import PyPDF2


def pdf_to_image(pdf_path, output_dir):
    # if pdf_path is url, download the pdf
    if pdf_path.startswith('http'):
        import requests
        response = requests.get(pdf_path)
        pdf_path = os.path.join(tempfile.gettempdir(), 'temp.pdf')
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
    # Check if the PDF file exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert PDF to images
    images = convert_from_path(pdf_path)
    # Save images to the output directory
    image_paths = []
    for i, image in enumerate(images):
        # Save to output_dir
        image_path = os.path.join(output_dir, f"page_{i + 1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)

        # Also save to training_data/images with unique name
        training_images_dir = os.path.join('training_images')
        if not os.path.exists(training_images_dir):
            os.makedirs(training_images_dir)
        unique_image_name = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i + 1}.png"
        training_image_path = os.path.join(training_images_dir, unique_image_name)
        image.save(training_image_path, "PNG")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths


def pdf_to_image_exportable(pdf_path, output_dir):
    try:
        image_paths = pdf_to_image(pdf_path, output_dir)
        return image_paths
    except Exception as e:
        print(f"Error: {e}")
        return None