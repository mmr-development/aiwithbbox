import os
import tkinter as tk
from tkinter import filedialog, messagebox
from pdf2image import convert_from_path

OUTPUT_DIR = "training_images"

def save_pdf_as_images(pdf_path, output_dir=OUTPUT_DIR, dpi=300, poppler_path=None):
    os.makedirs(output_dir, exist_ok=True)
    pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
    images = convert_from_path(pdf_path, dpi=300, poppler_path=r"C:/poppler/poppler-24.08.0/Library/bin")
    saved_files = []
    for i, img in enumerate(images, start=1):
        out_path = os.path.join(output_dir, f"{pdf_base}_page_{i}.png") 
        img.save(out_path)
        saved_files.append(out_path)
    return saved_files

def select_pdf_and_save():
    pdf_path = filedialog.askopenfilename(
        title="Select PDF file",
        filetypes=[("PDF Files", "*.pdf")]
    )
    if not pdf_path:
        return
    try:
        saved_files = save_pdf_as_images(pdf_path)
        messagebox.showinfo("Success", f"Saved {len(saved_files)} images to '{OUTPUT_DIR}'.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to convert PDF:\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("PDF to Training Images")
    root.geometry("400x200")
    tk.Label(root, text="Convert PDF to images for annotation", font=("Arial", 14)).pack(pady=20)
    tk.Button(root, text="Select PDF and Save Images", command=select_pdf_and_save, font=("Arial", 12)).pack(pady=20)
    root.mainloop()