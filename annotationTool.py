import os
import json
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import pytesseract

IMAGES_DIR = "training_images"
TRAINING_DATA_DIR = "training_data"
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

class AnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Menu Annotation Tool")
        self.image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.current_image_idx = 0
        self.annotations = []
        self.bbox = None
        self.start_x = self.start_y = 0

        # Set window size to max screen size
        screen_width = root.winfo_screenwidth() - 100
        screen_height = root.winfo_screenheight() - 100
        root.geometry(f"{screen_width}x{screen_height}")

        # Scrollable canvas setup
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.v_scroll = tk.Scrollbar(self.frame, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll = tk.Scrollbar(self.frame, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(self.frame, cursor="cross",
                               yscrollcommand=self.v_scroll.set,
                               xscrollcommand=self.h_scroll.set,
                               bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        btn_frame = tk.Frame(root)
        btn_frame.pack(fill=tk.X)
        tk.Button(btn_frame, text="Prev Image", command=self.prev_image).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Next Image", command=self.next_image).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Save Annotations", command=self.save_annotations).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Clear Boxes", command=self.clear_boxes).pack(side=tk.LEFT)

        self.load_image()

    def load_image(self):
        self.annotations = []
        self.canvas.delete("all")
        if not self.image_files:
            messagebox.showinfo("Info", "No images found in images folder.")
            self.root.quit()
            return
        img_path = os.path.join(IMAGES_DIR, self.image_files[self.current_image_idx])
        self.img = Image.open(img_path)
        self.tk_img = ImageTk.PhotoImage(self.img)
        # Set scroll region and display image
        self.canvas.config(scrollregion=(0, 0, self.tk_img.width(), self.tk_img.height()))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.root.title(f"Annotating: {self.image_files[self.current_image_idx]}")

    def on_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.bbox = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_drag(self, event):
        cur_x = self.canvas.canvasx(event.x)
        cur_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.bbox, self.start_x, self.start_y, cur_x, cur_y)

    def on_release(self, event):
        x1, y1 = self.start_x, self.start_y
        x2, y2 = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        x1, x2 = sorted([int(x1), int(x2)])
        y1, y2 = sorted([int(y1), int(y2)])
        cropped = self.img.crop((x1, y1, x2, y2))
        extracted_text = pytesseract.image_to_string(cropped).strip()
        self.show_annotation_type_dialog(x1, y1, x2, y2, extracted_text)

    def show_annotation_type_dialog(self, x1, y1, x2, y2, extracted_text):
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Annotation Type")
        dialog.geometry("400x200")
        tk.Label(dialog, text="Select annotation type:").pack(pady=10)

        def add_category():
            dialog.destroy()
            name = self.ask_large_input("Category Name", "Category Name:", extracted_text)
            if name:
                self.annotations.append({"type": "category", "name": name, "bbox": [x1, y1, x2, y2]})

        def add_item():
            dialog.destroy()
            name = self.ask_large_input("Item Name", "Item Name:", extracted_text)
            if name:
                self.annotations.append({"type": "item", "name": name, "bbox": [x1, y1, x2, y2]})

        def add_price():
            dialog.destroy()
            price = self.ask_large_input("Price", "Price:", extracted_text)
            if price:
                self.annotations.append({"type": "price", "price": price, "bbox": [x1, y1, x2, y2]})

        def add_description():
            dialog.destroy()
            description = self.ask_large_input("Description", "Description:", extracted_text)
            if description:
                self.annotations.append({"type": "description", "description": description, "bbox": [x1, y1, x2, y2]})

        btn_frame = tk.Frame(dialog)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Category", width=12, command=add_category).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Item", width=12, command=add_item).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Price", width=12, command=add_price).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Description", width=12, command=add_description).pack(side=tk.LEFT, padx=5)

        def on_close():
            dialog.destroy()
            self.canvas.delete(self.bbox)
        dialog.protocol("WM_DELETE_WINDOW", on_close)

    def ask_large_input(self, title, prompt, initialvalue):
        # Custom dialog with larger input box
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.geometry("400x200")
        tk.Label(dialog, text=prompt).pack(pady=10)
        text_box = tk.Text(dialog, height=4, width=40)
        text_box.pack()
        text_box.insert(tk.END, initialvalue)
        text_box.focus_set()
        result = {"value": None}

        def on_ok():
            result["value"] = text_box.get("1.0", tk.END).strip()
            dialog.destroy()

        tk.Button(dialog, text="OK", command=on_ok).pack(pady=10)
        dialog.wait_window()
        return result["value"]

    def save_annotations(self):
        img_name = os.path.splitext(self.image_files[self.current_image_idx])[0]
        save_path = os.path.join(TRAINING_DATA_DIR, f"{img_name}.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)
        messagebox.showinfo("Saved", f"Annotations saved to {save_path}")

    def next_image(self):
        if self.current_image_idx < len(self.image_files) - 1:
            self.current_image_idx += 1
            self.load_image()

    def prev_image(self):
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.load_image()

    def clear_boxes(self):
        self.annotations = []
        self.load_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationTool(root)
    root.mainloop()