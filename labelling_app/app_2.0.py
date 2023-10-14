import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from skimage import io
from skimage.color import gray2rgb
from functions_app import single_input_image, weighted_iou_loss, resize_image
import cv2
import os

class SemanticSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Semantic Segmentation App")

        # Load trained neural network
        self.model = tf.keras.models.load_model("MRI_Segmentation_Model.h5", custom_objects={'weighted_iou_loss': weighted_iou_loss})

        self.toolbar_frame = tk.Frame(root)
        self.toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.canvas_frame, cursor="fleur")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self.zoom)
        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan)
        self.canvas.bind("<ButtonRelease-2>", self.stop_pan)

        self.label_frame = tk.Frame(self.toolbar_frame)
        self.label_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.label_var = tk.StringVar()
        self.label_var.set("Select Label:")

        self.label_menu = tk.OptionMenu(self.label_frame, self.label_var, "Muscle", "Fat", "Bone")
        self.label_menu.pack(fill=tk.X)

        self.load_button = tk.Button(self.toolbar_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(self.toolbar_frame, text="Save Annotations", command=self.save_annotations)
        self.save_button.pack(side=tk.LEFT)

        self.brush_size = tk.IntVar()
        self.brush_size.set(15) # Initial brush size 

        self.image = None
        self.annotation = None
        self.drawing = False
        self.eraser_mode = False
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.pan_start_x = None
        self.pan_start_y = None

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.label_colors = {"Muscle": (255, 0, 0), "Fat": (0, 255, 0), "Bone": (0, 0, 255)}

        self.brush_size_label = tk.Label(self.toolbar_frame, text="Brush Size:")
        self.brush_size_label.pack(side=tk.LEFT)

        self.brush_size_scale = tk.Scale(self.toolbar_frame, from_=1, to=20, orient=tk.HORIZONTAL, variable=self.brush_size)
        self.brush_size_scale.pack(side=tk.LEFT)
        self.brush_size_scale.bind("<Motion>", self.show_brush_size)

        self.eraser_button = tk.Button(self.toolbar_frame, text="Eraser", command=self.toggle_eraser)
        self.eraser_button.pack(side=tk.LEFT)

    def load_image(self):
        default_dir = r'C:\Users\n10766316\Desktop\unlabelledMRIs'
        self.file_path = filedialog.askopenfilename(initialdir=default_dir, filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if self.file_path:
            self.image = cv2.imread(self.file_path)
            if len(self.image.shape) == 2:
                self.image = gray2rgb(self.image)
            self.annotation = np.zeros_like(self.image)
            self.pan_x = 0
            self.pan_y = 0
            self.show_image()
            self.root.geometry(f"{900}x{900 + self.toolbar_frame.winfo_height()}")
            self.annotation = self.segment_image().astype(self.image.dtype)
            print(os.path.basename(self.file_path))

    def show_image(self):
        if self.image is not None:
            img_rgb = self.image.copy()
            img_rgb = cv2.addWeighted(img_rgb, 1, self.annotation, 0.5, 0)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil = img_pil.resize((int(img_pil.width * self.zoom_factor), int(img_pil.height * self.zoom_factor)), Image.BILINEAR)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.canvas.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk

    def start_drawing(self, event):
        if self.image is not None:
            self.drawing = True
            x, y = event.x - self.pan_x, event.y - self.pan_y
            x, y = self.zoom_coordinates(x, y)
            self.draw_brush(x, y)

    def draw(self, event):
        if self.drawing:
            x, y = event.x - self.pan_x, event.y - self.pan_y
            x, y = self.zoom_coordinates(x, y)
            self.draw_brush(x, y)

    def draw_brush(self, x, y):
        if self.label_var.get() is not None:
            if self.eraser_mode:
                label_color = (0, 0, 0)
            else:
                label_color = self.label_colors.get(self.label_var.get(), (255, 255, 255))
            brush_size = self.brush_size.get()
            cv2.circle(self.annotation, (x, y), int(brush_size / self.zoom_factor), label_color, -1)
            self.show_image()

    def stop_drawing(self, event):
        self.drawing = False

    def show_brush_size(self, event):
        brush_size = self.brush_size.get()
        self.brush_size_label.config(text=f"Brush Size: {brush_size}")

    def toggle_eraser(self):
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self.eraser_button.config(text="Draw")
        else:
            self.eraser_button.config(text="Eraser")

    def save_annotations(self):
        if self.annotation is not None:
            default_dir = r'C:\Users\n10766316\Desktop\imJLabels'
            file_path = filedialog.asksaveasfilename(initialdir=default_dir, defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                x = self.annotation
                muscle = 1*(x[:,:,0]>100)
                fat = 2*(x[:,:,1]>100)
                bone = 3*(x[:,:,2]>100)
                mask = muscle+fat+bone
                cv2.imwrite(file_path, mask)

    def zoom(self, event):
        if self.image is not None:
            if event.delta > 0:
                self.zoom_factor *= 1.1
            else:
                self.zoom_factor /= 1.1
            #self.zoom_factor = max(0.1, min(3.0, self.zoom_factor))
            self.show_image()

    def start_pan(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def pan(self, event):
        if self.pan_start_x is not None and self.pan_start_y is not None:
            delta_x = event.x - self.pan_start_x
            delta_y = event.y - self.pan_start_y
            self.pan_x += delta_x
            self.pan_y += delta_y
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self.show_image()

    def stop_pan(self, event):
        self.pan_start_x = None
        self.pan_start_y = None

    def zoom_coordinates(self, x, y):
        x = int(x / self.zoom_factor)
        y = int(y / self.zoom_factor)
        return x, y
    
    # NEURAL NET STUFF
    def segment_image(self):
        # Load the MRI image
        image = self.image #io.imread(self.file_path)
        (a, b, _) = image.shape

        net_input = single_input_image(image)
        net_input = np.array([net_input])

        # Perform segmentation using the model
        output = self.model.predict(net_input)

        print(np.shape(output.squeeze()[:,:,1:4]))
        io.imshow(output.squeeze()[:,:,1:4])
        mask_r = output.squeeze()[:,:,1]>0.5
        mask_g = output.squeeze()[:,:,2]>0.5
        mask_b = output.squeeze()[:,:,3]>0.5

        result = np.zeros((output.shape[1], output.shape[2], 3), dtype=np.float32)

        # Apply the rules
        result[(mask_r & mask_g & mask_b)] = [1, 0, 0]  # R and G and B is R
        result[(mask_r & mask_g & ~mask_b)] = [0, 1, 0]  # R and G is G
        result[(mask_r & mask_b & ~mask_g)] = [1, 0, 0]  # R and B is R
        result[(mask_g & mask_b & ~mask_r)] = [0, 1, 0]  # G and B is G
        result[(mask_r & ~mask_g & ~mask_b)] = [1, 0, 0]  # R only
        result[(mask_g & ~mask_r & ~mask_b)] = [0, 1, 0]  # G only
        result[(mask_b & ~mask_r & ~mask_g)] = [0, 0, 1]  # B only

        io.imshow(result)
        return resize_image(255*result, (a,b))

if __name__ == "__main__":
    root = tk.Tk()
    app = SemanticSegmentationApp(root)
    root.mainloop()