import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import plotly.express as px
import os

def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_img = px.imshow(img)
        original_img.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        original_img.show()
        process_image(img, file_path)

def process_image(img, file_path):
    # Resize image
    scale_percent = 0.60
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Sharpening Image
    kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(resized, -1, kernel_sharpening)

    # Convert to grayscale
    grayscale = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)

    # Inverting the image
    invs = 255 - grayscale

    # Smoothing the image
    gauss = cv2.GaussianBlur(invs, ksize=(15, 15), sigmaX=0, sigmaY=0)

    # Obtaining the final sketch
    def dodgeV2(image, mask):
        return cv2.divide(image, 255 - mask, scale=256)

    pencil_img = dodgeV2(grayscale, gauss)
    sketch = px.imshow(pencil_img, color_continuous_scale='gray')
    sketch.update_layout(coloraxis_showscale=False)
    sketch.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    sketch.show()

    # Save the output image
    output_path = os.path.splitext(file_path)[0] + "_sketch.png"
    cv2.imwrite(output_path, pencil_img)

def exit_application():
    root.destroy()

# Create a Tkinter window
root = tk.Tk()
root.title("Image Processing")
root.geometry("400x150")  # Increase window size

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=select_image)
select_button.pack(pady=10)

# Create an Exit button
exit_button = tk.Button(root, text="Exit", command=exit_application)
exit_button.pack(pady=10)

root.mainloop()
