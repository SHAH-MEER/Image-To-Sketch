import gradio as gr
import cv2
import numpy as np
from PIL import Image

def convert_to_sketch(image: Image.Image):
    # Convert PIL to OpenCV format (numpy array)
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_img = cv2.bitwise_not(gray_img)
    blurred_img = cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0, sigmaY=0)
    inverted_blur_img = cv2.bitwise_not(blurred_img)
    sketch_img = cv2.divide(gray_img, inverted_blur_img, scale=256.0)

    # Make lines darker by enhancing contrast
    darker_sketch = cv2.multiply(sketch_img, np.array([0.7], dtype=np.float32))
    darker_sketch = np.clip(darker_sketch, 0, 255).astype(np.uint8)

    # Convert back to PIL Image
    sketch_pil = Image.fromarray(darker_sketch)
    return sketch_pil

examples = ['examples/parrot.jpeg', 'examples/flower2.jpeg', 'examples/cat.jpeg']

with gr.Blocks(title="Pencil Sketch Converter",theme=gr.themes.Monochrome()) as app:
    gr.Markdown("# 🎨 Pencil Sketch Converter")
    gr.Markdown("Upload an image and get a pencil sketch version with darker lines.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload Image",
                type="pil",
                height=400
            )
        with gr.Column():
            output_image = gr.Image(
                label="Pencil Sketch",
                type="pil",
                height=400
            )
    with gr.Row():
        gr.Examples(
            examples=examples,
            inputs=input_image,
            outputs=output_image,
            fn=convert_to_sketch,
            cache_examples=True
        )
    input_image.change(
        fn=convert_to_sketch,
        inputs=input_image,
        outputs=output_image
    )

app.launch(inbrowser=True)
