import gradio as gr
from generate import generate_image_from_text

def generate(text):
    return generate_image_from_text(text)

with gr.Blocks() as app:
    gr.Markdown("# Text-to-Image GAN Generator (MS-COCO + StackGAN)")
    gr.Markdown("Enter a caption below and generate a 128x128 image. Click 'Regenerate' to try again with the same caption.")
    text_input = gr.Textbox(label="Enter caption", placeholder="Describe your desired image here...")
    generate_btn = gr.Button("Generate")
    regenerate_btn = gr.Button("Regenerate")
    output_image = gr.Image(label="Generated Image")

    generate_btn.click(generate, inputs=text_input, outputs=output_image)
    regenerate_btn.click(generate, inputs=text_input, outputs=output_image)

if __name__ == "__main__":
    app.launch()
