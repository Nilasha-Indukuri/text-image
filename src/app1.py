import gradio as gr
from generate import generate_image_from_text

def generate(text):
    return generate_image_from_text(text)

with gr.Blocks() as demo:
    gr.Markdown(f"## {title}\n\n{desc}")
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt", placeholder="e.g. red car", lines=2, value=" ")
            steps = gr.Slider(minimum=1, maximum=50, value=20, step=1, label="Steps")
            guidance = gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.5, label="Guidance scale")
            seed = gr.Number(value=-1, label="Seed (-1 for random)")
            width = gr.Slider(128, 1024, value=512, step=64, label="Width (px)")
            height = gr.Slider(128, 1024, value=512, step=64, label="Height (px)")
            btn = gr.Button("Generate")

        with gr.Column(scale=3):
            output = gr.Image(type="pil", label="Generated Image")
            download_btn = gr.Button("Download Image")
            download_file_output = gr.File(label="Download generated image", interactive=False, file_count="single")

    def on_generate(p, s, g, sd, w, h):
        sd_int = int(sd) if sd is not None else -1
        img = generate(p, steps=s, guidance=g, seed=sd_int, width=w, height=h)
        return img

    btn.click(on_generate, inputs=[prompt, steps, guidance, seed, width, height], outputs=[output])
    download_btn.click(lambda img: img, inputs=[output], outputs=[download_file_output])
    
if __name__ == "__main__":
    app.launch()
