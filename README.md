# Text-to-Image GAN Generator (MS-COCO with StackGAN)

This project implements a text-to-image generator trained on the MS-COCO dataset using a two-stage StackGAN architecture, designed to generate high-quality 128x128 images conditioned on textual captions.

---

## Features

- Generate realistic 128x128 images from natural language text.
- Two-stage GAN architecture refines images progressively for better details.
- Uses CLIP embeddings for rich semantic text representation.
- Interactive Gradio interface for easy text input and multimodal image generation.
- Regenerate images multiple times per caption for diverse outputs.
- Adjustable creativity slider affecting noise level and variation.

---

## Demo Interface Overview

- **Caption input**: Enter a detailed text description (e.g., "a red sports car speeding on a highway at sunset").
- **Number of Images to Generate**: Choose how many images to generate simultaneously (1 to 5).
- **Creativity (Noise Level)**: Higher noise values produce more varied and creative outputs.
- **Generate Images** button: Triggers image generation from caption with specified parameters.
- **Refine (Regenerate)** button: Repeats generation with the same input to explore variations.
- **Output Gallery**: Displays generated images with respective match scores (note: in your current setup, errors indicate model or loading issues; ensure your weights and environment are correctly set up).

---

## Typical Errors and Solutions

If you see `Error` placeholders instead of images:

- Verify that your generator weights file (`generator_final.pth` or equivalent) exists, is correctly located, and compatible with the code.
- Ensure that all required dependencies are installed correctly and that your Python environment matches the project requirements.
- Check for GPU availability or adjust batch sizes and epochs if running on CPU to avoid timeout or memory issues.
- Restart the Gradio app after fixing the above and retry.

---

## Installation

1. Clone the repository or copy the source files.
2. Install dependencies:

```bash
pip install torch torchvision clip-by-openai gradio pillow numpy
