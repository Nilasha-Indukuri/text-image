import matplotlib.pyplot as plt
import torch
import clip
import numpy as np
from PIL import Image
from model import GeneratorStage1, GeneratorStage2

device = "cuda" if torch.cuda.is_available() else "cpu"

gen1 = GeneratorStage1().to(device)
gen2 = GeneratorStage2().to(device)

gen1.load_state_dict(torch.load('generator_stage1_epoch_10.pth', map_location=device))  # Adjust path if needed
gen2.load_state_dict(torch.load('generator_stage2_epoch_10.pth', map_location=device))

gen1.eval()
gen2.eval()

clip_model, _ = clip.load("ViT-B/32", device=device)

def generate_image_from_text(text, noise_dim=100):
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_emb = clip_model.encode_text(text_tokens).float()

    noise = torch.randn(1, noise_dim).to(device)
    with torch.no_grad():
        fake_img64 = gen1(noise, text_emb)
        fake_img128 = gen2(fake_img64, text_emb)

    img = fake_img128.squeeze().cpu().numpy()
    img = (img + 1.0) / 2.0  # Normalize to [0,1]
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))  # CHW to HWC
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

if __name__ == "__main__":
    example_text = "a red sports car speeding on a highway at sunset"
    image = generate_image_from_text(example_text)
    image.show()
    plt.imshow(image)
    plt.axis('off')
    plt.show()  