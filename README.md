Text-to-Image Synthesis using GANs and Skip Thought Vectors
Project Overview

This project implements a Text-to-Image Synthesis model using Generative Adversarial Networks (GAN-CLS) and Skip Thought Vectors to generate realistic images directly from text captions.
The system learns the relationship between descriptive language and visual features using the Oxford 102 Flowers Dataset, producing images that match input text descriptions.

⚙️ Installation and Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/paarthneekhara/text-to-image.git
cd text-to-image

2️⃣ Create a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Verify Setup

Run the setup notebook to confirm environment and data access:

jupyter notebook notebooks/setup.ipynb


You should see:

Python and TensorFlow versions printed

Sample flower image displayed

Caption examples and summary plots

📊 Dataset Information

Dataset: Oxford 102 Flowers Dataset

Type: Image + text (captions)
Structure:

data/
 ├── flowers/
 │   ├── jpg/                # flower images
 │   └── text_c10/           # captions per image
 └── skipthoughts/           # pretrained Skip Thought model files


How to Obtain Data:
Run the provided dataset download script:

python download_datasets.py


Alternatively, manually download:

Images: from Oxford Flowers dataset site

Captions: from the provided Google Drive link

Skip Thought Models: from Skip-Thoughts repository

Preprocessing Includes:

Image resizing to 64×64 pixels

Caption embedding via Skip Thought Vectors

Alignment between text and corresponding image

🚀 How to Run the Project
1️⃣ Train the Model
python train.py --data_set="flowers"


Options:

--z_dim: Noise dimension (default: 100)

--t_dim: Text feature dimension (default: 256)

--batch_size: Training batch size (default: 64)

--epochs: Total epochs (default: 600)

2️⃣ Generate Images from Captions

Write sample captions in Data/sample_captions.txt, then run:

python generate_thought_vectors.py --caption_file="Data/sample_captions.txt"
python generate_images.py --model_path="Data/Models/latest_model_flowers_temp.ckpt" --n_images=5


Generated outputs will be saved under:

data/val_samples/

🖥️ User Interface (Coming Soon)

A Streamlit-based interface will allow users to:

Input text captions

Generate and visualize corresponding images

Save or download generated results

👨‍💻 Author Information

Author: Venkata Krishna Raj Abhishek Gade
Institution: Northeastern University, Boston
Email: gade.v@northeastern.edu

GitHub: https://github.com/paarthneekhara/text-to-image
