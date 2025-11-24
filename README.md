Text-to-Image GAN Generator (MS-COCO with StackGAN)

This project implements a text-to-image generator trained on the MS-COCO dataset using a two-stage StackGAN architecture, designed to produce 128×128 images from natural-language captions.
The pipeline integrates CLIP text embeddings, a two-stage GAN refinement process, and an interactive Gradio interface for real-time generation.

🌐 Project Pipeline Overview

The complete system pipeline consists of four major stages:

Data Preparation → Loading and preprocessing MS-COCO captions and images

Text Embedding → Encoding captions using OpenAI’s CLIP model

Model Training → Training Stage 1 (64×64) and Stage 2 (128×128) generators with adversarial loss

Interface Deployment → Running a Gradio web UI for text-to-image synthesis

🧩 Architecture Diagram
graph LR
A[Text Caption (MS-COCO)] --> B[CLIP Text Encoder (512-D Embedding)]
B --> C[Stage 1 Generator (64×64 Coarse Image)]
C --> D[Stage 2 Generator (128×128 Refined Image)]
D --> E[Gradio Interface]
E --> F[User Output (Generated Image Gallery)]
C -.-> C1[Stage 1 Discriminator]
D -.-> D1[Stage 2 Discriminator]

⚙️ Features

Generate realistic 128×128 images from descriptive text.

Two-stage refinement improves clarity and detail progressively.

Incorporates CLIP embeddings for semantically rich text understanding.

Provides an interactive Gradio UI for easy experimentation.

Adjustable creativity slider controls the generator’s noise variance.

Supports checkpoint resumption for continuing training from saved weights.

🧠 Pipeline Components
1. Data Preparation

Download and organize the MS-COCO 2017 dataset:

mkdir data && cd data
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip


Preprocess captions and generate CLIP embeddings:

python preprocess_data.py --data_dir data/val2017 --captions annotations/captions_val2017.json


This creates .npy files of pre-encoded caption embeddings to accelerate training.

2. Model Training

Train the StackGAN in two stages:

Stage 1:

python train_stage1.py --epochs 10 --batch_size 16 --lr 0.0002


Stage 2:

python train_stage2.py --epochs 10 --batch_size 16 --lr 0.0002 --resume_from checkpoints/stage1_last.pth


Training Details:

Optimizer: Adam (β₁ = 0.5, β₂ = 0.999)

Loss: Binary Cross-Entropy (GAN loss) + Feature Matching

Dataset: MS-COCO 2017 (5 000 images subset)

Typical runtime (CPU): ~8 hrs total; GPU: ~30–45 min

Checkpoints saved automatically every epoch to /checkpoints/

To resume training:

python train_stage2.py --resume_from checkpoints/stage2_last.pth

3. Evaluation

To quickly evaluate the trained model:

python evaluate_model.py --metric fid --samples 500


Currently supported metrics:

FID (Fréchet Inception Distance)

Inception Score (IS)

CLIP-based similarity (optional qualitative metric)

These scores quantify how close generated images are to real MS-COCO images.

4. Gradio Interface

Launch the text-to-image generator UI:

python app.py


Then open the displayed local or public URL to test captions like:

"a red sports car speeding on a highway at sunset"
"a brown dog playing with a ball in the park"

🧮 Typical Errors & Solutions
Issue	Cause	Solution
Error placeholder in output	Model weights missing or mismatched	Verify generator .pth files and model definitions
Memory overflow	Batch size too large	Reduce --batch_size or lower image resolution
Slow inference	Running on CPU	Use a CUDA-enabled GPU or reduce image count
Blank images	CLIP embeddings not found	Regenerate embeddings via preprocess_data.py
📊 Quantitative Metrics (for next report)
Metric	Definition	Ideal Range
FID	Measures similarity between generated and real images	Lower is better (< 50 for early tests)
IS	Measures image diversity and class confidence	Higher is better (> 5 for realistic samples)
CLIP Similarity	Cosine similarity between text and generated image	Higher indicates better text-image alignment

Include these metrics in your report using even small test subsets (e.g., 500 images) to validate progress quantitatively.

🔋 Reproducibility Checklist

Python ≥ 3.10

PyTorch ≥ 2.0

CLIP (by OpenAI)

Gradio ≥ 3.0

Pillow, NumPy, Matplotlib

Install all requirements:

pip install -r requirements.txt

🧩 Future Improvements

Extend training to 50+ epochs for finer detail.

Integrate attention modules (e.g., AttnGAN).

Add progressive growing for higher resolutions (256×256+).

Deploy lightweight version on mobile or edge devices.

Collect user feedback through the Gradio interface logs.
Text-to-Image GAN Generator (MS-COCO with StackGAN)

This project implements a text-to-image generator trained on the MS-COCO dataset using a two-stage StackGAN architecture, designed to produce 128×128 images from natural-language captions.
The pipeline integrates CLIP text embeddings, a two-stage GAN refinement process, and an interactive Gradio interface for real-time generation.

🌐 Project Pipeline Overview

The complete system pipeline consists of four major stages:

Data Preparation → Loading and preprocessing MS-COCO captions and images

Text Embedding → Encoding captions using OpenAI’s CLIP model

Model Training → Training Stage 1 (64×64) and Stage 2 (128×128) generators with adversarial loss

Interface Deployment → Running a Gradio web UI for text-to-image synthesis

🧩 Architecture Diagram
graph LR
A[Text Caption (MS-COCO)] --> B[CLIP Text Encoder (512-D Embedding)]
B --> C[Stage 1 Generator (64×64 Coarse Image)]
C --> D[Stage 2 Generator (128×128 Refined Image)]
D --> E[Gradio Interface]
E --> F[User Output (Generated Image Gallery)]
C -.-> C1[Stage 1 Discriminator]
D -.-> D1[Stage 2 Discriminator]

⚙️ Features

Generate realistic 128×128 images from descriptive text.

Two-stage refinement improves clarity and detail progressively.

Incorporates CLIP embeddings for semantically rich text understanding.

Provides an interactive Gradio UI for easy experimentation.

Adjustable creativity slider controls the generator’s noise variance.

Supports checkpoint resumption for continuing training from saved weights.

🧠 Pipeline Components
1. Data Preparation

Download and organize the MS-COCO 2017 dataset:

mkdir data && cd data
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip


Preprocess captions and generate CLIP embeddings:

python preprocess_data.py --data_dir data/val2017 --captions annotations/captions_val2017.json


This creates .npy files of pre-encoded caption embeddings to accelerate training.

2. Model Training

Train the StackGAN in two stages:

Stage 1:

python train_stage1.py --epochs 10 --batch_size 16 --lr 0.0002


Stage 2:

python train_stage2.py --epochs 10 --batch_size 16 --lr 0.0002 --resume_from checkpoints/stage1_last.pth


Training Details:

Optimizer: Adam (β₁ = 0.5, β₂ = 0.999)

Loss: Binary Cross-Entropy (GAN loss) + Feature Matching

Dataset: MS-COCO 2017 (5 000 images subset)

Typical runtime (CPU): ~8 hrs total; GPU: ~30–45 min

Checkpoints saved automatically every epoch to /checkpoints/

To resume training:

python train_stage2.py --resume_from checkpoints/stage2_last.pth

3. Evaluation

To quickly evaluate the trained model:

python evaluate_model.py --metric fid --samples 500


Currently supported metrics:

FID (Fréchet Inception Distance)

Inception Score (IS)

CLIP-based similarity (optional qualitative metric)

These scores quantify how close generated images are to real MS-COCO images.

4. Gradio Interface

Launch the text-to-image generator UI:

python app.py


Then open the displayed local or public URL to test captions like:

"a red sports car speeding on a highway at sunset"
"a brown dog playing with a ball in the park"

🧮 Typical Errors & Solutions
Issue	Cause	Solution
Error placeholder in output	Model weights missing or mismatched	Verify generator .pth files and model definitions
Memory overflow	Batch size too large	Reduce --batch_size or lower image resolution
Slow inference	Running on CPU	Use a CUDA-enabled GPU or reduce image count
Blank images	CLIP embeddings not found	Regenerate embeddings via preprocess_data.py
📊 Quantitative Metrics (for next report)
Metric	Definition	Ideal Range
FID	Measures similarity between generated and real images	Lower is better (< 50 for early tests)
IS	Measures image diversity and class confidence	Higher is better (> 5 for realistic samples)
CLIP Similarity	Cosine similarity between text and generated image	Higher indicates better text-image alignment

Include these metrics in your report using even small test subsets (e.g., 500 images) to validate progress quantitatively.

🔋 Reproducibility Checklist

Python ≥ 3.10

PyTorch ≥ 2.0

CLIP (by OpenAI)

Gradio ≥ 3.0

Pillow, NumPy, Matplotlib

Install all requirements:

pip install -r requirements.txt

🧩 Future Improvements

Extend training to 50+ epochs for finer detail.

Integrate attention modules (e.g., AttnGAN).

Add progressive growing for higher resolutions (256×256+).

Deploy lightweight version on mobile or edge devices.

Collect user feedback through the Gradio interface logs.
