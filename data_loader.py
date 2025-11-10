import os
import torch
import clip
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from pycocotools.coco import COCO

# Choose dataset: 'val2017' (small, ~5k images) or 'train2017' (large, ~118k images)
subset = 'val2017'  # Change to 'train2017' for full dataset
data_dir = f'/Users/praneethvempati/Desktop/Nilasha_GANs/data/coco/'
os.makedirs(data_dir, exist_ok=True)

# Download images
img_url = f'http://images.cocodataset.org/zips/val2017.zip'
if not os.path.exists(f'{data_dir}/images'):
    os.system(f'wget -P ./data {img_url}')
    os.system(f'unzip ./data/val2017.zip -d ./data/coco')
    os.rename(f'/Users/praneethvempati/Desktop/Nilasha_GANs/data/coco/val2017', f'{data_dir}/images')

# Download annotations (captions)
ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
if not os.path.exists(f'{data_dir}/annotations'):
    os.system(f'wget -P ./data {ann_url}')
    os.system(f'unzip ./data/annotations_trainval2017.zip -d ./data/coco')
    os.rename('/Users/praneethvempati/Desktop/Nilasha_GANs/data/coco/annotations', f'{data_dir}/annotations')

# Load COCO captions
ann_file = f'{data_dir}/annotations/captions_val2017.json'
coco = COCO(ann_file)

# Custom dataset class
class COCODataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())
        
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Get a random caption for this image
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        caption = anns[np.random.randint(len(anns))]['caption']  # Random caption per image
        
        # CLIP text embedding
        text_tokens = clip.tokenize([caption]).to(self.device)
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_tokens).float()
        
        return image, text_embedding.squeeze(), caption

# Transforms
transform64 = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform128 = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Datasets and Dataloaders
dataset64 = COCODataset(f'{data_dir}/images', ann_file, transform=transform64)
dataset128 = COCODataset(f'{data_dir}/images', ann_file, transform=transform128)
dataloader64 = DataLoader(dataset64, batch_size=16, shuffle=True)
dataloader128 = DataLoader(dataset128, batch_size=16, shuffle=True)

if __name__ == "__main__":
    print(f"Dataset loaded ({subset}). Sample size:", len(dataset64))
    sample_img, sample_text_emb, sample_caption = dataset64[0]
    print("Sample caption:", sample_caption)