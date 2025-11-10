import torch
import torch.nn as nn
import torch.optim as optim
from data_loader import dataloader64, dataloader128
from model import GeneratorStage1, DiscriminatorStage1, GeneratorStage2, DiscriminatorStage2
from tqdm import tqdm
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

gen1 = GeneratorStage1().to(device)
disc1 = DiscriminatorStage1().to(device)
g1_opt = optim.Adam(gen1.parameters(), lr=0.0002, betas=(0.5, 0.999))
d1_opt = optim.Adam(disc1.parameters(), lr=0.0002, betas=(0.5, 0.999))

gen2 = GeneratorStage2().to(device)
disc2 = DiscriminatorStage2().to(device)
g2_opt = optim.Adam(gen2.parameters(), lr=0.0002, betas=(0.5, 0.999))
d2_opt = optim.Adam(disc2.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCEWithLogitsLoss()


def train_stage1(epochs=10):
    for epoch in range(epochs):
        gen1.train()
        disc1.train()
        loop = tqdm(dataloader64)
        for real_imgs, text_embs, _ in loop:
            real_imgs, text_embs = real_imgs.to(device), text_embs.to(device)
            batch_size = real_imgs.size(0)
            # Train discriminator
            d1_opt.zero_grad()
            noise = torch.randn(batch_size, 100, device=device)
            fake_imgs = gen1(noise, text_embs)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            d1_loss_real = criterion(disc1(real_imgs, text_embs), real_labels)
            d1_loss_fake = criterion(disc1(fake_imgs.detach(), text_embs), fake_labels)
            d1_loss = (d1_loss_real + d1_loss_fake) / 2
            d1_loss.backward()
            d1_opt.step()

            # Train generator
            g1_opt.zero_grad()
            g1_loss = criterion(disc1(fake_imgs, text_embs), real_labels)
            g1_loss.backward()
            g1_opt.step()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(D_loss=d1_loss.item(), G_loss=g1_loss.item())

        # Save checkpoint
        torch.save(gen1.state_dict(), f'generator_stage1_epoch_{epoch+1}.pth')
        torch.save(disc1.state_dict(), f'discriminator_stage1_epoch_{epoch+1}.pth')


def train_stage2(epochs=10):
    # Load best Stage1 generator
    gen1.eval()
    for epoch in range(epochs):
        gen2.train()
        disc2.train()
        loop = tqdm(dataloader128)
        for real_imgs, text_embs, _ in loop:
            real_imgs, text_embs = real_imgs.to(device), text_embs.to(device)
            batch_size = real_imgs.size(0)
            
            # Generate 64x64 fake images as input to Stage2 Gen
            noise = torch.randn(batch_size, 100, device=device)
            with torch.no_grad():
                fake_imgs64 = gen1(noise, text_embs)
            
            # Train discriminator Stage2
            d2_opt.zero_grad()
            fake_imgs128 = gen2(fake_imgs64.detach(), text_embs)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            d2_loss_real = criterion(disc2(real_imgs, text_embs), real_labels)
            d2_loss_fake = criterion(disc2(fake_imgs128.detach(), text_embs), fake_labels)
            d2_loss = (d2_loss_real + d2_loss_fake) / 2
            d2_loss.backward()
            d2_opt.step()

            # Train generator Stage2
            g2_opt.zero_grad()
            g2_loss = criterion(disc2(fake_imgs128, text_embs), real_labels)
            g2_loss.backward()
            g2_opt.step()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(D_loss=d2_loss.item(), G_loss=g2_loss.item())

        # Save checkpoint
        torch.save(gen2.state_dict(), f'generator_stage2_epoch_{epoch+1}.pth')
        torch.save(disc2.state_dict(), f'discriminator_stage2_epoch_{epoch+1}.pth')


if __name__ == "__main__":
    print("Starting Stage 1 Training")
    train_stage1(epochs=10)   # Adjust epochs depending on your setup
    print("Starting Stage 2 Training")
    train_stage2(epochs=10)
