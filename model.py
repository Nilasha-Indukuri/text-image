import torch
import torch.nn as nn

# Stage 1 Generator - 64x64 image generation
class GeneratorStage1(nn.Module):
    def __init__(self, noise_dim=100, text_dim=512, img_channels=3):
        super(GeneratorStage1, self).__init__()
        self.fc = nn.Linear(noise_dim + text_dim, 1024 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1), # 8x8
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, img_channels, 4, 2, 1), # 64x64
            nn.Tanh()
        )
    
    def forward(self, noise, text_emb):
        x = torch.cat([noise, text_emb], dim=1)
        x = self.fc(x).view(-1, 1024, 4, 4)
        img = self.net(x)
        return img


# Stage 1 Discriminator - for 64x64 images
class DiscriminatorStage1(nn.Module):
    def __init__(self, text_dim=512, img_channels=3):
        super(DiscriminatorStage1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1), # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1), # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1), # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(512*4*4 + text_dim, 1)

    def forward(self, img, text_emb):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, text_emb], dim=1)
        out = self.fc(x)
        return out


# Stage 2 Generator - 128x128 refinement
class GeneratorStage2(nn.Module):
    def __init__(self, text_dim=512, img_channels=3):
        super(GeneratorStage2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1), # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1), #8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(256*8*8 + text_dim, 512*8*8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # 16x16
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1), # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, img_channels, 4, 2, 1), # 128x128
            nn.Tanh()
        )
    
    def forward(self, img64, text_emb):
        x = self.encoder(img64)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, text_emb], dim=1)
        x = self.fc(x)
        x = x.view(-1, 512, 8, 8)
        img128 = self.decoder(x)
        return img128


# Stage 2 Discriminator - for 128x128 images
class DiscriminatorStage2(nn.Module):
    def __init__(self, text_dim=512, img_channels=3):
        super(DiscriminatorStage2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1), # 64x64
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1), # 32x32
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1), # 16x16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1), # 8x8
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1), # 4x4
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(1024*4*4 + text_dim, 1)

    def forward(self, img, text_emb):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, text_emb], dim=1)
        out = self.fc(x)
        return out
