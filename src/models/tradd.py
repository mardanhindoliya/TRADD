import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def create_dct_matrix(N=8):
    matrix = torch.zeros(N, N)

    for k in range(N):
        alpha = (1 / N) ** 0.5 if k == 0 else (2 / N) ** 0.5

        for n in range(N):
            matrix[k, n] = alpha * torch.cos(
                torch.tensor(
                    torch.pi * (2 * n + 1) * k / (2 * N)
                )
            )

    return matrix


class TRADD(nn.Module):

    def __init__(self, num_classes=2, use_dct=True):

        super().__init__()

        self.use_dct = use_dct

        # Spatial branch
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )

        self.spatial_features = backbone.features
        self.spatial_pool = backbone.avgpool

        # ImageNet normalization
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # DCT matrix
        self.register_buffer("dct_matrix", create_dct_matrix())

        # Frequency branch
        self.frequency_branch = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1)
        )

        # Identical classifier for both configurations
        self.classifier = nn.Sequential(
            nn.Linear(576 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def extract_frequency(self, x):

        x = (x * self.std + self.mean).clamp(0, 1)

        gray = (
            0.299 * x[:, 0:1] +
            0.587 * x[:, 1:2] +
            0.114 * x[:, 2:3]
        )

        blocks = F.unfold(
            gray, kernel_size=8, stride=8
        )

        batch_size = x.shape[0]

        blocks = blocks.transpose(1, 2).reshape(
            batch_size, -1, 8, 8
        )

        dct = torch.einsum(
            "ij,bnjk,lk->bnil",
            self.dct_matrix,
            blocks,
            self.dct_matrix
        )

        frequency = dct.reshape(
            batch_size, -1, 64
        ).transpose(1, 2).reshape(
            batch_size, 64, 28, 28
        )

        frequency = (
            torch.sign(frequency) *
            torch.log1p(torch.abs(frequency))
        )

        return frequency

    def forward(self, x):

        # Spatial features
        spatial = self.spatial_features(x)
        spatial = self.spatial_pool(spatial)
        spatial = torch.flatten(spatial, 1)

        # Frequency features or zero vector
        if self.use_dct:

            frequency = self.extract_frequency(x)
            frequency = self.frequency_branch(frequency)
            frequency = torch.flatten(frequency, 1)

        else:

            frequency = torch.zeros(
                x.size(0), 128,
                device=x.device,
                dtype=spatial.dtype
            )

        # Feature fusion
        combined = torch.cat(
            [spatial, frequency], dim=1
        )

        return self.classifier(combined)
