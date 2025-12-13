import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2 
from torch.utils.data import Dataset
from torchvision import transforms

class FingerprintTextureDataset(Dataset):
    def __init__(self, pairs, core_finder_func, augment=False):
        self.pairs = pairs
        self.core_finder = core_finder_func 
        self.img_size = 64 
        self.augment = augment

        # UPDATED: Gentler augmentation to preserve thin ridges
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=10), # Reduced from 20 to 10
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)), # Reduced scale range
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.pairs)
        
    def preprocess_image(self, img_data, orientation_map):
        # SAFETY CHECK: Handle empty or broken skeletons
        if img_data is None or img_data.size == 0:
            return np.zeros((self.img_size, self.img_size), dtype=np.float32)

        core = self.core_finder(orientation_map, block_size=16) 
        if core is None:
            cy, cx = img_data.shape[0]//2, img_data.shape[1]//2
        else:
            cx, cy, _ = core
            
        half = self.img_size // 2
        padded = np.pad(img_data, ((half, half), (half, half)), mode='constant')
        cy += half
        cx += half
        
        # Crop around core
        patch = padded[cy-half:cy+half, cx-half:cx+half]
        
        # Thicken ridges (Dilation)
        kernel = np.ones((3,3), np.uint8)
        patch_uint8 = patch.astype(np.uint8)
        patch_thick = cv2.dilate(patch_uint8, kernel, iterations=1)

        patch = patch_thick.astype(np.float32) / 255.0
        return patch

    def __getitem__(self, idx):
        g1, g2, label = self.pairs[idx]
        
        # Preprocess
        p1 = self.preprocess_image(g1.skeleton, g1.orientation_map)
        p2 = self.preprocess_image(g2.skeleton, g2.orientation_map)
        
        img1 = torch.from_numpy(p1).unsqueeze(0)
        img2 = torch.from_numpy(p2).unsqueeze(0)
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.layer1 = self._make_layer(32, 64, stride=2) 
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # UPDATED: Feature Embedder with higher Dropout
        self.embedder = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5) # Increased dropout to 0.5
        )
        
        # UPDATED: Classifier Head (Binary Classification)
        # Learns to map the difference vector to a probability
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1) # Output Logit
        )

    def _make_layer(self, in_dim, out_dim, stride):
        return nn.Sequential(
            ResidualBlock(in_dim, out_dim, stride),
            ResidualBlock(out_dim, out_dim, stride=1)
        )

    def forward_one(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedder(x)
        return x

    def forward(self, img1, img2):
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)
        
        # Calculate absolute difference features
        diff = torch.abs(emb1 - emb2)
        
        # Classify the difference
        logits = self.classifier(diff)
        return logits.squeeze()

class EarlyStopping:
    def __init__(self, patience=7, delta=0.001, path='best_cnn.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            print(f'   EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)