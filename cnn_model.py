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

        # Gentler augmentation for sparse skeletons
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.pairs)
        
    def preprocess_image(self, img_data, orientation_map):
        # 1. Sanity Check: Is there data?
        if img_data is None or img_data.size == 0 or np.sum(img_data) == 0:
            return np.zeros((self.img_size, self.img_size), dtype=np.float32)

        # 2. INTELLIGENT CROPPING
        # Skeletons are sparse. We must find where the ink actually is.
        # We calculate the Center of Mass (centroid) of the white pixels.
        try:
            # Convert to uint8 for opencv
            img_u8 = img_data.astype(np.uint8)
            M = cv2.moments(img_u8)
            
            if M["m00"] != 0:
                # Centroid found
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # Fallback: finding the core via orientation map
                core = self.core_finder(orientation_map, block_size=16) 
                if core:
                    cx, cy, _ = core
                else:
                    cy, cx = img_data.shape[0]//2, img_data.shape[1]//2
        except:
            cy, cx = img_data.shape[0]//2, img_data.shape[1]//2
            
        # 3. Padding & Cropping around the Centroid/Core
        half = self.img_size // 2
        padded = np.pad(img_data, ((half, half), (half, half)), mode='constant')
        # Adjust coordinates for padding
        cy += half
        cx += half
        
        patch = padded[cy-half:cy+half, cx-half:cx+half]
        
        # 4. CRITICAL: Dilation
        # Skeletons are 1px thin. Convolutional strides might skip them.
        # We thicken them to 3px so the CNN sees them clearly.
        kernel = np.ones((3,3), np.uint8)
        patch_uint8 = patch.astype(np.uint8)
        patch_thick = cv2.dilate(patch_uint8, kernel, iterations=1)

        patch = patch_thick.astype(np.float32) / 255.0
        return patch

    def __getitem__(self, idx):
        g1, g2, label = self.pairs[idx]
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
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, 0.1)
        return out

class DeeperCNN(nn.Module):
    def __init__(self):
        super(DeeperCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Use MaxPool instead of strided conv for the first downsampling
        # MaxPool is better at preserving sparse features (white lines on black)
        self.layer1 = self._make_layer(32, 64, stride=2) 
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        
        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1)) # Changed to MaxPool
        
        # Embedder
        self.embedder = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            # No Dropout here (keep features deterministic)
        )
        
        # Classifier Head
        # Inputs: Emb1 (256) + Emb2 (256) + Diff (256) = 768
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def _make_layer(self, in_dim, out_dim, stride):
        return nn.Sequential(
            ResidualBlock(in_dim, out_dim, stride),
            ResidualBlock(out_dim, out_dim, stride=1)
        )

    def forward_one(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedder(x)
        # We normalize to prevent one vector exploding in magnitude
        return F.normalize(x, p=2, dim=1)

    def forward(self, img1, img2):
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)
        
        # Robust Fusion: Concatenate features
        # This tells the classifier: "Here is A, Here is B, Here is how they differ"
        diff = torch.abs(emb1 - emb2)
        combined = torch.cat((emb1, emb2, diff), dim=1)
        
        logits = self.classifier(combined)
        return logits.squeeze()

class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, path='best_cnn.pth'):
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