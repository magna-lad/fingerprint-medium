import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class FingerprintTextureDataset(Dataset):
    """
    PyTorch Dataset that loads SKELETON images.
    Uses 'skeleton' instead of 'orientation_map' for robust visual matching.
    """
    def __init__(self, pairs, core_finder_func, augment=False):
        self.pairs = pairs
        self.core_finder = core_finder_func 
        self.img_size = 64 
        self.augment = augment

        # --- DATA AUGMENTATION ---
        # Rotation works perfectly on skeletons!
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomRotation(degrees=15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                # No ColorJitter needed for binary images
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.pairs)
        
    def preprocess_image(self, img_data, orientation_map):
        # 1. We still need orientation_map just to find the core coordinates
        core = self.core_finder(orientation_map, block_size=16) # Original block size logic
        
        if core is None:
            cy, cx = img_data.shape[0]//2, img_data.shape[1]//2
        else:
            # Unpack (x, y, orientation) -> we only need x,y
            cx, cy, _ = core
            
        # 2. Pad and Crop the SKELETON
        half = self.img_size // 2
        padded = np.pad(img_data, ((half, half), (half, half)), mode='constant')
        cy += half
        cx += half
        
        patch = padded[cy-half:cy+half, cx-half:cx+half]
        
        # 3. Normalize Binary Image (0-255 -> 0-1)
        patch = patch.astype(np.float32) / 255.0
        return patch

    def __getitem__(self, idx):
        g1, g2, label = self.pairs[idx]
        
        # USE SKELETON (Visual Ridge Pattern)
        img1_raw = g1.skeleton
        img2_raw = g2.skeleton
        
        # Extract patches (using orientation map only for centering)
        np_img1 = self.preprocess_image(img1_raw, g1.orientation_map)
        np_img2 = self.preprocess_image(img2_raw, g2.orientation_map)
        
        # Convert to Tensor (1, 64, 64)
        img1 = torch.from_numpy(np_img1).unsqueeze(0)
        img2 = torch.from_numpy(np_img2).unsqueeze(0)
        
        # Apply Augmentation
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)

class TinyCNN(nn.Module):
    """
    Standard Siamese CNN for ridge pattern matching.
    """
    def __init__(self):
        super(TinyCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc = nn.Linear(128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.4)
        self.out = nn.Linear(256, 128) 

    def forward_one(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        x = self.out(x)
        return x

    def forward(self, img1, img2):
        emb1 = self.forward_one(img1)
        emb2 = self.forward_one(img2)
        dist = F.pairwise_distance(emb1, emb2)
        return dist

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, euclidean_distance, label):
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                                      (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive