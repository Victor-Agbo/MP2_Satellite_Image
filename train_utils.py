# train_utils.py
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, accuracy_score,precision_recall_curve
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as T
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ============================================================
# 1. Dataset definition with safe folder and band checks
# ============================================================

class BigEarthNetCustomDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            df: Pandas DataFrame with 'patch_id' and 'encoded' columns
            root_dir: path to BigEarthNet-S2 root
            transform: optional transforms for torch tensors
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.band_order = [
            "B02", "B03", "B04", "B05", "B06",
            "B07", "B08", "B8A", "B11", "B12"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_dir = os.path.join(self.root_dir, row["patch_id"][:-6], row["patch_id"])

        try:
            # --- check patch folder ---
            if not os.path.isdir(patch_dir):
                raise FileNotFoundError(f"Missing folder: {patch_dir}")

            # --- reference band (B02) ---
            ref_band_path = os.path.join(patch_dir, f"{row['patch_id']}_B02.tif")
            if not os.path.exists(ref_band_path):
                raise FileNotFoundError(f"Missing reference band: {ref_band_path}")

            with rasterio.open(ref_band_path) as ref:
                ref_height, ref_width = ref.shape

            # --- read and normalize all bands ---
            bands = []
            for band in self.band_order:
                path = os.path.join(patch_dir, f"{row['patch_id']}_{band}.tif")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing band: {path}")

                with rasterio.open(path) as src:
                    arr = src.read(
                        out_shape=(1, ref_height, ref_width),
                        resampling=Resampling.bilinear
                    )[0].astype(np.float32)
                    arr = np.nan_to_num(arr)
                    arr /= np.max(arr) if np.max(arr) > 0 else 1.0
                    bands.append(arr)

            img = np.stack(bands, axis=0)  # shape: (10, H, W)
            img = torch.tensor(img, dtype=torch.float32)

            img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
            label = torch.tensor(row["encoded"], dtype=torch.float32)

            if self.transform:
                img = self.transform(img)

            return img, label

        except Exception as e:
            print(f"⚠️ Skipping {row['patch_id']}: {e}")
            return None

# ============================================================
# 2. Collate function (skips None safely)
# ============================================================

def safe_collate_fn(batch):
    """Skips None samples (missing or broken patches)."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

# --- Model builder ---
def build_model(num_classes=19, in_channels=10, pretrained=True, device="cuda"):
    model = timm.create_model(
        "swin_small_patch4_window7_224",
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=num_classes
    )
    return model.to(device)

# --- Optimizer ---
def build_optimizer(model, lr=1e-4, weight_decay=1e-4):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

# --- Scheduler ---
def build_scheduler(optimizer, num_epochs=1):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# --- Criterion ---
def build_criterion(pos_weights, device="cuda"):
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)

def get_weights(model):
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def set_weights(model, weights):
    sd = model.state_dict()
    for (k, _), w in zip(sd.items(), weights):
        sd[k] = torch.tensor(w)
    model.load_state_dict(sd)
    
# ============================================================
# 6. Training and evaluation functions
# ============================================================
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        if batch is None:
            continue
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, criterion, device, thresholds=0.2):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if batch is None:
                continue
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Stack all predictions and ground truth
    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    
    y_pred_bin = (y_pred >= thresholds).astype(int)

    # --- Classic multi-label metrics ---
    f1_micro = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    precision_micro = precision_score(y_true, y_pred_bin, average="micro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred_bin, average="micro", zero_division=0)

    # --- Average Precision metrics ---
    try:
        ap_micro = average_precision_score(y_true, y_pred, average="micro")
        ap_macro = average_precision_score(y_true, y_pred, average="macro")
    except ValueError:
        ap_micro, ap_macro = np.nan, np.nan

    return (
        total_loss / len(dataloader.dataset),  # average loss
        f1_micro, f1_macro,
        precision_micro, recall_micro,
        ap_micro, ap_macro,
        y_true, y_pred,
    )
    pass

# --- Dataset partitioning for federated clients ---
def get_client_dataloaders(client_id, train_df, val_df, root_dir, batch_size=4):

    # Apply transforms
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(10),

    ])

    train_dataset = BigEarthNetCustomDataset(train_df, root_dir, transform=train_transform)
    val_dataset   = BigEarthNetCustomDataset(val_df, root_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=safe_collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=safe_collate_fn)

    return train_loader, val_loader
