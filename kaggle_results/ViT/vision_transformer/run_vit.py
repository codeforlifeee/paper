import os
import sys
import time
import math
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sys as _sys2; _sys2.path.insert(0,'/kaggle/working')
from common_data import load_indian_pines

class ViT(nn.Module):
    def __init__(self, in_channels=200, num_classes=16, patch_size=11, embed_dim=64, depth=4, num_heads=4, mlp_dim=128):
        super().__init__()
        self.num_patches = patch_size * patch_size
        self.patch_to_embedding = nn.Linear(in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=0.1, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, num_classes))
    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(b, h*w, c)
        tokens = self.patch_to_embedding(tokens)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat((cls_tokens, tokens), dim=1) + self.pos_embedding
        out = self.transformer(tokens)
        return self.mlp_head(out[:, 0])

def pad_with_zeros(X, margin=5):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX

def create_patches(X, y, windowSize=11):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = pad_with_zeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex += 1
    patchesData = patchesData[patchesLabels > 0, :, :, :]
    patchesLabels = patchesLabels[patchesLabels > 0] - 1
    return patchesData, patchesLabels

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running ViT on {device}")
    
    img, gt = load_indian_pines()
    # Normalize
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    patches, labels = create_patches(img, gt, windowSize=11)
    # Permute to (B, C, H, W)
    patches = patches.transpose(0, 3, 1, 2)
    
    # 10% Train, 10% Val, 80% Test
    X_train, X_test, y_train, y_test = train_test_split(patches, labels, test_size=0.90, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.8888, random_state=42, stratify=y_test)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)), batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)), batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)), batch_size=32, shuffle=False)
    
    model = ViT(in_channels=200, num_classes=16, patch_size=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 200
    train_losses = []
    val_losses = []
    best_val_acc = 0.0
    
    start_time = time.time()
    for ep in range(epochs):
        model.train()
        t_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * bx.size(0)
        t_loss /= len(train_loader.dataset)
        train_losses.append(t_loss)
        
        model.eval()
        v_loss = 0.0
        v_corr = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                out = model(bx)
                loss = criterion(out, by)
                v_loss += loss.item() * bx.size(0)
                v_corr += (out.argmax(1) == by).sum().item()
        v_loss /= len(val_loader.dataset)
        v_acc = v_corr / len(val_loader.dataset)
        val_losses.append(v_loss)
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), "best_vit.pth")
            
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch [{ep+1}/{epochs}] - Train Loss: {t_loss:.4f} - Val Loss: {v_loss:.4f} - Val Acc: {v_acc*100:.2f}%")
            
    train_time = (time.time() - start_time) / 60.0
    
    # Test evaluation
    model.load_state_dict(torch.load("best_vit.pth"))
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            out = model(bx)
            preds.extend(out.argmax(1).cpu().numpy().tolist())
            targets.extend(by.numpy().tolist())
            
    oa = accuracy_score(targets, preds) * 100
    cm = confusion_matrix(targets, preds)
    aa = np.mean(np.nan_to_num(np.diag(cm) / np.sum(cm, axis=1))) * 100
    kappa = cohen_kappa_score(targets, preds) * 100
    
    print(f"ViT Final Results: OA = {oa:.2f}%, AA = {aa:.2f}%, Kappa = {kappa:.2f}%, Train Time = {train_time:.2f} min")
    
    os.makedirs("./outputs", exist_ok=True)
    # Save standalone loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="#4f46e5", lw=2)
    plt.plot(val_losses, label="Val Loss", color="#ef4444", lw=2)
    plt.title(f"Vision Transformer (ViT) Loss Curve - Indian Pines")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig("./outputs/loss_vision_transformer.png", bbox_inches="tight", dpi=300)
    plt.close()
    
    # Predict full map
    full_pred = np.zeros_like(gt)
    margin = 5
    padded_full = pad_with_zeros(img, margin)
    coords = [(r, c) for r in range(gt.shape[0]) for c in range(gt.shape[1]) if gt[r, c] > 0]
    
    all_patches = []
    for r, c in coords:
        p = padded_full[r:r + 11, c:c + 11, :].transpose(2, 0, 1)
        all_patches.append(p)
    all_patches = np.array(all_patches)
    
    pred_labels = []
    batch_sz = 64
    with torch.no_grad():
        for i in range(0, len(all_patches), batch_sz):
            batch = torch.FloatTensor(all_patches[i:i + batch_sz]).to(device)
            p_out = model(batch)
            pred_labels.extend(p_out.argmax(1).cpu().numpy().tolist())
            
    for idx, (r, c) in enumerate(coords):
        full_pred[r, c] = pred_labels[idx] + 1
        
    plt.figure(figsize=(8, 8))
    plt.imshow(full_pred, cmap="nipy_spectral")
    plt.title(f"ViT Prediction Map (OA: {oa:.2f}%)")
    plt.axis("off")
    plt.savefig("./outputs/pred_vision_transformer.png", bbox_inches="tight", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
