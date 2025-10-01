# train_edge_detector.py
import os, glob
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm

# -------------------------
# Adatmappák
# -------------------------
IMG_DIRS = {"train":"coco_subset/train2017_subset", "val":"coco_subset/val2017"}
EDGE_DIRS = {"train":"edges_sobel/train2017_subset", "val":"edges_sobel/val2017"}

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

VAL_PREDS_DIR = os.path.join(CHECKPOINT_DIR, "val_preds")
os.makedirs(VAL_PREDS_DIR, exist_ok=True)

# -------------------------
# Hyperparaméterek
# -------------------------
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (256,256)
LOSS_TYPE = "combined"  # "mse", "bce", "combined"

print("Használt eszköz:", DEVICE)

# -------------------------
# Dataset
# -------------------------
class EdgeDataset(Dataset):
    def __init__(self, img_dir, edge_dir, size=(256,256), binarize=False):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.edge_dir = edge_dir
        self.size = size
        self.binarize = binarize
        self.t_img = T.Compose([
            T.Resize(size), T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.t_edge = T.Compose([T.Resize(size), T.ToTensor()])

    def __len__(self): 
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]
        edge_path = os.path.join(self.edge_dir, base+".png")
        img = Image.open(img_path).convert("RGB")
        edge = Image.open(edge_path).convert("L")
        img_t, edge_t = self.t_img(img), self.t_edge(edge)
        if self.binarize:
            edge_t = (edge_t > 0.2).float()
        return img_t, edge_t

# -------------------------
# VGG4 rövidített háló
# -------------------------
class VGGEdgeNet(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        # csak az első 4 layer (Conv-ReLU-Conv-ReLU)
        self.features = nn.Sequential(*list(vgg.features.children())[:4])
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
        out_channels = 64  # az utolsó layer Conv2d(64,64)
        self.head = nn.Sequential(
            nn.Conv2d(out_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feat = self.features(x)
        out = self.head(feat)
        out = nn.functional.interpolate(out, (h, w), mode="bilinear", align_corners=False)
        return out

# -------------------------
# Loss választó
# -------------------------
def get_loss(loss_type):
    if loss_type=="mse": return "mse", nn.MSELoss()
    if loss_type=="bce": return "bce", nn.BCEWithLogitsLoss()
    if loss_type=="combined": return "combined", None
    raise ValueError

# -------------------------
# Train loop
# -------------------------
def train():
    binarize = LOSS_TYPE in ["bce","combined"]
    train_set = EdgeDataset(IMG_DIRS["train"], EDGE_DIRS["train"], size=IMG_SIZE, binarize=binarize)
    val_set   = EdgeDataset(IMG_DIRS["val"], EDGE_DIRS["val"], size=IMG_SIZE, binarize=binarize)
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_set, BATCH_SIZE)

    model = VGGEdgeNet().to(DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=LR)
    mode, criterion = get_loss(LOSS_TYPE)

    best_val = 1e9
    for epoch in range(1, EPOCHS+1):
        # --- train ---
        model.train(); tot=0
        for imgs, edges in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            imgs, edges = imgs.to(DEVICE), edges.to(DEVICE)
            logits = model(imgs)
            if mode=="mse":
                loss = criterion(torch.sigmoid(logits), edges)
            elif mode=="bce":
                loss = criterion(logits, edges)
            else: # combined
                loss_mse = nn.MSELoss()(torch.sigmoid(logits), edges)
                loss_bce = nn.BCEWithLogitsLoss()(logits, edges)
                loss = 0.5*loss_mse + 0.5*loss_bce
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot += loss.item() * imgs.size(0)
        tr_loss = tot / len(train_set)

        # --- val ---
        model.eval(); tot=0
        sample_count = 0
        with torch.no_grad():
            for imgs, edges in val_loader:
                imgs, edges = imgs.to(DEVICE), edges.to(DEVICE)
                logits = model(imgs)
                if mode=="mse":
                    loss = criterion(torch.sigmoid(logits), edges)
                elif mode=="bce":
                    loss = criterion(logits, edges)
                else:
                    loss_mse = nn.MSELoss()(torch.sigmoid(logits), edges)
                    loss_bce = nn.BCEWithLogitsLoss()(logits, edges)
                    loss = 0.5*loss_mse + 0.5*loss_bce
                tot += loss.item() * imgs.size(0)

                # előrejelzett képek mentése (max 5 kép epochonként)
                for i in range(imgs.size(0)):
                    if sample_count >= 5:
                        break
                    pred = torch.sigmoid(logits[i]).cpu().numpy()
                    pred_img = (pred[0]*255).astype('uint8')
                    fname = f"epoch{epoch}_val{i}.png"
                    Image.fromarray(pred_img).save(os.path.join(VAL_PREDS_DIR, fname))
                    sample_count += 1

        val_loss = tot / len(val_set)
        print(f"Epoch {epoch}: train={tr_loss:.4f} val={val_loss:.4f}")

        # modell mentése
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"best_{LOSS_TYPE}.pth"))
            print("Best model saved.")

if __name__=="__main__":
    train()