
import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision import models

# -------------------------
# Modell definíció
# -------------------------
class VGGEdgeNet(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(vgg.features.children())[:4])
        if freeze:
            for p in self.features.parameters():
                p.requires_grad = False
        out_channels = 64
        self.head = nn.Sequential(
            nn.Conv2d(out_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32,1,kernel_size=1)
        )

    def forward(self,x):
        h,w = x.shape[2:]
        feat = self.features(x)
        out = self.head(feat)
        out = nn.functional.interpolate(out, (h,w), mode="bilinear", align_corners=False)
        return out

# -------------------------
# Sobel gradiens generálás
# -------------------------
def compute_sobel(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return grad

# -------------------------
# Transforms
# -------------------------
IMG_SIZE = (256,256)
t_img = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# -------------------------
# Betöltés
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Használt eszköz:", DEVICE)

model = VGGEdgeNet().to(DEVICE)
checkpoint_path = "checkpoints/best_combined.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()
print("Modell betöltve:", checkpoint_path)

# -------------------------
# Kép betöltés
# -------------------------
img_path = input("Add meg a kép elérési útját: ")
image = cv2.imread(img_path)
if image is None:
    raise ValueError("Nem találom a képet!")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Sobel előállítása
sobel_img = compute_sobel(image)

# Modell előrejelzés
img_pil = Image.fromarray(image)
inp_tensor = t_img(img_pil).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    pred = torch.sigmoid(model(inp_tensor)).cpu().numpy()[0,0]
pred_img = (pred*255).astype('uint8')

# -------------------------
# Megjelenítés
# -------------------------
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title("Eredeti kép")
plt.imshow(image)
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Sobel gradiens")
plt.imshow(sobel_img, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Modell predikció")
plt.imshow(pred_img, cmap='gray')
plt.axis('off')

plt.show()
