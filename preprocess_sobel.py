import os, cv2, numpy as np
from tqdm import tqdm

DATA_DIRS = ["coco_subset/train2017_subset", "coco_subset/val2017"]  # két mappa
OUT_BASE = "edges_sobel"                    # ide mentjük
os.makedirs(OUT_BASE, exist_ok=True)

def sobel_magnitude(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    mag = (mag / mag.max() * 255).astype(np.uint8)
    return mag

for d in DATA_DIRS:
    out_dir = os.path.join(OUT_BASE, os.path.basename(d))
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(d) if f.lower().endswith((".png",".jpg",".jpeg"))]
    for fname in tqdm(files, desc=f"Processing {d}"):
        path = os.path.join(d, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mag = sobel_magnitude(img)
        cv2.imwrite(os.path.join(out_dir, os.path.splitext(fname)[0]+".png"), mag)