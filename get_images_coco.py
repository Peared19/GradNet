import os
import random
import requests
from tqdm import tqdm
from zipfile import ZipFile

# --- Beállítások ---
SAVE_DIR = "coco_subset"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# 1. val2017 letöltés (5000 kép)
# -------------------------
VAL_URL = "http://images.cocodataset.org/zips/val2017.zip"
VAL_FILE = "val2017.zip"

if not os.path.exists(VAL_FILE):
    print("val2017.zip letöltése...")
    r = requests.get(VAL_URL, stream=True)
    with open(VAL_FILE, "wb") as f:
        for chunk in tqdm(r.iter_content(chunk_size=8192), desc="val2017.zip"):
            if chunk:
                f.write(chunk)
else:
    print("val2017.zip már megvan.")

# Kicsomagolás coco_subset/val2017 alá
VAL_DIR = os.path.join(SAVE_DIR, "val2017")
if not os.path.exists(VAL_DIR):
    print("val2017.zip kicsomagolása...")
    with ZipFile(VAL_FILE, 'r') as zip_ref:
        zip_ref.extractall(SAVE_DIR)
else:
    print("val2017 képek már kicsomagolva.")

# -------------------------
# 2. Train2017-ből 5000 random kép letöltése
# -------------------------
TRAIN_BASE = "http://images.cocodataset.org/train2017/"
TRAIN_DIR = os.path.join(SAVE_DIR, "train2017_subset")
os.makedirs(TRAIN_DIR, exist_ok=True)

subset_file = "train_subset_list.txt"  # főmappába mentjük

# Ha már létezik a lista, akkor abból töltünk újra
if os.path.exists(subset_file):
    print("Korábbi subset listát használok.")
    with open(subset_file, "r") as f:
        selected_files = [line.strip() for line in f.readlines()]
else:
    print("Új subset generálása train2017-ből (5000 kép)...")
    max_id = 581929  # legnagyobb COCO train ID
    selected_ids = random.sample(range(1, max_id), 5000)
    selected_files = [f"{img_id:012d}.jpg" for img_id in selected_ids]

    # fájlnevek mentése txt-be
    with open(subset_file, "w") as f:
        for name in selected_files:
            f.write(name + "\n")
    print(f"{subset_file} mentve ({len(selected_files)} fájl).")

# Képek letöltése
for file_name in tqdm(selected_files, desc="Train képek letöltése"):
    url = TRAIN_BASE + file_name
    save_path = os.path.join(TRAIN_DIR, file_name)

    if not os.path.exists(save_path):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)
        except Exception as e:
            print(f"Hiba {file_name}: {e}")
