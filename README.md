# GradNet - Edge Detection Neural Network

Egy CNN alapú éldetektáló neurális háló COCO adathalmazon Sobel operátorral generált címkékkel.

## 🚀 Gyors kezdés

### 1. Függőségek telepítése

**Windows:**
```bash
install_dependencies.bat
```

**Linux/Mac:**
```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

**Vagy manuálisan pip-pel:**
```bash
pip install -r requirements.txt
```

### 2. Adatok letöltése és előkészítése

```bash
# COCO képek letöltése (train + val)
python get_images_coco.py

# Sobel élek generálása
python preprocess_sobel.py
```

### 3. Modell tanítása

```bash
python train_edge_detector.py
```

### 4. Modell tesztelése

```bash
python test.py
```

## 📁 Projekt struktúra

```
GradNet/
├── requirements.txt              # Python függőségek
├── install_dependencies.bat      # Windows telepítő
├── install_dependencies.sh       # Linux/Mac telepítő
├── train_subset_list.txt         # Train képek listája
├── get_images_coco.py           # COCO képek letöltése
├── preprocess_sobel.py          # Sobel élek generálása
├── train_edge_detector.py       # Modell tanítása
├── test.py                      # Modell tesztelése
└── .gitignore                   # Git kizárások
```

## 🔧 Rendszerkövetelmények

- Python 3.7+
- CUDA kompatibilis GPU (opcionális, de ajánlott)
- ~10GB szabad hely az adatoknak

## 📊 Modell részletek

- **Architektúra:** VGG16 alapú encoder + egyedi decoder
- **Input:** RGB képek (256x256)
- **Output:** Grayscale élképek
- **Loss:** MSE + BCE kombinált loss
- **Optimalizáló:** Adam (lr=1e-4)

## 🎯 Használat

A tanított modell automatikusan GPU-t használ, ha elérhető, egyébként CPU-ra vált.

A modell checkpointok a `checkpoints/` mappába kerülnek, validációs előrejelzések a `checkpoints/val_preds/` mappába.
