# =========================================
# 🌿 MORGANA AI - Pré-processamento de Imagens (Data Augmentation Offline)
# Usa OpenCV para gerar novas versões aprimoradas do dataset
# =========================================

import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random

# =========================================
# CAMINHOS DE ENTRADA E SAÍDA
# =========================================
base_path = Path(r"F:\Projetos\morganaAI\datasets")
output_path = base_path.parent / "datasets_aumentado"

# cria estrutura de saída
for split in ["train", "val", "test"]:
    (output_path / split).mkdir(parents=True, exist_ok=True)

print(f"📂 Dataset original: {base_path}")
print(f"💾 Dataset aumentado será salvo em: {output_path}")

# =========================================
# FUNÇÕES DE PROCESSAMENTO
# =========================================
def aplicar_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def aplicar_sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def aplicar_blur(img):
    return cv2.GaussianBlur(img, (3,3), 0)

def alterar_brilho(img):
    fator = random.uniform(0.7, 1.4)
    return cv2.convertScaleAbs(img, alpha=fator, beta=0)

def rotacionar(img):
    angulo = random.choice([-25, -15, -10, 10, 15, 25])
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angulo, 1)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def zoom(img):
    h, w = img.shape[:2]
    fator = random.uniform(0.85, 1.15)
    nh, nw = int(h * fator), int(w * fator)
    zoomed = cv2.resize(img, (nw, nh))
    # recorta ou preenche
    if fator > 1:
        x = (nw - w)//2; y = (nh - h)//2
        zoomed = zoomed[y:y+h, x:x+w]
    else:
        top = (h - nh)//2; bottom = h - nh - top
        left = (w - nw)//2; right = w - nw - left
        zoomed = cv2.copyMakeBorder(zoomed, top, bottom, left, right,
                                    cv2.BORDER_REFLECT)
    return zoomed

# =========================================
# LOOP DE PROCESSAMENTO
# =========================================
for split in ["train", "val", "test"]:
    input_dir = base_path / split
    output_dir = output_path / split

    for classe in os.listdir(input_dir):
        class_in = input_dir / classe
        class_out = output_dir / classe
        class_out.mkdir(exist_ok=True, parents=True)

        imagens = list(class_in.glob("*.jpg")) + list(class_in.glob("*.png"))

        print(f"\n🖼️ Processando classe '{classe}' ({len(imagens)} imagens)...")

        for img_path in tqdm(imagens, desc=f"{split}/{classe}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Redimensiona para tamanho padrão
            img = cv2.resize(img, (150,150))

            # Gera e salva múltiplas variações
            variantes = [
                aplicar_clahe(img),
                aplicar_sharpen(img),
                aplicar_blur(img),
                alterar_brilho(img),
                rotacionar(img),
                zoom(img)
            ]

            for i, var in enumerate(variantes):
                nome = f"{img_path.stem}_aug{i}.jpg"
                cv2.imwrite(str(class_out / nome), var)

print("\n✅ Processamento concluído!")
print("📁 Imagens aumentadas salvas em:", output_path)
