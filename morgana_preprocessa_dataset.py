# ============================================
# 🌿 MORGANA AI - Pré-processamento com Background e Qualidade Visual
# ============================================

import os, cv2, random, csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from albumentations import (
    Compose, RandomBrightnessContrast, GaussianBlur, GaussNoise, CLAHE, RGBShift
)

# ============================================
# CONFIGURAÇÕES
# ============================================
IMG_SIZE = (260, 260)
THREADS = 6
PSNR_LIMIAR = 20.0
SSIM_LIMIAR = 0.6
BASE_PATH = Path(r"F:\Projetos\Morgana-AI")
OUTPUT_PATH = Path(r"F:\Projetos\Morgana-AI\datasets_balanceado_v2")
LOG_PATH = OUTPUT_PATH / "log_qc.csv"

for split in ["train", "val", "test"]:
    (OUTPUT_PATH / split).mkdir(parents=True, exist_ok=True)
    (BASE_PATH / split / "nao_reconhecido").mkdir(parents=True, exist_ok=True)

random.seed(42)
np.random.seed(42)

# ============================================
# FUNÇÕES AUXILIARES
# ============================================
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def ssim_val(img1, img2):
    return ssim(
        cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
        data_range=255
    )

def build_pipeline(leves=True):
    if leves:
        return Compose([
            RandomBrightnessContrast(p=0.5, brightness_limit=0.25),
            GaussianBlur(blur_limit=(3, 5), p=0.3),
            GaussNoise(var_limit=(5, 20), p=0.3),
            RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.3),
            CLAHE(clip_limit=2.0, p=0.4)
        ])
    else:
        return Compose([
            RandomBrightnessContrast(p=0.7, brightness_limit=0.4),
            GaussianBlur(blur_limit=(3, 7), p=0.5),
            GaussNoise(var_limit=(10, 40), p=0.5)
        ])

def contar_imgs(split_path):
    dist = {}
    for c in os.listdir(split_path):
        dir_c = split_path / c
        if dir_c.is_dir():
            dist[c] = len(list(dir_c.glob("*.jpg"))) + len(list(dir_c.glob("*.png")))
    return dist

# ============================================
# PROCESSAMENTO DE IMAGEM
# ============================================
def processa(img_path, classe, split, writer, lock, leves=True):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return
        img = cv2.resize(img, IMG_SIZE)
        pipeline = build_pipeline(leves)
        aug = pipeline(image=img)["image"]

        ps, ss = psnr(img, aug), ssim_val(img, aug)
        if ps < PSNR_LIMIAR or ss < SSIM_LIMIAR:
            return

        out_dir = OUTPUT_PATH / split / classe
        out_dir.mkdir(parents=True, exist_ok=True)
        nome_out = f"{img_path.stem}_aug_{random.randint(1000,9999)}.jpg"
        cv2.imwrite(str(out_dir / nome_out), aug)

        with lock:
            writer.writerow([datetime.now(), split, classe, img_path.name, nome_out, round(ps,2), round(ss,3)])
    except Exception as e:
        print(f"⚠️ {img_path} erro: {e}")

# ============================================
# EXECUÇÃO
# ============================================
lock = Lock()
with open(LOG_PATH, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["timestamp", "split", "classe", "img_original", "img_gerada", "psnr", "ssim"])

    for split in ["train", "val", "test"]:
        dist = contar_imgs(BASE_PATH / split)
        media = int(np.mean(list(dist.values())))
        print(f"\n📊 [{split}] média por classe: {media}")

        for classe, qtd in dist.items():
            imagens = list((BASE_PATH / split / classe).glob("*.jpg")) + list((BASE_PATH / split / classe).glob("*.png"))
            fator = 1 if qtd >= media else int(media / max(qtd, 1))
            print(f"🧠 Classe '{classe}' ({qtd} imgs) → x{fator}")

            leves = (classe == "nao_reconhecido")
            with ThreadPoolExecutor(max_workers=THREADS) as ex:
                for img_path in tqdm(imagens, desc=f"{split}/{classe}", ncols=100):
                    ex.submit(processa, img_path, classe, split, writer, lock, leves)

print("\n✅ Pré-processamento concluído! Dataset balanceado com background incluído.")
