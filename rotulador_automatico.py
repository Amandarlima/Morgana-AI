import os
import shutil
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# =========================================
# CONFIGURAÇÕES
# =========================================
IMG_SIZE = 224
THRESHOLD = 0.7  # confiança mínima para classificar automaticamente

modelo_path = r"F:\Projetos\Morgana-AI\modelo_morgana_final.h5"
novas_imagens = r"F:\Projetos\\FOTOs morangos"
saida_organizada = r"F:\Projetos\Morgana-AI\classificacao_automatica"

# Carrega modelo treinado
model = tf.keras.models.load_model(modelo_path)

# Suas classes originais (mesmo nome das pastas do dataset)
class_names = [
     "saudavel",
      "anthracnose_fruit_rot",
      "powdery_mildew_leaf",
      "gray_mold",
      "blossom_blight",
      "leaf_spot",
      "angular_leafspot",
      "powdery_mildew_fruit"
]

# Cria diretórios de saída
for classe in class_names + ["duvidosas"]:
    os.makedirs(os.path.join(saida_organizada, classe), exist_ok=True)

# =========================================
# FUNÇÃO PARA CLASSIFICAR UMA IMAGEM
# =========================================
def classificar_imagem(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = model.predict(x, verbose=0)
    conf = np.max(preds)
    classe_pred = np.argmax(preds)
    label = class_names[classe_pred]

    return label, conf

# =========================================
# LOOP DE CLASSIFICAÇÃO
# =========================================
imagens = [
    os.path.join(novas_imagens, f)
    for f in os.listdir(novas_imagens)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

print(f"Encontradas {len(imagens)} novas imagens.")

for img_path in tqdm(imagens, desc="Classificando imagens"):
    label, conf = classificar_imagem(img_path)
    nome_arquivo = os.path.basename(img_path)

    if conf < THRESHOLD:
        destino = os.path.join(saida_organizada, "duvidosas", nome_arquivo)
    else:
        destino = os.path.join(saida_organizada, label, nome_arquivo)

    shutil.copy(img_path, destino)

print("\nClassificação automática concluída!")
print(f"Imagens organizadas em: {saida_organizada}")
print(f"As com confiança < {THRESHOLD*100:.0f}% estão em 'duvidosas'.")
