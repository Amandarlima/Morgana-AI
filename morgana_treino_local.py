# =========================================
# 🌿 MORGANA AI - Treinamento com EfficientNetV2S e Métricas de Background
# =========================================

import os, random, psutil
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# =========================================
# CONFIG
# =========================================
IMG_SIZE = (260, 260)
BATCH_SIZE = 32
EPOCHS_FASE1 = 25
EPOCHS_FASE2 = 15
BASE_LR = 1e-4
FINE_TUNE_LR = 1e-5
DATA_DIR = Path(r"F:\Projetos\Morgana-AI\datasets_balanceado_v2")
MODEL_NAME = "morganaAI_EfficientNetV2S_bg"

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

print(f"💻 CPU: {psutil.cpu_count(logical=True)} núcleos")
print("🧠 GPU detectada:", tf.config.list_physical_devices('GPU'))

# =========================================
# DATA AUGMENTATION
# =========================================
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.25,
    shear_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.7, 1.4],
    horizontal_flip=True
)

val_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(DATA_DIR / "train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_data = val_gen.flow_from_directory(DATA_DIR / "val", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
test_data = val_gen.flow_from_directory(DATA_DIR / "test", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

print(f"📁 Dataset: {len(train_data.class_indices)} classes → {train_data.class_indices}")

# =========================================
# MODELO
# =========================================
base_model = EfficientNetV2S(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
output = Dense(len(train_data.class_indices), activation='softmax')(x)
model = Model(base_model.input, output)

model.compile(
    optimizer=Adam(learning_rate=BASE_LR),
    loss=CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy', Precision(), Recall()]
)

# =========================================
# CALLBACKS
# =========================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.4, patience=3, min_lr=1e-6),
    ModelCheckpoint(f"{MODEL_NAME}_best.h5", save_best_only=True)
]

# =========================================
# TREINAMENTO
# =========================================
print("\n🚀 Fase 1: Treinamento base congelado")
model.fit(train_data, validation_data=val_data, epochs=EPOCHS_FASE1, callbacks=callbacks, verbose=1)

print("\n🔧 Fase 2: Fine-tuning (últimas camadas destravadas)")
for layer in base_model.layers[-80:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss=CategoricalCrossentropy(label_smoothing=0.05),
    metrics=['accuracy', Precision(), Recall()]
)

model.fit(train_data, validation_data=val_data, epochs=EPOCHS_FASE2, callbacks=callbacks, verbose=1)

# =========================================
# AVALIAÇÃO
# =========================================
print("\n📊 Avaliando desempenho final...")
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)
y_true = test_data.classes
labels = list(test_data.class_indices.keys())

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.title("Matriz de Confusão - Morgana AI EfficientNetV2S")
plt.xlabel("Previsto"); plt.ylabel("Real")
plt.tight_layout()
plt.savefig(f"{MODEL_NAME}_matriz_confusao.png", dpi=150)
plt.show()

# Relatório geral
report = classification_report(y_true, y_pred, target_names=labels, digits=3)
print(report)

# =========================================
# MÉTRICAS ESPECÍFICAS DE BACKGROUND
# =========================================
if "nao_reconhecido" in labels:
    idx_bg = labels.index("nao_reconhecido")
    y_true_bg = (y_true == idx_bg).astype(int)
    y_pred_bg = (y_pred == idx_bg).astype(int)

    f1_bg = f1_score(y_true_bg, y_pred_bg)
    rec_bg = recall_score(y_true_bg, y_pred_bg)
    prec_bg = precision_score(y_true_bg, y_pred_bg)

    print(f"\n🎯 Métricas específicas da classe 'nao_reconhecido':")
    print(f"   ➤ Precision: {prec_bg:.3f}")
    print(f"   ➤ Recall:    {rec_bg:.3f}")
    print(f"   ➤ F1-score:  {f1_bg:.3f}")
else:
    print("\n⚠️ Classe 'nao_reconhecido' não encontrada no conjunto de teste!")

# =========================================
# GRÁFICOS DE MÉTRICAS POR CLASSE
# =========================================
precisions, recalls, f1s, _ = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(len(labels)))
df_metrics = pd.DataFrame({
    "Classe": labels,
    "Precision": precisions,
    "Recall": recalls,
    "F1-Score": f1s
}).sort_values("F1-Score", ascending=False)

df_metrics.to_csv(f"{MODEL_NAME}_metricas_por_classe.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(10, 5))
sns.barplot(data=df_metrics, x="Classe", y="F1-Score", palette="crest")
plt.xticks(rotation=45, ha="right")
plt.title("📈 F1-Score por Classe (incluindo 'nao_reconhecido')")
plt.ylabel("F1-Score"); plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{MODEL_NAME}_f1_por_classe.png", dpi=150)
plt.show()

plt.figure(figsize=(10, 5))
df_melted = df_metrics.melt(id_vars="Classe", value_vars=["Precision", "Recall"], var_name="Métrica", value_name="Valor")
sns.barplot(data=df_melted, x="Classe", y="Valor", hue="Métrica", palette="viridis")
plt.xticks(rotation=45, ha="right")
plt.title("📊 Precision vs Recall por Classe")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{MODEL_NAME}_precision_recall_por_classe.png", dpi=150)
plt.show()

print(f"✅ Gráficos salvos:")
print(f"   - {MODEL_NAME}_f1_por_classe.png")
print(f"   - {MODEL_NAME}_precision_recall_por_classe.png")
print(f"   - {MODEL_NAME}_metricas_por_classe.csv")

# =========================================
# EXPORTAÇÃO TFLITE
# =========================================
model.save(f"{MODEL_NAME}_final.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(f"{MODEL_NAME}_final_INT8.tflite", "wb") as f:
    f.write(tflite_model)

print("\n✅ Treinamento concluído e modelo exportado com sucesso!")
