# =========================================
# 🌿 MORGANA AI - Treinamento completo local
# TensorFlow 2.14 | Ajustado com Data Augmentation avançado
# =========================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# =========================================
# CONFIGURAÇÕES DE CAMINHO
# =========================================
base_path = r"F:\Projetos\morganaAI\datasets_aumentado"
train_dir = os.path.join(base_path, "train")
val_dir   = os.path.join(base_path, "val")
test_dir  = os.path.join(base_path, "test")

print("✅ Dataset carregado em:", base_path)
print("TensorFlow versão:", tf.__version__)

# =========================================
# GERADORES DE IMAGEM (DATA AUGMENTATION AVANÇADO)
# =========================================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=35,            # mais variação de ângulo
    zoom_range=0.3,               # simula distâncias diferentes
    shear_range=0.25,             # leve distorção
    width_shift_range=0.15,       # deslocamento horizontal
    height_shift_range=0.15,      # deslocamento vertical
    brightness_range=[0.6, 1.5],  # simula luz/sombra
    channel_shift_range=30.0,     # variação de cor entre folhas
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=(150,150),
    class_mode='categorical', batch_size=32
)
val_data = val_gen.flow_from_directory(
    val_dir, target_size=(150,150),
    class_mode='categorical', batch_size=32
)
test_data = val_gen.flow_from_directory(
    test_dir, target_size=(150,150),
    class_mode='categorical', shuffle=False
)

class_names = list(train_data.class_indices.keys())
print("Classes detectadas:", class_names)

# =========================================
# CALLBACK VISUAL PARA PLOTS AO FINAL DE CADA ÉPOCA
# =========================================
class VisualizaPrevisoesCallback(Callback):
    def __init__(self, data, class_names):
        super().__init__()
        self.data = data
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        images, labels = next(self.data)
        preds = self.model.predict(images)
        fig, axs = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(f"Época {epoch + 1} - Previsões vs Real", fontsize=16)
        for i, ax in enumerate(axs.flat):
            ax.imshow(images[i])
            pred_idx = np.argmax(preds[i])
            true_idx = np.argmax(labels[i])
            cor = 'green' if pred_idx == true_idx else 'red'
            ax.set_title(f"Prev: {self.class_names[pred_idx]}\nReal: {self.class_names[true_idx]}", color=cor)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

# =========================================
# FASE 1 - CNN BASE
# =========================================
print("\n🧠 FASE 1 - Treinamento CNN Base")

cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
hist1 = cnn_model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[early, VisualizaPrevisoesCallback(val_data, class_names)]
)
print("✅ CNN Base concluída.")

# =========================================
# FASE 2 - TRANSFER LEARNING (MobileNetV2)
# =========================================
print("\n⚙️ FASE 2 - Transfer Learning com MobileNetV2")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150,150,3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(class_names), activation='softmax')(x)

model_v2 = Model(inputs=base_model.input, outputs=output)

model_v2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

hist2 = model_v2.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        VisualizaPrevisoesCallback(val_data, class_names)
    ]
)
print("✅ Transfer Learning concluído.")

# =========================================
# FASE 3 - FINE-TUNING
# =========================================
print("\n🚀 FASE 3 - Fine-Tuning (ajuste fino das últimas camadas)")

for layer in base_model.layers[-40:]:
    layer.trainable = True

model_v2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

hist3 = model_v2.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        VisualizaPrevisoesCallback(val_data, class_names)
    ]
)
print("✅ Fine-Tuning concluído.")

# =========================================
# AVALIAÇÃO FINAL E SALVAMENTO
# =========================================
print("\n📊 Avaliando modelo final...")

y_pred = np.argmax(model_v2.predict(test_data), axis=1)
y_true = test_data.classes
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão - Modelo Final")
plt.show()

# =========================================
# GRÁFICOS DE EVOLUÇÃO DAS FASES
# =========================================
def plot_history(hist, titulo):
    plt.figure(figsize=(6,4))
    plt.plot(hist.history['accuracy'], label='Treino')
    plt.plot(hist.history['val_accuracy'], label='Validação')
    plt.title(titulo)
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid()
    plt.show()

plot_history(hist1, "Fase 1 - CNN Base")
plot_history(hist2, "Fase 2 - Transfer Learning")
plot_history(hist3, "Fase 3 - Fine-Tuning")

# =========================================
# SALVANDO MODELOS
# =========================================
model_v2.save("modelo_morgana_final.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model_v2)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # otimiza para dispositivos embarcados
tflite_model = converter.convert()

with open("modelo_morgana_final.tflite", "wb") as f:
    f.write(tflite_model)

print("\n🎉 Treinamento completo!")
print("Modelos salvos em:")
print(" - modelo_morgana_final.h5")
print(" - modelo_morgana_final.tflite")
