# treino_avancado_corrigido.py

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, 
    LearningRateScheduler, TerminateOnNaN
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Tentar importar tensorflow_addons, mas ter fallback se não disponível
try:
    import tensorflow_addons as tfa
    TFA_AVAILABLE = True
    print("✅ TensorFlow Addons disponível")
except ImportError:
    TFA_AVAILABLE = False
    print("⚠️ TensorFlow Addons não disponível - usando otimizadores padrão")

class AdvancedMorganaTrainer:
    """
    Treinamento avançado com técnicas de regularização e domain adaptation
    Versão corrigida e funcional
    """
    
    def __init__(self, data_dir, img_size=(320, 320), batch_size=16):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = None
        self.num_classes = None
        
    def create_advanced_augmentation(self):
        """
        Data augmentation em tempo real durante treinamento
        """
        return tf.keras.Sequential([
            layers.RandomRotation(0.5),  # Reduzido para 10% para maior estabilidade
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.2),
            layers.RandomTranslation(0.05, 0.05),
        ])
    
    def build_regularized_model(self, num_classes, base_weights='imagenet'):
        """
        Modelo com regularização avançada
        """
        self.num_classes = num_classes
        
        # Base model
        base_model = EfficientNetV2S(
            weights=base_weights,
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Congelamento inicial
        base_model.trainable = False
        
        # Camadas customizadas
        inputs = tf.keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        
        # Feature extraction com regularização
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        # Camada densa com regularização
        x = layers.Dense(
            512,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-4),
            bias_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Segunda camada
        x = layers.Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l2(1e-4)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Output layer
        outputs = layers.Dense(
            num_classes, 
            activation='softmax'
        )(x)
        
        model = Model(inputs, outputs)
        return model, base_model
    
    def get_optimizer(self, lr):
        """
        Retorna otimizador com fallback se tensorflow_addons não estiver disponível
        """
        if TFA_AVAILABLE:
            return tfa.optimizers.AdamW(
                learning_rate=lr,
                weight_decay=1e-4
            )
        else:
            # Usar Adam padrão com decay
            return tf.keras.optimizers.Adam(
                learning_rate=lr
            )
    
    def get_metrics(self):
        """
        Retorna métricas com fallback para F1 Score
        """
        metrics = [
            'accuracy',
            Precision(name='precision'),
            Recall(name='recall')
        ]
        
        # Adicionar F1 score apenas se disponível
        if TFA_AVAILABLE and self.num_classes:
            metrics.append(
                tfa.metrics.F1Score(num_classes=self.num_classes, name='f1_score', average='weighted')
            )
        
        return metrics
    
    def progressive_unfreezing(self, model, base_model, stage):
        """
        Descongelamento progressivo
        """
        if stage == 1:
            # Fase 1: Apenas camadas superiores
            for layer in base_model.layers[-20:]:  # Reduzido para maior estabilidade
                layer.trainable = True
            lr = 1e-4
            
        elif stage == 2:
            # Fase 2: Mais camadas
            for layer in base_model.layers[-40:]:
                layer.trainable = True
            lr = 5e-5
            
        elif stage == 3:
            # Fase 3: Quase todas as camadas
            for layer in base_model.layers[-80:]:
                layer.trainable = True
            lr = 1e-5
            
        else:
            # Fase 4: Todas as camadas
            base_model.trainable = True
            lr = 5e-6
        
        # Compilar com learning rate específico
        model.compile(
            optimizer=self.get_optimizer(lr),
            loss=CategoricalCrossentropy(label_smoothing=0.1),
            metrics=self.get_metrics()
        )
        
        return lr
    
    def create_advanced_callbacks(self, model_name):
        """
        Callbacks avançados para treinamento estável
        """
        callbacks = [
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,  # Ajustado
                restore_best_weights=True,
                mode='min',
                verbose=1
            ),
            
            # Redução de LR adaptativa
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Checkpoint
            ModelCheckpoint(
                f"{model_name}_best.h5",
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            
            # Prevenção de NaN
            TerminateOnNaN(),
        ]
        
        return callbacks
    
    def load_data(self):
        """
        Carregar dados com validação
        """
        # Data generators
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,
            width_shift_range=0.15,
            height_shift_range=0.15,
            brightness_range=[0.8, 1.3],
            zoom_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode='reflect'
        )
        
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        
        train_data = train_datagen.flow_from_directory(
            self.data_dir / "train",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_data = val_datagen.flow_from_directory(
            self.data_dir / "val", 
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        test_data = val_datagen.flow_from_directory(
            self.data_dir / "test",
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        class_names = list(train_data.class_indices.keys())
        print(f"📊 Classes detectadas: {class_names}")
        print(f"📈 Distribuição: {dict(zip(class_names, np.bincount(train_data.classes)))}")
        
        return train_data, val_data, test_data, class_names
    
    def train_with_progressive_unfreezing(self, model_name="morgana_advanced"):
        """
        Treinamento com descongelamento progressivo - VERSÃO SIMPLIFICADA E ESTÁVEL
        """
        # Carregar dados
        train_data, val_data, test_data, class_names = self.load_data()
        num_classes = len(class_names)
        self.class_names = class_names
        
        print(f"🎯 Iniciando treinamento para {num_classes} classes")
        
        # Construir modelo
        model, base_model = self.build_regularized_model(num_classes)
        
        # FASE 1: Treinamento inicial com base congelada
        print("\n🔒 FASE 1: Base congelada - Feature Extraction")
        model.compile(
            optimizer=self.get_optimizer(1e-4),
            loss=CategoricalCrossentropy(label_smoothing=0.1),
            metrics=self.get_metrics()
        )
        
        history1 = model.fit(
            train_data,
            epochs=6,
            validation_data=val_data,
            callbacks=self.create_advanced_callbacks(f"{model_name}_stage1"),
            verbose=1
        )
        
        # FASE 2: Fine-tuning moderado
        print("\n🔓 FASE 2: Fine-tuning moderado")
        lr = self.progressive_unfreezing(model, base_model, stage=2)
        print(f"📐 Learning Rate: {lr}")
        
        history2 = model.fit(
            train_data,
            epochs=8,
            validation_data=val_data,
            callbacks=self.create_advanced_callbacks(f"{model_name}_stage2"),
            verbose=1
        )
        
        # Combinar históricos
        full_history = self.combine_histories([history1, history2])
        
        return model, full_history, test_data
    
    def combine_histories(self, histories):
        """
        Combinar históricos de múltiplas fases
        """
        combined = {}
        for key in histories[0].history.keys():
            combined[key] = []
            for history in histories:
                combined[key].extend(history.history[key])
        return combined
    
    def evaluate_model(self, model, test_data, output_dir="results"):
        """
        Avaliação completa do modelo
        """
        print("\n📊 Avaliando modelo final...")
        
        # Criar diretório de resultados
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Avaliação básica
        test_results = model.evaluate(test_data, verbose=0)
        print("✅ Métricas no conjunto de teste:")
        for metric, value in zip(model.metrics_names, test_results):
            print(f"   {metric}: {value:.4f}")
        
        # Predições
        y_pred = model.predict(test_data, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_data.classes
        
        # Matriz de confusão
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Matriz de Confusão - Morgana AI Avançado')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.tight_layout()
        plt.savefig(output_path / 'matriz_confusao_avancada.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Relatório de classificação
        report = classification_report(y_true, y_pred_classes, 
                                     target_names=self.class_names, digits=4)
        print("\n📈 Relatório de Classificação:")
        print(report)
        
        # Salvar relatório
        with open(output_path / 'relatorio_classificacao.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Gráfico de métricas por classe
        from sklearn.metrics import precision_recall_fscore_support
        precisions, recalls, f1_scores, _ = precision_recall_fscore_support(
            y_true, y_pred_classes, average=None
        )
        
        metrics_df = pd.DataFrame({
            'Classe': self.class_names,
            'Precision': precisions,
            'Recall': recalls,
            'F1-Score': f1_scores
        }).sort_values('F1-Score', ascending=False)
        
        plt.figure(figsize=(12, 6))
        X_axis = np.arange(len(metrics_df))
        
        plt.bar(X_axis - 0.2, metrics_df['Precision'], 0.2, label='Precision', alpha=0.8)
        plt.bar(X_axis, metrics_df['Recall'], 0.2, label='Recall', alpha=0.8)
        plt.bar(X_axis + 0.2, metrics_df['F1-Score'], 0.2, label='F1-Score', alpha=0.8)
        
        plt.xticks(X_axis, metrics_df['Classe'], rotation=45, ha='right')
        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Métricas por Classe - Morgana AI Avançado')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'metricas_por_classe.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Salvar métricas em CSV
        metrics_df.to_csv(output_path / 'metricas_detalhadas.csv', index=False, encoding='utf-8-sig')
        
        return test_results, metrics_df

# SCRIPT DE EXECUÇÃO SIMPLIFICADO
def main():
    """
    Execução principal corrigida
    """
    print("🚀 MORGANA AI - Treinamento Avançado Corrigido")
    
    # Configurações - AJUSTE ESTE CAMINHO
    DATA_DIR = Path(r"C:\Users\Gustavo\Desktop\TCC\strawberry-ai\ia\MorganaAI\datasets_balanceado_v3")
    OUTPUT_DIR = "resultados_avancados"
    
    if not DATA_DIR.exists():
        print(f"❌ Diretório de dados não encontrado: {DATA_DIR}")
        print("⚠️ Por favor, ajuste o caminho DATA_DIR no código")
        return
    
    # Verificar estrutura do dataset
    required_folders = ["train", "val", "test"]
    for folder in required_folders:
        folder_path = DATA_DIR / folder
        if not folder_path.exists():
            print(f"❌ Pasta '{folder}' não encontrada em {DATA_DIR}")
            return
        print(f"✅ {folder}: {len(list(folder_path.glob('*')))} classes")
    
    try:
        # Inicializar trainer
        trainer = AdvancedMorganaTrainer(
            data_dir=DATA_DIR,
            img_size=(320, 320),
            batch_size=16
        )
        
        # Treinar modelo
        model, history, test_data = trainer.train_with_progressive_unfreezing()
        
        # Avaliar modelo
        test_results, metrics_df = trainer.evaluate_model(model, test_data, OUTPUT_DIR)
        
        # Salvar modelo
        model.save(Path(OUTPUT_DIR) / "morgana_advanced_final.h5")
        
        # Converter para TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(Path(OUTPUT_DIR) / "morgana_advanced.tflite", "wb") as f:
            f.write(tflite_model)
        
        print(f"\n🎊 Treinamento concluído com sucesso!")
        print(f"📁 Resultados salvos em: {OUTPUT_DIR}")
        print(f"📊 Melhor F1-Score: {metrics_df['F1-Score'].max():.4f}")
        print(f"📈 Acurácia média: {metrics_df['F1-Score'].mean():.4f}")
        
    except Exception as e:
        print(f"❌ Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()