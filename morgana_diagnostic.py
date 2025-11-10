# ============================================
# 🧪 CLASSE DE DIAGNÓSTICO CIENTÍFICO - MORGANA AI
# ============================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import cv2
import os
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MorganaDiagnostic:
    """
    Classe para diagnóstico científico completo do modelo Morgana AI
    Baseado em: 
    - "Model Monitoring and Diagnostics in Production" (Sculley et al., 2015)
    - "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)
    """
    
    def __init__(self, base_path="diagnostic_results"):
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Configurações de plot
        plt.style.use('seaborn-v0_8')
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
        
        print(f"🔬 Iniciando Sistema de Diagnóstico Morgana AI - {self.timestamp}")
    
    def analyze_data_distribution(self, data_dir):
        """
        Analisa a distribuição dos dados entre splits e classes
        Base teórica: "Learning from Imbalanced Data" (IEEE TKDE, 2009)
        """
        print("\n📊 ANALISANDO DISTRIBUIÇÃO DOS DADOS...")
        
        splits = ['train', 'val', 'test']
        distribution_data = {}
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, split in enumerate(splits):
            split_path = Path(data_dir) / split
            if not split_path.exists():
                continue
                
            class_counts = {}
            for class_name in os.listdir(split_path):
                class_path = split_path / class_name
                if class_path.is_dir():
                    num_images = len(list(class_path.glob("*.jpg"))) + len(list(class_path.glob("*.png")))
                    class_counts[class_name] = num_images
            
            distribution_data[split] = class_counts
            
            # Plot
            classes = list(class_counts.keys())
            counts = list(class_counts.values())
            
            axes[i].bar(classes, counts, color=self.colors[:len(classes)])
            axes[i].set_title(f'Distribuição - {split.upper()}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Classes')
            axes[i].set_ylabel('Número de Imagens')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Adicionar valores nas barras
            for j, count in enumerate(counts):
                axes[i].text(j, count + max(counts)*0.01, str(count), 
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.base_path / f"data_distribution_{self.timestamp}.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Salvar dados em JSON
        with open(self.base_path / f"distribution_analysis_{self.timestamp}.json", 'w') as f:
            json.dump(distribution_data, f, indent=2)
        
        return distribution_data
    
    def analyze_image_characteristics(self, data_dir, sample_size=50):
        """
        Analisa características das imagens (brilho, contraste, entropia)
        Base teórica: "Image Quality Assessment" (Wang & Bovik, 2006)
        """
        print("\n🎨 ANALISANDO CARACTERÍSTICAS DAS IMAGENS...")
        
        def calculate_brightness(img):
            """Calcula brilho médio da imagem"""
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            return np.mean(hsv[:,:,2])
        
        def calculate_contrast(img):
            """Calcula contraste usando desvio padrão"""
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return np.std(gray)
        
        def calculate_entropy(img):
            """Calcula entropia da imagem"""
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            hist = hist[hist > 0]  # Remove zeros para log
            return -np.sum(hist * np.log2(hist))
        
        characteristics = {'split': [], 'class': [], 'brightness': [], 'contrast': [], 'entropy': []}
        
        for split in ['train', 'val', 'test']:
            split_path = Path(data_dir) / split
            if not split_path.exists():
                continue
                
            for class_name in os.listdir(split_path):
                class_path = split_path / class_name
                if not class_path.is_dir():
                    continue
                    
                images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
                sampled_images = np.random.choice(images, min(sample_size, len(images)), replace=False)
                
                for img_path in sampled_images:
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            continue
                            
                        characteristics['split'].append(split)
                        characteristics['class'].append(class_name)
                        characteristics['brightness'].append(calculate_brightness(img))
                        characteristics['contrast'].append(calculate_contrast(img))
                        characteristics['entropy'].append(calculate_entropy(img))
                    except Exception as e:
                        print(f"Erro ao processar {img_path}: {e}")
        
        df = pd.DataFrame(characteristics)
        
        # Plot características por split e classe
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Brilho por split
        sns.boxplot(data=df, x='split', y='brightness', ax=axes[0,0], palette=self.colors)
        axes[0,0].set_title('Distribuição de Brilho por Split', fontweight='bold')
        
        # Contraste por split
        sns.boxplot(data=df, x='split', y='contrast', ax=axes[0,1], palette=self.colors)
        axes[0,1].set_title('Distribuição de Contraste por Split', fontweight='bold')
        
        # Brilho por classe
        sns.boxplot(data=df, x='class', y='brightness', ax=axes[1,0], palette='viridis')
        axes[1,0].set_title('Distribuição de Brilho por Classe', fontweight='bold')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Contraste por classe
        sns.boxplot(data=df, x='class', y='contrast', ax=axes[1,1], palette='viridis')
        axes[1,1].set_title('Distribuição de Contraste por Classe', fontweight='bold')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.base_path / f"image_characteristics_{self.timestamp}.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Estatísticas resumidas
        summary_stats = df.groupby(['split', 'class']).agg({
            'brightness': ['mean', 'std', 'min', 'max'],
            'contrast': ['mean', 'std', 'min', 'max'],
            'entropy': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        print("\n📈 ESTATÍSTICAS DAS CARACTERÍSTICAS DAS IMAGENS:")
        print(summary_stats)
        
        summary_stats.to_csv(self.base_path / f"image_stats_{self.timestamp}.csv")
        
        return df
    
    def analyze_augmentation_effectiveness(self, original_dir, augmented_dir):
        """
        Analisa a efetividade do data augmentation
        Base teórica: "A Survey on Image Data Augmentation" (Shorten & Khoshgoftaar, 2019)
        """
        print("\n🔄 ANALISANDO EFETIVIDADE DO DATA AUGMENTATION...")
        
        # Esta análise requer comparação entre diretórios original e aumentado
        # Implementação simplificada para demonstração
        
        augmentation_report = {
            'total_original': 0,
            'total_augmented': 0,
            'augmentation_factor': 0,
            'classes_analysis': {}
        }
        
        # Análise por classe
        for class_name in os.listdir(original_dir):
            original_class_path = Path(original_dir) / class_name
            augmented_class_path = Path(augmented_dir) / class_name
            
            if not original_class_path.exists() or not augmented_class_path.exists():
                continue
                
            original_count = len(list(original_class_path.glob("*.jpg"))) + len(list(original_class_path.glob("*.png")))
            augmented_count = len(list(augmented_class_path.glob("*.jpg"))) + len(list(augmented_class_path.glob("*.png")))
            
            augmentation_report['classes_analysis'][class_name] = {
                'original': original_count,
                'augmented': augmented_count,
                'factor': augmented_count / original_count if original_count > 0 else 0
            }
            
            augmentation_report['total_original'] += original_count
            augmentation_report['total_augmented'] += augmented_count
        
        if augmentation_report['total_original'] > 0:
            augmentation_report['augmentation_factor'] = (
                augmentation_report['total_augmented'] / augmentation_report['total_original']
            )
        
        # Plot
        classes = list(augmentation_report['classes_analysis'].keys())
        original_counts = [augmentation_report['classes_analysis'][c]['original'] for c in classes]
        augmented_counts = [augmentation_report['classes_analysis'][c]['augmented'] for c in classes]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(classes))
        width = 0.35
        
        ax.bar(x - width/2, original_counts, width, label='Original', color=self.colors[0])
        ax.bar(x + width/2, augmented_counts, width, label='Aumentado', color=self.colors[1])
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Número de Imagens')
        ax.set_title('Efetividade do Data Augmentation por Classe', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)
        ax.legend()
        
        # Adicionar valores
        for i, (orig, aug) in enumerate(zip(original_counts, augmented_counts)):
            ax.text(i - width/2, orig + max(augmented_counts)*0.01, str(orig), 
                   ha='center', va='bottom', fontweight='bold')
            ax.text(i + width/2, aug + max(augmented_counts)*0.01, str(aug), 
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.base_path / f"augmentation_analysis_{self.timestamp}.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Salvar relatório
        with open(self.base_path / f"augmentation_report_{self.timestamp}.json", 'w') as f:
            json.dump(augmentation_report, f, indent=2)
        
        print(f"\n📊 FATOR TOTAL DE AUMENTAÇÃO: {augmentation_report['augmentation_factor']:.2f}x")
        
        return augmentation_report
    
    def analyze_training_curves(self, history, model_name="model"):
        """
        Analisa as curvas de treinamento para detectar overfitting/underfitting
        Base teórica: "Visualizing the Loss Landscape of Neural Nets" (NeurIPS, 2018)
        """
        print("\n📈 ANALISANDO CURVAS DE TREINAMENTO...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss
        axes[0,0].plot(history.history['loss'], label='Train Loss', color=self.colors[0], linewidth=2)
        axes[0,0].plot(history.history['val_loss'], label='Val Loss', color=self.colors[1], linewidth=2)
        axes[0,0].set_title('Curva de Loss', fontweight='bold')
        axes[0,0].set_xlabel('Época')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0,1].plot(history.history['accuracy'], label='Train Accuracy', color=self.colors[0], linewidth=2)
        axes[0,1].plot(history.history['val_accuracy'], label='Val Accuracy', color=self.colors[1], linewidth=2)
        axes[0,1].set_title('Curva de Acurácia', fontweight='bold')
        axes[0,1].set_xlabel('Época')
        axes[0,1].set_ylabel('Acurácia')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Precision (se disponível)
        if 'precision' in history.history:
            axes[1,0].plot(history.history['precision'], label='Train Precision', color=self.colors[0], linewidth=2)
            axes[1,0].plot(history.history['val_precision'], label='Val Precision', color=self.colors[1], linewidth=2)
            axes[1,0].set_title('Curva de Precision', fontweight='bold')
            axes[1,0].set_xlabel('Época')
            axes[1,0].set_ylabel('Precision')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Recall (se disponível)
        if 'recall' in history.history:
            axes[1,1].plot(history.history['recall'], label='Train Recall', color=self.colors[0], linewidth=2)
            axes[1,1].plot(history.history['val_recall'], label='Val Recall', color=self.colors[1], linewidth=2)
            axes[1,1].set_title('Curva de Recall', fontweight='bold')
            axes[1,1].set_xlabel('Época')
            axes[1,1].set_ylabel('Recall')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.base_path / f"training_curves_{model_name}_{self.timestamp}.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Análise de overfitting
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        overfitting_ratio = final_val_loss / final_train_loss if final_train_loss > 0 else 0
        
        analysis = {
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'overfitting_ratio': overfitting_ratio,
            'overfitting_diagnosis': 'CRÍTICO' if overfitting_ratio > 2.0 else 
                                    'MODERADO' if overfitting_ratio > 1.5 else 
                                    'LEVE' if overfitting_ratio > 1.2 else 'MINIMO'
        }
        
        print(f"🔍 DIAGNÓSTICO DE OVERFITTING: {analysis['overfitting_diagnosis']} (Ratio: {overfitting_ratio:.2f})")
        
        return analysis
    
    def analyze_feature_space(self, model, test_data, class_names):
        """
        Analisa o espaço de características usando t-SNE
        Base teórica: "Visualizing Data using t-SNE" (van der Maaten & Hinton, 2008)
        """
        print("\n🧠 ANALISANDO ESPAÇO DE CARACTERÍSTICAS...")
        
        # Extrai features da penúltima camada
        feature_model = tf.keras.Model(
            inputs=model.input,
            outputs=model.layers[-2].output  # Camada antes da softmax
        )
        
        # Coleta features e labels
        features, labels = [], []
        for i in range(min(1000, len(test_data))):  # Amostra para performance
            batch_x, batch_y = test_data[i]
            batch_features = feature_model.predict(batch_x, verbose=0)
            features.extend(batch_features)
            labels.extend(np.argmax(batch_y, axis=1))
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Aplica t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, 
                            cmap='viridis', alpha=0.7, s=50)
        plt.colorbar(scatter)
        plt.title('Visualização t-SNE do Espaço de Características', fontweight='bold', fontsize=14)
        plt.xlabel('Componente t-SNE 1')
        plt.ylabel('Componente t-SNE 2')
        
        # Adicionar legendas para algumas classes
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label < len(class_names):
                indices = np.where(labels == label)[0]
                if len(indices) > 0:
                    centroid = np.mean(features_2d[indices], axis=0)
                    plt.annotate(class_names[label], centroid, 
                               xytext=(5, 5), textcoords='offset points',
                               fontweight='bold', fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(self.base_path / f"feature_space_tsne_{self.timestamp}.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        # Análise de clusters
        from sklearn.metrics import silhouette_score
        if len(unique_labels) > 1:
            silhouette_avg = silhouette_score(features_2d, labels)
            print(f"📊 Score de Silhueta: {silhouette_avg:.3f}")
            
            return {
                'silhouette_score': silhouette_avg,
                'feature_space_quality': 'EXCELENTE' if silhouette_avg > 0.7 else
                                        'BOM' if silhouette_avg > 0.5 else
                                        'RAZOÁVEL' if silhouette_avg > 0.3 else 'FRACO'
            }
        
        return {'silhouette_score': 0, 'feature_space_quality': 'INDETERMINADO'}
    
    def generate_comprehensive_report(self, analyses, output_file="comprehensive_report.md"):
        """
        Gera relatório completo com todos os diagnósticos
        """
        print("\n📋 GERANDO RELATÓRIO COMPREENSIVO...")
        
        report_path = self.base_path / f"{output_file}_{self.timestamp}"
        
        with open(report_path, 'w') as f:
            f.write("#RELATÓRIO DE DIAGNÓSTICO - MORGANA AI\n\n")
            f.write(f"**Data da Análise:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("##  RESUMO EXECUTIVO\n\n")
            
            # Resumo dos principais problemas
            f.write("### PRINCIPAIS PROBLEMAS IDENTIFICADOS:\n\n")
            
            if 'training_analysis' in analyses:
                of_diagnosis = analyses['training_analysis'].get('overfitting_diagnosis', 'N/A')
                f.write(f"- **Overfitting:** {of_diagnosis}\n")
            
            if 'feature_analysis' in analyses:
                fs_quality = analyses['feature_analysis'].get('feature_space_quality', 'N/A')
                f.write(f"- **Qualidade do Espaço de Features:** {fs_quality}\n")
            
            f.write("\n##  ANÁLISES DETALHADAS\n\n")
            
            for analysis_name, analysis_data in analyses.items():
                f.write(f"### {analysis_name.upper()}\n\n")
                f.write("```json\n")
                f.write(json.dumps(analysis_data, indent=2))
                f.write("\n```\n\n")
            
            f.write("##  RECOMENDAÇÕES TÉCNICAS\n\n")
            
            # Recomendações baseadas nas análises
            recommendations = [
                "1. **Data Augmentation**: Implementar transformações específicas do domínio agrícola",
                "2. **Regularização**: Aumentar dropout e considerar weight decay",
                "3. **Balanceamento**: Revisar estratégia de balanceamento de classes",
                "4. **Feature Engineering**: Considerar extração de características específicas de plantas",
                "5. **Validação Cruzada**: Implementar k-fold cross validation"
            ]
            
            for rec in recommendations:
                f.write(f"- {rec}\n")
            
            f.write("\n##ARQUIVOS GERADOS\n\n")
            for file in self.base_path.glob(f"*{self.timestamp}*"):
                f.write(f"- `{file.name}`\n")
        
        print(f"✅ Relatório salvo em: {report_path}")
        
        return report_path

# ============================================
# 🔧 INTEGRAÇÃO COM SEUS SCRIPTS EXISTENTES
# ============================================

def integrate_diagnostics_in_preprocessing():
    """
    Exemplo de integração no pré-processamento
    """
    diagnostico = MorganaDiagnostic()
    
    # Analisar distribuição original
    original_dist = diagnostico.analyze_data_distribution(r"F:\Projetos\Morgana-AI")
    
    # Analisar características das imagens
    img_chars = diagnostico.analyze_image_characteristics(r"F:\Projetos\Morgana-AI")
    
    # Após aumento de dados, analisar efetividade
    # aug_effect = diagnostico.analyze_augmentation_effectiveness(
    #     r"F:\Projetos\Morgana-AI",
    #     r"F:\Projetos\Morgana-AI\datasets_balanceado_v2"
    # )
    
    return diagnostico

def integrate_diagnostics_in_training(history, model, test_data, class_names):
    """
    Exemplo de integração no treinamento
    """
    diagnostico = MorganaDiagnostic()
    
    # Analisar curvas de treinamento
    training_analysis = diagnostico.analyze_training_curves(history, "morgana_model")
    
    # Analisar espaço de características
    feature_analysis = diagnostico.analyze_feature_space(model, test_data, class_names)
    
    # Gerar relatório completo
    comprehensive_report = diagnostico.generate_comprehensive_report({
        'training_analysis': training_analysis,
        'feature_analysis': feature_analysis
    })
    
    return diagnostico

# ============================================
# 🎯 EXEMPLO DE USO
# ============================================

if __name__ == "__main__":
    # Exemplo de uso
    print("🧪 EXECUTANDO DIAGNÓSTICO DA MORGANA AI...")
    
    # 1. Diagnóstico no pré-processamento
    diag_preprocess = integrate_diagnostics_in_preprocessing()
    
    print("\n" + "="*60)
    print("✅ DIAGNÓSTICO DO PRÉ-PROCESSAMENTO CONCLUÍDO!")
    print("="*60)
    
    # Nota: Para o treinamento, você chamaria:
    # diag_training = integrate_diagnostics_in_training(history, model, test_data, class_names)