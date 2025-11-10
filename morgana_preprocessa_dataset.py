# ============================================
# 🌿 MORGANA AI - Pré-processamento com Visualizações Corrigidas
# ============================================

import os
import cv2
import random
import csv
import numpy as np
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from datetime import datetime
from albumentations import (
    Compose, RandomBrightnessContrast, GaussianBlur, GaussNoise, 
    HueSaturationValue, ShiftScaleRotate, MotionBlur
)


class DataPreprocessor:
    """
    Classe para pré-processamento e aumento de dados com visualizações funcionais
    """
    
    def __init__(self, base_path, output_path, img_size=(320, 320), threads=6):
        self.base_path = Path(base_path)
        self.output_path = Path(output_path)
        self.img_size = img_size
        self.threads = threads
        self.visualization_dir = output_path / "augmentation_visualizations"
        
        # Criar diretórios
        for split in ["train", "val", "test"]:
            (self.output_path / split).mkdir(parents=True, exist_ok=True)
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar seeds para reproducibilidade
        random.seed(42)
        np.random.seed(42)
        
        # 🔥 CORREÇÃO: Importar diagnóstico apenas quando necessário
        try:
            sys.path.append(str(self.base_path.parent))
            from morgana_diagnostic import MorganaDiagnostic
            self.diagnostic = MorganaDiagnostic(base_path=self.output_path / "diagnostics")
            self.diagnostic_available = True
        except ImportError:
            print("⚠️  Diagnóstico não disponível - continuando sem análises detalhadas")
            self.diagnostic_available = False
    
    def create_augmentation_pipeline(self, augmentation_intensity="moderate"):
        """
        Cria pipeline de aumento de dados
        """
        if augmentation_intensity == "light":
            return Compose([
                RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.5
                ),
                HueSaturationValue(
                    hue_shift_limit=5,
                    sat_shift_limit=10,
                    val_shift_limit=5,
                    p=0.3
                )
            ])
        
        elif augmentation_intensity == "moderate":
            return Compose([
                ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=15,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.6
                ),
                RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.7
                ),
                HueSaturationValue(
                    hue_shift_limit=10,
                    sat_shift_limit=20,
                    val_shift_limit=10,
                    p=0.5
                ),
                MotionBlur(blur_limit=(3, 5), p=0.3),
                GaussNoise(var_limit=(5, 15), p=0.3),
            ])
        
        else:  # strong
            return Compose([
                ShiftScaleRotate(
                    shift_limit=0.08,
                    scale_limit=0.15,
                    rotate_limit=20,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.7
                ),
                RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.8
                ),
                HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=25,
                    val_shift_limit=15,
                    p=0.6
                ),
                MotionBlur(blur_limit=(3, 6), p=0.4),
                GaussNoise(var_limit=(10, 20), p=0.4),
                GaussianBlur(blur_limit=(3, 5), p=0.3)
            ])
    
    def get_pipeline_for_class(self, class_name):
        """
        Determina a intensidade do augmentation baseado no tipo de classe
        """
        background_classes = ["nao_reconhecido"]
        new_classes = ["morango_verde", "folha_saudavel"]
        
        if class_name in background_classes:
            return "light"
        elif class_name in new_classes:
            return "moderate"
        else:
            return "moderate"
    
    def save_augmentation_comparison(self, original, augmented, class_name, split_name, image_name):
        """
        🔥 CORREÇÃO: Salva visualização de forma segura sem problemas de threading
        """
        try:
            # Usar backend Agg que não depende de GUI e é thread-safe
            import matplotlib
            matplotlib.use('Agg')  # 🔥 CRÍTICO: Backend não-interativo
            import matplotlib.pyplot as plt
            
            # Converter BGR para RGB
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            augmented_rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
            
            # Criar figura comparativa
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Imagem original
            axes[0].imshow(original_rgb)
            axes[0].set_title(f'Original\n{class_name} - {split_name}', 
                            fontsize=10, fontweight='bold')
            axes[0].axis('off')
            
            # Imagem aumentada
            axes[1].imshow(augmented_rgb)
            axes[1].set_title('Augmented', fontsize=10, fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            
            # Salvar imagem
            filename = f"{class_name}_{split_name}_{image_name}_comparison.png"
            filepath = self.visualization_dir / filename
            plt.savefig(filepath, dpi=100, bbox_inches='tight', 
                       facecolor='white', pad_inches=0.1)
            plt.close(fig)  # 🔥 Fechar figura para liberar memória
            
            return filename
            
        except Exception as e:
            print(f"⚠️  Erro ao salvar visualização: {str(e)[:100]}...")
            return None
    
    def count_images(self, directory_path):
        """
        Conta imagens em um diretório por classe
        """
        distribution = {}
        for class_name in os.listdir(directory_path):
            class_path = directory_path / class_name
            if class_path.is_dir():
                jpg_count = len(list(class_path.glob("*.jpg")))
                png_count = len(list(class_path.glob("*.png")))
                jpeg_count = len(list(class_path.glob("*.jpeg")))
                distribution[class_name] = jpg_count + png_count + jpeg_count
        return distribution
    
    def calculate_balancing_factors(self, distribution, split_name):
        """
        Calcula fatores de balanceamento para equalizar as classes
        """
        factors = {}
        average = np.mean(list(distribution.values()))
        
        print(f"\n📊 Balanceamento - {split_name}:")
        print(f"   Média por classe: {average:.0f} imagens")
        
        for class_name, count in distribution.items():
            if count == 0:
                factors[class_name] = 0
                print(f"   ⚠️  {class_name}: SEM IMAGENS")
                continue
            
            # Calcular fator de aumento
            if count >= average:
                factor = 1
            else:
                factor = max(1, int(average / count))
            
            # Limitar aumento máximo
            factor = min(factor, 8)
            
            factors[class_name] = factor
            print(f"   {class_name:20} ({count:3} imagens) → x{factor}")
        
        return factors
    
    def process_single_image(self, image_path, class_name, split_name, csv_writer, lock, augmentation_round=0):
        """
        Processa uma única imagem com augmentation e visualização
        """
        try:
            # Ler e redimensionar imagem
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"⚠️  Não foi possível ler: {image_path}")
                return False
            
            image_resized = cv2.resize(image, self.img_size)
            
            # Determinar pipeline baseado na classe
            pipeline_intensity = self.get_pipeline_for_class(class_name)
            pipeline = self.create_augmentation_pipeline(pipeline_intensity)
            
            # Aplicar augmentation
            augmented = pipeline(image=image_resized)["image"]
            
            # Salvar imagem aumentada
            output_dir = self.output_path / split_name / class_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            round_suffix = f"_r{augmentation_round}" if augmentation_round > 0 else ""
            output_filename = f"{image_path.stem}_aug{round_suffix}_{random.randint(1000, 9999)}.jpg"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), augmented)
            
            # 🔥 CORREÇÃO: Salvar visualização de forma controlada
            visualization_file = None
            should_save_viz = (random.random() < 0.15)  # 15% de chance
            
            if should_save_viz:
                visualization_file = self.save_augmentation_comparison(
                    image_resized, augmented, class_name, split_name, 
                    f"{image_path.stem}{round_suffix}"
                )
            
            # Registrar no CSV
            with lock:
                csv_writer.writerow([
                    datetime.now(),
                    split_name,
                    class_name,
                    image_path.name,
                    output_filename,
                    pipeline_intensity,
                    visualization_file or ""
                ])
                
            return True
                
        except Exception as e:
            print(f"❌ Erro ao processar {image_path}: {e}")
            return False
    
    def generate_visualization_report(self):
        """
        Gera relatório das visualizações criadas
        """
        visualization_files = list(self.visualization_dir.glob("*.png"))
        
        if not visualization_files:
            print("📝 Nenhuma visualização foi gerada")
            return
        
        print(f"\n📈 Relatório de Visualizações:")
        print(f"   Total de visualizações: {len(visualization_files)}")
        
        # Agrupar por classe
        class_counts = {}
        for file in visualization_files:
            class_name = file.name.split('_')[0]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"   Distribuição por classe:")
        for class_name, count in class_counts.items():
            print(f"     {class_name}: {count} visualizações")
        
        # Mostrar exemplos
        sample_files = random.sample(visualization_files, min(3, len(visualization_files)))
        print(f"\n👀 Exemplos de visualizações:")
        for viz_file in sample_files:
            print(f"   📸 {viz_file.name}")
    
    def analyze_data_distribution(self):
        """
        Análise simplificada da distribuição de dados
        """
        print("\n🔍 Analisando distribuição de dados...")
        
        distribution_data = {}
        
        for split_name in ["train", "val", "test"]:
            split_path = self.base_path / split_name
            if not split_path.exists():
                continue
                
            distribution = self.count_images(split_path)
            distribution_data[split_name] = distribution
            
            print(f"\n📁 {split_name.upper()}:")
            total_images = sum(distribution.values())
            print(f"   Total de imagens: {total_images}")
            print(f"   Número de classes: {len(distribution)}")
            
            for class_name, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_images) * 100
                print(f"   {class_name:20}: {count:3} imagens ({percentage:5.1f}%)")
        
        return distribution_data
    
    def run_preprocessing(self):
        """
        Executa o pipeline completo de pré-processamento
        """
        print("🚀 Iniciando pré-processamento...")
        
        # Analisar dataset original
        original_distribution = self.analyze_data_distribution()
        
        log_path = self.output_path / "processing_log.csv"
        lock = Lock()
        
        statistics = {
            'total_processed': 0,
            'total_augmented': 0,
            'augmented_by_class': {},
            'visualizations_created': 0
        }
        
        with open(log_path, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "timestamp", "split", "class", "original_image", 
                "augmented_image", "augmentation_intensity", "visualization"
            ])
            
            for split_name in ["train", "val", "test"]:
                split_path = self.base_path / split_name
                if not split_path.exists():
                    print(f"⚠️  Diretório {split_name} não encontrado")
                    continue
                
                distribution = self.count_images(split_path)
                balancing_factors = self.calculate_balancing_factors(distribution, split_name)
                
                for class_name, original_count in distribution.items():
                    if original_count == 0:
                        continue
                    
                    statistics['augmented_by_class'][class_name] = 0
                    factor = balancing_factors[class_name]
                    
                    # Encontrar imagens
                    image_patterns = ["*.jpg", "*.png", "*.jpeg"]
                    image_paths = []
                    for pattern in image_patterns:
                        image_paths.extend(list((split_path / class_name).glob(pattern)))
                    
                    print(f"\n🎯 Processando {split_name}/{class_name}: {len(image_paths)} imagens → x{factor}")
                    
                    # Processar múltiplas vezes baseado no fator
                    for augmentation_round in range(factor):
                        round_suffix = f" (rodada {augmentation_round + 1}/{factor})" if factor > 1 else ""
                        
                        with ThreadPoolExecutor(max_workers=self.threads) as executor:
                            futures = []
                            for image_path in tqdm(image_paths, 
                                                 desc=f"{split_name}/{class_name}{round_suffix}",
                                                 ncols=80):
                                statistics['total_processed'] += 1
                                
                                future = executor.submit(
                                    self.process_single_image,
                                    image_path, class_name, split_name, 
                                    csv_writer, lock, augmentation_round
                                )
                                futures.append(future)
                            
                            # Aguardar conclusão e contar sucessos
                            for future in futures:
                                try:
                                    success = future.result(timeout=60)  # Timeout aumentado
                                    if success:
                                        statistics['total_augmented'] += 1
                                        statistics['augmented_by_class'][class_name] += 1
                                except Exception as e:
                                    print(f"❌ Falha no processamento: {e}")
        
        print("\n✅ Pré-processamento concluído!")
        
        # Gerar relatórios
        self.generate_visualization_report()
        
        # Estatísticas finais
        print(f"\n📊 Estatísticas do Processamento:")
        print(f"   Total de imagens processadas: {statistics['total_processed']}")
        print(f"   Total de imagens aumentadas: {statistics['total_augmented']}")
        print(f"   Eficiência: {(statistics['total_augmented']/statistics['total_processed']*100):.1f}%")
        
        print(f"\n📈 Imagens aumentadas por classe:")
        for class_name, count in sorted(statistics['augmented_by_class'].items(), 
                                      key=lambda x: x[1], reverse=True):
            original_count = sum([dist.get(class_name, 0) for dist in original_distribution.values()])
            increase_factor = count / original_count if original_count > 0 else 0
            print(f"   {class_name:20}: {count:5} imagens (x{increase_factor:.1f})")
        
        # Análise do dataset processado
        if self.diagnostic_available:
            print("\n🔍 Analisando dataset processado...")
            try:
                processed_distribution = self.diagnostic.analyze_data_distribution(self.output_path)
                
                # Gerar relatório completo
                self.diagnostic.generate_comprehensive_report({
                    'original_distribution': original_distribution,
                    'processed_distribution': processed_distribution,
                    'processing_statistics': statistics
                }, output_file="preprocessing_report")
            except Exception as e:
                print(f"⚠️  Erro na análise do diagnóstico: {e}")
        
        print(f"\n🎊 Processamento finalizado!")
        print(f"   📁 Dataset: {self.output_path}")
        print(f"   📊 Visualizações: {self.visualization_dir}")
        print(f"   📋 Log: {log_path}")


# ============================================
# EXECUÇÃO PRINCIPAL
# ============================================

if __name__ == "__main__":
    import sys
    
    # Configurações
    BASE_PATH = Path(r"C:\Users\Gustavo\Desktop\TCC\strawberry-ai\ia\MorganaAI")
    OUTPUT_PATH = Path(r"C:\Users\Gustavo\Desktop\TCC\strawberry-ai\ia\MorganaAI\datasets_balanceado_v3")
    IMG_SIZE = (320, 320)
    THREADS = 4  # Reduzido para maior estabilidade
    
    # Executar pré-processamento
    preprocessor = DataPreprocessor(BASE_PATH, OUTPUT_PATH, IMG_SIZE, THREADS)
    preprocessor.run_preprocessing()