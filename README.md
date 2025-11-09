#MorganaAI – Classificador de Pragas em Morangos Portátil

MorganaAI é um sistema embarcado com **inteligência artificial** desenvolvido para **detecção e classificação de pragas em morangos em tempo real**.  
O modelo foi treinado com técnicas de *Transfer Learning* (usando **MobileNetV2**) e otimizado para execução em dispositivos de baixo custo, como a **Raspberry Pi**.

O sistema permite identificar diversas pragas diretamente no campo, exibindo na tela o nome da praga detectada e sugestões de manejo, auxiliando o produtor rural na tomada de decisão imediata.

---

##🚀Tecnologias Utilizadas

- **Python 3.10**
- **TensorFlow 2.14.0**
- **Keras / MobileNetV2**
- **OpenCV**
- **Albumentations** (Data Augmentation)
- **Matplotlib / Seaborn** (visualizações)
- **NumPy / Pandas**
- **TQDM / PSUtil**

---

##Estrutura do Projeto

```
Morgana-AI/
├── .gitignore
├── requirements.txt
├── morgana_preprocessa_dataset.py
├── morgana_treino_local.py
├── rotulador_automatico.py
├── train/
├── val/
├── test/
└── datasets_balanceado_v2/  ← gerada automaticamente após o pré-processamento
```

---

##Dataset – MorganaAI

Antes de rodar o projeto, é necessário baixar o dataset utilizado nos scripts de pré-processamento e treinamento.

###Passos para configuração

1. **Acesse a pasta oficial do projeto no Google Drive:**
   👉 [MorganaAI - Google Drive](https://drive.google.com/drive/folders/1k1wPIpe3yUI3VJjb67UL81gTwHdqHR6S?usp=sharing)

2. **Baixe as três pastas disponíveis** e mova-as para a raiz do projeto local.

3. **Instale todas as dependências do projeto:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute o pré-processamento:**
   ```bash
   python morgana_preprocessa_dataset.py
   ```
   Isso criará automaticamente a pasta:
   ```
   datasets_balanceado_v2/
   ```
   Essa pasta contém o dataset processado e balanceado para a próxima etapa.

5. **Treine o modelo principal:**
   ```bash
   python morgana_treino_local.py
   ```
   Esse script realiza o **treinamento completo da IA**, aplicando *fine-tuning* e *data augmentation avançada*.  
   Ao final, são gerados:
   - O modelo final em `.h5` e `.tflite` (para uso embarcado);
   - A **matriz de confusão** (para análise de acertos e erros);
   - Gráficos de:
     - **Precisão (Precision)** vs **Revocação (Recall)** por classe;
     - **F1-Score** por classe;
   - Um **arquivo CSV com métricas detalhadas por classe**, ideal para análises e comparações futuras.

6. **(Opcional) Rotulagem automática:**
   O arquivo `rotulador_automatico.py` pode ser utilizado para rotular automaticamente novas imagens com base no modelo já treinado.

---

###Sobre a classe “não reconhecido”

A classe **`não reconhecido`** (ou *background*) contém imagens **aleatórias** — como pessoas, ruas, folhas, prédios e outros cenários que **não representam morangos nem pragas**.  
Ela é essencial para que a IA aprenda a **ignorar contextos irrelevantes**, reduzindo falsos positivos e tornando o modelo mais confiável em campo.

Essa abordagem segue boas práticas de *robust training* adotadas em sistemas de visão computacional de uso agrícola, melhorando a capacidade de generalização da rede.

---

##Saídas e Resultados

Após o treinamento, o modelo gera automaticamente:

- 📊 `matriz_confusao.png` — Matriz de confusão com o desempenho por classe.  
- 📈 `metricas_por_classe.csv` — Métricas individuais (Precision, Recall e F1-score) de cada classe.  
- 🧮 `metricas_globais.txt` — Resumo de desempenho (F1-macro e acurácia balanceada).  
- 🧠 `morganaAI_final.h5` e `morganaAI_final.tflite` — Modelos prontos para uso e exportação para a Raspberry Pi.  
- 📉 `morganaAI_evolucao.png` — Evolução das curvas de treinamento e validação por época.

---

##Resumo do fluxo de execução

```
Baixar pastas do Drive ➜ Instalar dependências ➜ Rodar preprocessamento
➜ Gerar datasets_balanceado_v2 ➜ Rodar treino_local
➜ Gerar modelo final (.h5 / .tflite) + métricas (CSV, gráficos, matriz de confusão)
```

---

