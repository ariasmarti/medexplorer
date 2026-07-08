# MedExplorer — NLP y agente de búsqueda médica

Proyecto de procesamiento de lenguaje natural aplicado a un corpus de preguntas y respuestas médicas (MedQuAD). Incluye un pipeline de limpieza y tokenización de texto, entrenamiento de embeddings con Word2Vec y un agente de búsqueda híbrida que combina recuperación local con búsqueda web.

Desarrollado como Trabajo de Laboratorio para la materia *IA Generativa*, FCEFyN - UNC, 2025.

## Qué hace

**Procesamiento de texto**
- Limpieza, normalización, tokenización y lematización del corpus médico
- Detección automática de bigramas frecuentes (ej: `high_blood`, `prostate_cancer`)
- Eliminación de stopwords en inglés

**Embeddings con Word2Vec**
- Entrenamiento de un modelo Skip-Gram sobre el corpus procesado
- Análisis de similitud semántica entre términos médicos
- Resolución de analogías vectoriales
- Visualización del espacio latente con PCA y t-SNE (2D y 3D)
- Clustering con KMeans sobre las representaciones aprendidas

**Agente de búsqueda híbrida**
- Busca primero en el corpus MedQuAD por solapamiento de tokens
- Si no encuentra resultados relevantes, recurre a DuckDuckGo como respaldo
- Integración opcional con LLM externo (Qwen 2.5 vía Hugging Face Inference API)

## Estructura

```
├── procesar_corpus.py   # limpieza del texto + entrenamiento Word2Vec + guardado de artefactos
├── app.py               # interfaz Streamlit (requiere artefactos generados previamente)
├── requirements.txt
└── .env.example         # plantilla para el token de Hugging Face
```

## Cómo correrlo

**1. Instalá las dependencias**
```bash
pip install -r requirements.txt
```

**2. Configurá el token de Hugging Face** (necesario solo para el agente con LLM)

Copiá `.env.example` como `.env` y pegá tu token:
```
HF_TOKEN=tu_token_aqui
```
Podés generar uno en https://huggingface.co/settings/tokens

**3. Descargá el dataset**

El corpus MedQuAD no está incluido en el repositorio por su tamaño. Podés descargarlo desde [Kaggle](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research) y colocarlo como `medquad.csv` en la carpeta raíz.

**4. Procesá el corpus**
```bash
python procesar_corpus.py
```
Esto genera los artefactos necesarios para la app (`w2v_model.model`, `corpus_procesado.pkl`, etc). Tarda algunos minutos la primera vez.

**5. Levantá la app**
```bash
streamlit run app.py
```

## Tecnologías

Python · NLTK · Gensim · scikit-learn · Plotly · Streamlit · Hugging Face · DuckDuckGo Search
