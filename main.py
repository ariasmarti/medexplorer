
"""
TRABAJO DE LABORATORIO 2: IA GENERATIVA
PROCESAMIENTO INICIAL DE TEXTO

El objetivo de este trabajo de laboratorio es **explorar el uso de técnicas de Procesamiento de Lenguaje Natural (PLN) y modelos de IA generativa**. A lo largo del notebook se desarrollan las siguientes etapas:

**•Procesamiento inicial de texto:** limpieza, normalización, tokenización, lematización y eliminación de stopwords.

**•Representaciones vectoriales:** construcción de embeddings con Word2Vec para analizar similitud semántica entre palabras y visualizar relaciones en espacios reducidos (PCA, t-SNE).

**•Agentes basados en LLMs:** implementación de un mini-agente capaz de responder preguntas médicas a partir de un corpus (medquad.csv) y, en otros casos, apoyarse en una búsqueda web externa (DuckDuckGo).
"""
"""## LEMATIZACION Y STOPWORDS"""

import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import re
import string
import contractions
import multiprocessing
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('stopwords')

"""wordnet: base de datos léxica del inglés que se usa para la lematización, es decir, para reducir las palabras a su forma raíz o “lema” (por ejemplo, running → run).

stopwords: lista de palabras vacías o irrelevantes (como “the”, “is”, “and”) que suelen eliminarse para mejorar el análisis semántico del texto.
"""

import re
import string

def clean_text_round1(text):
    text = text.lower()
    # eliminar símbolos raros como ¿ y %
    text = re.sub(r"[¿%]", " ", text)
    # eliminar puntuación estándar
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    # eliminar números y palabras con números
    text = re.sub(r"\w*\d\w*", "", text)
    return text

def clean_text_round2(text):
    # eliminar comillas raras y saltos de línea
    text = re.sub(r"[‘’“”…«»]", "", text)
    text = re.sub(r"\n", " ", text)
    return text

lemmatizer = WordNetLemmatizer()
sw = stopwords.words('english')

"""En este bloque se definen las funciones encargadas de preprocesar el texto bruto del corpus antes de aplicar técnicas de análisis o entrenamiento del modelo Word2Vec.
El objetivo es eliminar ruido y dejar las palabras en un formato uniforme y manejable.

**Detalles**:

*clean_text_round1(text):*
Convierte todo el texto a minúsculas.
Elimina símbolos especiales como ¿ y %.
Elimina signos de puntuación usando string.punctuation.
Quita números y palabras que contienen dígitos (por ejemplo, “covid19”).

*clean_text_round2(text):*
Elimina comillas tipográficas y caracteres especiales (‘’“”…«»).
Sustituye los saltos de línea por espacios, dejando el texto en una sola línea.

**Además, se inicializan dos componentes de NLTK:**

•lemmatizer = WordNetLemmatizer(): prepara el lematizador, que reducirá las palabras a su forma base (por ejemplo, “studies” → “study”).
•sw = stopwords.words('english'): carga la lista de stopwords en inglés, que luego podrán eliminarse por no aportar significado semántico relevante.

## LIMPIEZA
"""

df = pd.read_csv('medquad.csv', engine='python', quotechar='"') 

# usar las columnas 'question y answer' juntas
df['text'] = df['question'].fillna('') + " " + df['answer'].fillna('')

# funciones de limpieza
def clean_text_round1(text):
    text = text.lower()
    text = re.sub(r"[¿%]", " ", text)  # eliminar ¿ y %
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)  # puntuación
    text = re.sub(r"\w*\d\w*", "", text)  # palabras con números
    return text

def clean_text_round2(text):
    text = re.sub(r"[‘’“”…«»]", "", text)  # comillas raras
    text = re.sub(r"\n", " ", text)        # saltos de línea
    return text

# aplicar limpieza
round0 = lambda x: contractions.fix(x)
round1 = lambda x: clean_text_round1(x)
round2 = lambda x: clean_text_round2(x)

df['text'] = df['text'].astype(str)
df['clean_text'] = df['text'].apply(round0)
df['clean_text'] = df['clean_text'].apply(round1)
df['clean_text'] = df['clean_text'].apply(round2)

df[['question', 'answer', 'clean_text']].head()

"""##TOKENIZACION"""

input_data = [row.split() for row in df['clean_text']]
input_data[0]

"""En este paso se realiza la tokenización, es decir, la división del texto en palabras individuales (tokens).

• Cada fila del corpus se convierte en una lista de palabras.
• El conjunto completo (input_data) representa todo el corpus ya tokenizado.
--> Muestra el primer registro tokenizado, permitiendo verificar que las palabras se hayan separado correctamente.
"""

phrases = Phrases(input_data, min_count=30)
bigram = Phraser(phrases)
sentences = bigram[input_data]
sentences[0]

"""En este bloque se aplica la técnica de detector de bigramas usando la librería gensim.

Phrases(input_data, min_count=30): Entrena un modelo para detectar combinaciones frecuentes de palabras que aparezcan al menos 30 veces en el corpus.
Phraser(phrases): Crea una versión más eficiente del modelo de bigramas entrenado.
sentences = bigram[input_data]: Aplica el modelo a los textos tokenizados, reemplazando las palabras que forman bigramas por una única palabra unida con guion bajo (_).
sentences[0]: Muestra el primer texto con los bigramas detectados.
"""

bigrams_unicos = set()

for sentence in sentences:
    for word in sentence:
        if "_" in word:
            bigrams_unicos.add(word)

# mostrar los bigramas sin repetir
for b in bigrams_unicos:
    print("Bigrama encontrado:", b)

"""Este paso permite verificar qué combinaciones frecuentes de palabras fueron detectadas y unidas durante el preprocesamiento. Es una forma de inspeccionar la calidad del modelo de frases antes de entrenar el Word2Vec.

#EMBEDDINGS: WORD2VEC

## SKIP-GRAM
"""

cores = multiprocessing.cpu_count()
w2v_model = Word2Vec(sg=1, min_count=2, window=2, vector_size=300, sample=6e-5, alpha=0.03, min_alpha=0.0007, negative=20, workers=cores)

"""En este bloque se crea y configura el modelo Word2Vec, una técnica de embedding que convierte palabras en vectores numéricos según su contexto semántico.
Cada palabra del corpus se representa como un vector en un espacio de alta dimensión, donde las palabras con significados similares quedan cerca unas de otras.

**Explicación de los parámetros:**

•sg=1: usa el modelo Skip-Gram, que predice el contexto de una palabra. Es más eficaz con corpus grandes y técnicos.
•min_count=2: ignora palabras que aparezcan menos de 2 veces (reduce ruido).
•window=2: tamaño de la ventana de contexto (número de palabras a la izquierda y derecha que el modelo considera).
•vector_size=300: dimensión de los vectores que representan cada palabra.
•sample=6e-5: tasa de muestreo para reducir la frecuencia de palabras muy comunes.
•alpha=0.03 / min_alpha=0.0007: tasa de aprendizaje inicial y mínima durante el entrenamiento.
•negative=20: número de ejemplos negativos para el entrenamiento negative sampling (mejora la precisión).
•workers=cores: usa todos los núcleos disponibles del CPU para acelerar el proceso.
### Diccionario
"""

w2v_model.build_vocab(sentences)
# Obtener las palabras del vocabulario
words = list(w2v_model.wv.key_to_index.keys())
print (len(words))


w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
w2v_model.init_sims(replace=True)

"""En este bloque se entrena el modelo Word2Vec con las oraciones ya preprocesadas (sentences), para que aprenda representaciones numéricas (vectores) de las palabras.
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1) ➜ Inicia el entrenamiento del modelo.
sentences: conjunto de oraciones que se usan como datos de entrada.
total_examples=w2v_model.corpus_count: indica cuántas oraciones hay en el corpus, útil para controlar la cantidad de ejemplos usados.
epochs=30: el modelo recorrerá el corpus 30 veces completas para refinar los pesos y mejorar la calidad de los vectores.
report_delay=1: permite mostrar información de progreso cada cierto intervalo (en segundos).
RESUMEN este bloque entrena el modelo para que “aprenda” las relaciones semánticas entre palabras y deja listos los vectores para ser usados en tareas de similitud o visualización
"""

w2v_model.wv.most_similar(positive=["glaucoma"])

"""Busca las palabras más similares a "glaucoma" según el modelo. Internamente, Word2Vec compara los vectores de todas las palabras y devuelve aquellas con mayor similitud de coseno, es decir, las que aparecen en contextos parecidos dentro del corpus.
"""

w2v_model.wv.most_similar(negative=["glaucoma"])

w2v_model.wv.most_similar(positive=["diabetes", "high_blood", "glaucoma", "insulin"])

"""Busca las palabras que son más cercanas al promedio vectorial de las palabras dadas en la lista positive."""

w2v_model.wv.most_similar(negative=["diabetes", "high_blood", "glaucoma", "insulin"])

import plotly.graph_objs as go
from sklearn.decomposition import PCA
import numpy as np

# Seleccionar palabra específica y sus vecinas (aumentamos el número de palabras vecinas)
target_word = 'glaucoma'  # Reemplaza con la palabra de interés
neighbors = w2v_model.wv.most_similar(target_word, topn=50)  # Obtener las 50 palabras vecinas más similares
words = [target_word] + [neighbor[0] for neighbor in neighbors]
word_vectors = np.array([w2v_model.wv[word] for word in words])

# Reducir dimensiones a 3D con PCA
pca = PCA(n_components=3)
word_vectors_pca = pca.fit_transform(word_vectors)

# Crear trazado 3D con Plotly
trace = go.Scatter3d(
    x=word_vectors_pca[:, 0],
    y=word_vectors_pca[:, 1],
    z=word_vectors_pca[:, 2],
    mode='markers+text',
    text=words,
    marker=dict(size=5, color='blue', opacity=0.8),  # Ajustar el tamaño y la opacidad
    textposition="top center"
)

# Destacar la palabra objetivo en rojo
trace_target = go.Scatter3d(
    x=[word_vectors_pca[0, 0]],
    y=[word_vectors_pca[0, 1]],
    z=[word_vectors_pca[0, 2]],
    mode='markers+text',
    text=[target_word],
    marker=dict(size=10, color='red'),
    textposition="top center"
)

# Crear el layout
layout = go.Layout(
    title="Visualización de un Espacio Vectorial Más Grande",
    scene=dict(
        xaxis_title="PCA 1",
        yaxis_title="PCA 2",
        zaxis_title="PCA 3"
    ),
    margin=dict(l=0, r=0, b=0, t=50),
)

# Combinar los trazos
fig = go.Figure(data=[trace, trace_target], layout=layout)

# Mostrar la gráfica interactiva
fig.show()

"""este bloque proyecta el espacio semántico aprendido por Word2Vec y permite observar cómo las palabras relacionadas con “glaucoma” se agrupan en torno a él, mostrando visualmente las similitudes que el modelo capturó."""

from scipy.spatial.distance import cosine, cityblock

# Seleccionar las palabras de interés
words_of_interest = ["diabetes", "tumor", "glaucoma", "cancer","cataracts"]  # Puedes agregar más palabras

# Obtener los vectores de las palabras
word_vectors = np.array([w2v_model.wv[word] for word in words_of_interest])

# Crear trazado 3D con Plotly usando las dimensiones originales
trace_words = go.Scatter3d(
    x=word_vectors[:, 0],
    y=word_vectors[:, 1],
    z=word_vectors[:, 2],  # Usamos las primeras 3 dimensiones
    mode='markers+text',
    text=words_of_interest,
    marker=dict(size=10, color='blue'),
    textposition="top center"
)

# Lista para almacenar las líneas entre palabras
trace_lines = []

# Calcular y dibujar líneas para cada par de palabras
for i in range(len(words_of_interest)):
    for j in range(i+1, len(words_of_interest)):
        # Obtener los vectores de las palabras
        vector1 = word_vectors[i]
        vector2 = word_vectors[j]

        # Calcular distancias
        euclidean_distance = np.linalg.norm(vector1 - vector2)
        cosine_similarity = 1 - cosine(vector1, vector2)  # 1 - cosine para que 1 sea más similar
        manhattan_distance = cityblock(vector1, vector2)

        # Mostrar las distancias en la consola
        print(f"Distancias entre '{words_of_interest[i]}' y '{words_of_interest[j]}':")
        print(f"  Euclidiana: {euclidean_distance}")
        print(f"  Similitud de coseno: {cosine_similarity}")
        print(f"  Distancia Manhattan: {manhattan_distance}")
        print()

        # Dibujar línea entre las palabras para representar la distancia euclidiana
        trace_line = go.Scatter3d(
            x=[vector1[0], vector2[0]],
            y=[vector1[1], vector2[1]],
            z=[vector1[2], vector2[2]],
            mode='lines',
            line=dict(color='gray', width=2),
            text=f"Euclidiana: {euclidean_distance:.2f}",
            hoverinfo="text"
        )
        trace_lines.append(trace_line)

# Crear el layout
layout = go.Layout(
    title="Comparación de Distancias en el Espacio Vectorial",
    scene=dict(
        xaxis_title="Vector Dimension 1",
        yaxis_title="Vector Dimension 2",
        zaxis_title="Vector Dimension 3"
    ),
    margin=dict(l=0, r=0, b=0, t=50),
)

# Combinar los trazos de puntos y líneas
fig = go.Figure(data=[trace_words] + trace_lines, layout=layout)

# Mostrar la gráfica interactiva
fig.show()

"""Para cada par de palabras, se calculan tres métricas distintas:

Distancia euclidiana: mide la distancia “en línea recta” entre los vectores.
Cuanto más pequeña, más parecidos son los significados.

Similitud de coseno: mide el ángulo entre los vectores.
Valores cercanos a 1 indican alta similitud semántica.

Distancia Manhattan: mide la diferencia acumulada por componente.
Es más sensible a pequeños cambios en cada dimensión del vector.

Calcula un nuevo vector resultante y busca en el vocabulario cuál palabra está más cerca de ese punto.
"""

#most_similar(positive=[B, C], negative=[A]) #A es a B como C es a ___
analogous_words = w2v_model.wv.most_similar(positive=["prostate_cancer", "women"], negative=["men"])

print("La palabra que completa la analogía 'los hombres son al cancer de prostata como las mujeres son al ' ", analogous_words[0][0])

#chequeo de palabras en el vocabulario
for word in ['prostate_cancer', 'glaucoma', 'insulin', 'insulin']:
    print(word, "está" if word in w2v_model.wv.key_to_index else "no está")

"""## Cluster"""

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random

# 1 Elegimos más palabras del vocabulario
vocab_words = list(w2v_model.wv.key_to_index.keys())

 # Aumentamos el número de palabras usadas
sample_size = min(3000, len(vocab_words))  
random_words = random.sample(vocab_words, sample_size)
word_vectors = [w2v_model.wv[word] for word in random_words]

    # 2 PCA para inicializar t-SNE de forma más estable
pca = PCA(n_components=min(50, sample_size))
pca_result = pca.fit_transform(word_vectors)

    # 3 t-SNE a 3D
tsne = TSNE(
        n_components=3,
        perplexity=min(50, sample_size - 1),      
        n_iter=3000,         # más iteraciones para mayor separación
        learning_rate=200,
        random_state=42
    )
word_vectors_tsne = tsne.fit_transform(pca_result)

    # 4 Gráfico 3D interactivo
fig = go.Figure(data=[go.Scatter3d(
        x=word_vectors_tsne[:, 0],
        y=word_vectors_tsne[:, 1],
        z=word_vectors_tsne[:, 2],
        mode='markers+text',
        text=random_words,  # etiquetas completas
        marker=dict(
            size=5,
            opacity=0.8
        )
    )],
    layout=go.Layout(  # Add layout for 3D plot
        title="Visualización 3D de Clústeres de Palabras con t-SNE",
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
            zaxis_title="Dim 3"
        )
    ))
fig.show()

from sklearn.cluster import KMeans
# --- Clustering con KMeans ---
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
labels = kmeans.fit_predict(word_vectors_tsne)

# --- Palabras representativas por clúster ---
# (buscamos las más cercanas al centroide dentro de cada cluster)
representative_words = {}

for cluster_id in range(n_clusters):
    # índices de palabras en ese cluster
    cluster_indices = np.where(labels == cluster_id)[0]

    # obtenemos los vectores de esas palabras
    cluster_vectors = word_vectors_tsne[cluster_indices]

    # centroide del cluster
    centroid = kmeans.cluster_centers_[cluster_id]

    # distancia euclidiana de cada palabra al centroide
    distances = np.linalg.norm(cluster_vectors - centroid, axis=1)

    # ordenamos las más cercanas
    closest_indices = cluster_indices[np.argsort(distances)[:10]]  # top 10

    # guardamos las palabras representativas
    representative_words[cluster_id] = [random_words[i] for i in closest_indices]

# Mostrar las palabras representativas
for cluster_id, words in representative_words.items():
    print(f"Cluster {cluster_id}: {', '.join(words)}")

# --- Visualización 3D ---
annotate_indices = random.sample(range(len(random_words)), 150)

fig = go.Figure(data=[go.Scatter3d(
    x=word_vectors_tsne[:, 0],
    y=word_vectors_tsne[:, 1],
    z=word_vectors_tsne[:, 2],
    mode='markers+text',
    text=[random_words[i] if i in annotate_indices else '' for i in range(len(random_words))],
    marker=dict(
        size=4,
        color=labels,
        colorscale='Viridis',
        opacity=0.8
    )
)])

fig.update_layout(
    title="Clusters de palabras en 3D con KMeans",
    scene=dict(
        xaxis_title='Dim 1',
        yaxis_title='Dim 2',
        zaxis_title='Dim 3'
    )
)

fig.show()

"""Cluster 0: términos médicos/anatómicos (endometrial, prostates, toxin…)
Cluster 1: alimentos o consumo (meat, water, refrigerated…)
Cluster 3: inmunidad o enfermedades (gut, attack, hiv_aids…)
Cluster 7: genes/proteínas/regulación molecular

# AGENTES BASADOS EN LLM

- Implementar un mini-agente capaz de responder preguntas sobre un corpus y
ejecutar al menos una acción adicional (ejemplo: cálculo matemático, búsqueda
externa simulada)
- Analizar el flujo de trabajo de su mini-agente
- Describir paso a paso cómo procesa una consulta desde que el usuario ingresa la
pregunta hasta que se genera la respuesta final.
"""

from huggingface_hub import InferenceClient

from dotenv import load_dotenv
import os

load_dotenv()  # busca el archivo .env y carga las variables
HF_TOKEN = os.environ["HF_TOKEN"]

"""Aquí se define una variable de entorno con el token personal de Hugging Face (HF_TOKEN).
Este token permite autenticar la sesión y usar los modelos sin restricciones de acceso.
"""

from openai import OpenAI

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

models = client.models.list()
print("Modelos disponibles:")
for m in models.data:
    print("-", m.id)

"""este bloque configura el cliente de conexión y recupera la lista de modelos de lenguaje disponibles en Hugging Face, dejando el entorno listo para ejecutar inferencias con cualquiera de ellos."""

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
resp = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{"role": "user", "content": "What is the capital of France?"}],
)

print(resp.choices[0].message.content)

"""## TOOL"""

import pandas as pd
from typing import Dict, Callable
df = pd.read_csv("medquad.csv")

def tool_corpus_search(query: str) -> str:
    """
    Busca respuestas en el corpus medquad.csv.
    Hace un match simple de palabras clave en la columna 'question' y devuelve respuestas.
    """
    query_low = query.lower()
    matches = df[df['question'].str.lower().str.contains(query_low, na=False)]
    if matches.empty:
        return "No encontré nada en el corpus."
    resultados = []
    for i, row in matches.head(3).iterrows():
        resultados.append(f"Q: {row['question']}\nA: {row['answer']}")
    return "\n\n".join(resultados)

"""En esta parte cargamos el dataset medquad.csv, que contiene pares de preguntas y respuestas médicas. Después definimos la función tool_corpus_search, que sirve como una especie de buscador interno: toma una consulta del usuario, la pasa a minúsculas y busca si esas palabras aparecen dentro de la columna question. Si encuentra coincidencias, devuelve hasta 3 resultados mostrando la pregunta original y su respuesta. Si no encuentra nada, devuelve un mensaje avisando que no se hallaron resultados."""


def tool_corpus_search(query: str, require_in_vocab=False) -> str:
    query_clean = clean_text(query)
    query_words = [w for w in query_clean.split() if w not in sw]
    if not query_words:
        return "No encontré palabras relevantes para buscar en el corpus."

    # Si pedimos que las palabras deben estar en el vocabulario del modelo
    if require_in_vocab:
        query_words = [w for w in query_words if w in VOCAB]
        if not query_words:
            return "Ninguna palabra relevante de la consulta está en el vocabulario del corpus."

    # heurística: si la query tiene 1 palabra -> exigir 1 coincidencia; si >1 -> exigir 2
    min_overlap = 1 if len(query_words) == 1 else 2

    # función que comprueba si la pregunta del CSV comparte suficientes tokens
    def is_match(tokens):
        # tokens ya vienen limpios de question_tokens
        return len(set(tokens) & set(query_words)) >= min_overlap

    matches = df[df['question_tokens'].apply(is_match)]

    # Debug extra: si no hay match, devolver por qué (palabras buscadas)
    if matches.empty:
        return f"No encontré nada en el corpus. (debug: query_words={query_words}, min_overlap={min_overlap})"

    resultados = []
    for i, row in matches.head(5).iterrows():
        resultados.append(f"Q: {row['question']}\nA: {row['answer']}")
    return "\n\n".join(resultados)

from ddgs import DDGS
def tool_ddg_search(arg: str) -> str:
    """
    Busca en la web con DuckDuckGo y devuelve top-K resultados (título, url, snippet).
    """
    try:
        results = []
        with DDGS() as ddgs:
            for item in ddgs.text(arg, region="es-es", safesearch="moderate", max_results=10):
                title = (item.get("title") or "").strip()
                href = (item.get("href") or "").strip()
                body = (item.get("body") or "").strip().replace("\n", " ")
                if title and href:
                    results.append(f"{title}\n{href}\n{body}")
        if not results:
            return "Sin resultados."
        return "\n\n".join(results)
    except Exception as e:
        return f"Error en ddg_search: {e}"

"""Aquí se define la función tool_ddg_search, que permite hacer búsquedas en la web usando la librería ddgs (DuckDuckGo Search). La función recibe una consulta (arg) y obtiene hasta 10 resultados relevantes. De cada resultado extrae el título, el enlace (URL) y un pequeño fragmento de texto (snippet). Luego devuelve esa información en un formato legible.
Si no encuentra nada, devuelve el mensaje “Sin resultados.”. En caso de que ocurra un error durante la búsqueda, devuelve un mensaje con la descripción del error.
En resumen, esta función actúa como un buscador externo, complementando al buscador interno que consulta solo el corpus.
"""

# Registro de tools
TOOLS: Dict[str, Callable[[str], str]] = {
    "corpus_search": tool_corpus_search,
    "ddg_search": tool_ddg_search,
}

"""En este bloque se crea un diccionario llamado TOOLS donde se registran las funciones que definimos antes:
"corpus_search" → apunta a la función que busca en el dataset medquad.csv.
"ddg_search" → apunta a la función que busca en la web con DuckDuckGo.
De esta forma, el agente puede llamar a cualquiera de estas herramientas de manera dinámica, simplemente invocando el nombre de la key del diccionario.
"""

TOOLS_DESC = "\n".join([
    "- corpus_search(input: str): busca respuestas en el dataset médico medquad.csv",
    "- ddg_search(input: str): hace una búsqueda en la web (top-3 resultados)"
])

"""Aquí se define la variable TOOLS_DESC, que contiene un texto explicativo de las herramientas registradas.
Básicamente se genera un string donde se enumeran:
corpus_search: indica que busca respuestas en el dataset médico medquad.csv.
ddg_search: indica que realiza una búsqueda en la web y devuelve los primeros resultados encontrados.
"""

from difflib import get_close_matches

VOCAB = set(w2v_model.wv.key_to_index.keys())
def agente(query: str) -> str:
    """
    Decide si buscar en el corpus o en la web.
    Si el corpus no devuelve resultados útiles, pasa a ddg_search.
    """
    q_clean = clean_text(query)
    q_tokens = [w for w in q_clean.split() if w not in sw]

    # 1Intentar buscar en el corpus (sin exigir vocab por si no hay modelo)
    corpus_result = tool_corpus_search(query, require_in_vocab=False)

    # 2 Si encontró resultados útiles → devolverlos
    if "Q:" in corpus_result:  # detecta si hay al menos una pregunta devuelta
        return corpus_result

    # 3 Si no encontró nada → usar la búsqueda web
    return TOOLS["ddg_search"](query)

print("\nPregunta 1 (médica, corpus):")
print(agente("What is obesity?"))

print("\nPregunta 2 (general, web):")
print(agente("avances IA medicina"))

