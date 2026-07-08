import pandas as pd
import numpy as np
import re
import string
import multiprocessing
import pickle

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import contractions

from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec

nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
sw = set(stopwords.words('english'))


def clean_text_round1(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[¿%]", " ", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


def clean_text_round2(text: str) -> str:
    text = re.sub(r"[''""…«»]", "", text)
    text = re.sub(r"\n", " ", text)
    return text


def clean_text(text: str) -> str:
    text = contractions.fix(str(text))
    text = clean_text_round1(text)
    text = clean_text_round2(text)
    return text


def tokenize_clean(text: str) -> list:
    cleaned = clean_text(text)
    return [w for w in cleaned.split() if w not in sw and len(w) > 1]


print("Cargando medquad.csv...")
df = pd.read_csv('medquad.csv', engine='python', quotechar='"')

df['text'] = df['question'].fillna('') + " " + df['answer'].fillna('')
df['text'] = df['text'].astype(str)

print("Limpiando texto...")
df['clean_text'] = df['text'].apply(clean_text)
df['question_tokens'] = df['question'].fillna('').apply(tokenize_clean)

print(f"Filas en el corpus: {len(df)}")

print("Tokenizando y detectando bigramas...")
input_data = [row.split() for row in df['clean_text']]

phrases = Phrases(input_data, min_count=30)
bigram = Phraser(phrases)
sentences = bigram[input_data]

bigrams_unicos = sorted({w for s in sentences for w in s if "_" in w})
print(f"Bigramas detectados: {len(bigrams_unicos)}")

print("Entrenando Word2Vec...")
cores = multiprocessing.cpu_count()

w2v_model = Word2Vec(
    sg=1,
    min_count=2,
    window=2,
    vector_size=300,
    sample=6e-5,
    alpha=0.03,
    min_alpha=0.0007,
    negative=20,
    workers=cores
)

w2v_model.build_vocab(sentences)
print(f"Vocabulario: {len(w2v_model.wv.key_to_index)} palabras")

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print("Guardando artefactos...")

w2v_model.save("w2v_model.model")
df[['question', 'answer', 'clean_text', 'question_tokens']].to_pickle("corpus_procesado.pkl")

with open("bigram_model.pkl", "wb") as f:
    pickle.dump(bigram, f)

with open("stopwords.pkl", "wb") as f:
    pickle.dump(sw, f)

print("Listo. Archivos generados: w2v_model.model, corpus_procesado.pkl, bigram_model.pkl, stopwords.pkl")