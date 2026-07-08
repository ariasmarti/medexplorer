# app.py
# Ejecutar con: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import os

from gensim.models import Word2Vec
from dotenv import load_dotenv

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="MedExplorer · NLP + Agente médico",
    page_icon="🩺",
    layout="wide",
)

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Fraunces:ital,wght@0,300;1,300&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding: 2rem 3rem; max-width: 1200px; }
    h1, h2, h3 { font-family: 'Fraunces', serif; font-weight: 300; }

    .hero { display:flex; align-items:baseline; gap:.75rem; margin-bottom:.25rem; }
    .hero-title { font-family:'Fraunces',serif; font-size:2.6rem; font-weight:300;
                  color:#0f172a; letter-spacing:-.02em; }
    .hero-sub { font-size:.95rem; color:#64748b; margin-bottom:1.5rem; }

    .chat-bubble-user { background:#0f172a; color:white; padding:.8rem 1.2rem;
                        border-radius:14px 14px 2px 14px; margin:.5rem 0; max-width:80%;
                        margin-left:auto; }
    .chat-bubble-bot { background:#f1f5f9; color:#0f172a; padding:.8rem 1.2rem;
                       border-radius:14px 14px 14px 2px; margin:.5rem 0; max-width:85%;
                       white-space:pre-wrap; }
    .source-tag { display:inline-block; font-size:.7rem; font-weight:600;
                  text-transform:uppercase; letter-spacing:.05em; padding:.2rem .6rem;
                  border-radius:6px; margin-bottom:.4rem; }
    .source-corpus { background:#dcfce7; color:#166534; }
    .source-web { background:#dbeafe; color:#1e40af; }

    div.stButton > button {
        background:#0f172a; color:white; border:none;
        padding:.6rem 1.5rem; border-radius:8px; font-weight:600; cursor:pointer;
    }
    div.stButton > button:hover { background:#1e293b; }

    .footer { font-size:.75rem; color:#94a3b8; text-align:center; margin-top:3rem; }
</style>
""", unsafe_allow_html=True)

# ── Carga de artefactos (cacheada) ────────────────────────────────────────────
@st.cache_resource
def cargar_artefactos():
    w2v_model = Word2Vec.load("w2v_model.model")
    df = pd.read_pickle("corpus_procesado.pkl")
    with open("bigram_model.pkl", "rb") as f:
        bigram = pickle.load(f)
    with open("stopwords.pkl", "rb") as f:
        sw = pickle.load(f)
    return w2v_model, df, bigram, sw


try:
    w2v_model, df, bigram, sw = cargar_artefactos()
    VOCAB = set(w2v_model.wv.key_to_index.keys())
    artefactos_ok = True
except FileNotFoundError:
    artefactos_ok = False

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <span class="hero-title">🩺 MedExplorer</span>
</div>
<p class="hero-sub">Procesamiento de Lenguaje Natural + Agente médico con búsqueda híbrida</p>
""", unsafe_allow_html=True)

if not artefactos_ok:
    st.error(
        "⚠️ No se encontraron los artefactos necesarios (`w2v_model.model`, `corpus_procesado.pkl`, etc). "
        "Ejecutá primero:\n\n```\npython procesar_corpus.py\n```"
    )
    st.stop()

tab1, tab2 = st.tabs(["💬 Agente médico", "🔍 Explorador de embeddings"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: AGENTE
# ════════════════════════════════════════════════════════════════════════════
with tab1:

    def clean_text_query(text: str) -> str:
        import contractions
        text = contractions.fix(str(text)).lower()
        text = re.sub(r"[¿%]", " ", text)
        text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
        text = re.sub(r"\w*\d\w*", "", text)
        text = re.sub(r"[‘’“”…«»]", "", text)
        text = re.sub(r"\n", " ", text)
        return text

    def tool_corpus_search(query: str, min_overlap_override=None) -> str | None:
        """Busca en el corpus medquad por solapamiento de tokens. None si no hay match útil."""
        q_clean = clean_text_query(query)
        q_words = [w for w in q_clean.split() if w not in sw and len(w) > 1]

        if not q_words:
            return None

        min_overlap = min_overlap_override or (1 if len(q_words) == 1 else 2)

        def is_match(tokens):
            return len(set(tokens) & set(q_words)) >= min_overlap

        matches = df[df['question_tokens'].apply(is_match)]
        if matches.empty:
            return None

        resultados = []
        for _, row in matches.head(3).iterrows():
            resultados.append(f"**P:** {row['question']}\n\n**R:** {row['answer']}")
        return "\n\n---\n\n".join(resultados)

    def tool_ddg_search(query: str) -> str:
        """Búsqueda web de respaldo cuando el corpus no tiene la respuesta."""
        try:
            from ddgs import DDGS
            results = []
            with DDGS() as ddgs:
                for item in ddgs.text(query, region="es-es", safesearch="moderate", max_results=5):
                    title = (item.get("title") or "").strip()
                    href = (item.get("href") or "").strip()
                    body = (item.get("body") or "").strip().replace("\n", " ")
                    if title and href:
                        results.append(f"**{title}**\n{href}\n{body}")
            return "\n\n".join(results) if results else "Sin resultados en la web."
        except Exception as e:
            return f"Error al buscar en la web: {e}"

    def agente(query: str) -> tuple[str, str]:
        """Devuelve (respuesta, fuente) donde fuente es 'corpus' o 'web'."""
        corpus_result = tool_corpus_search(query)
        if corpus_result:
            return corpus_result, "corpus"
        return tool_ddg_search(query), "web"

    st.markdown("### Hacé una pregunta médica")
    st.caption("El agente busca primero en el corpus médico (MedQuAD). Si no encuentra nada relevante, busca en la web.")

    if "historial" not in st.session_state:
        st.session_state.historial = []

    pregunta = st.text_input("Tu pregunta", placeholder="Ej: What is glaucoma?", label_visibility="collapsed")
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        enviar = st.button("Preguntar")

    if enviar and pregunta.strip():
        with st.spinner("Buscando..."):
            respuesta, fuente = agente(pregunta)
        st.session_state.historial.append((pregunta, respuesta, fuente))

    # Mostrar historial (más reciente primero)
    for pregunta_h, respuesta_h, fuente_h in reversed(st.session_state.historial):
        st.markdown(f'<div class="chat-bubble-user">{pregunta_h}</div>', unsafe_allow_html=True)

        tag_class = "source-corpus" if fuente_h == "corpus" else "source-web"
        tag_text  = "📚 Corpus médico" if fuente_h == "corpus" else "🌐 Búsqueda web"
        st.markdown(f'<span class="source-tag {tag_class}">{tag_text}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-bubble-bot">{respuesta_h}</div>', unsafe_allow_html=True)

    if st.session_state.historial:
        if st.button("🗑️ Limpiar historial"):
            st.session_state.historial = []
            st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: EXPLORADOR DE EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Explorá el espacio semántico aprendido por Word2Vec")
    st.caption(f"Vocabulario total: {len(VOCAB):,} palabras")

    sub1, sub2, sub3 = st.tabs(["Palabras similares", "Analogías", "Visualización 3D"])

    # --- Palabras similares ---
    with sub1:
        col1, col2 = st.columns([1, 2])
        with col1:
            palabra = st.text_input("Palabra a explorar", value="glaucoma")
            top_n = st.slider("Cantidad de vecinos", 5, 30, 10)
            buscar_sim = st.button("Buscar similares")

        with col2:
            if buscar_sim:
                if palabra.lower() not in VOCAB:
                    st.warning(f"'{palabra}' no está en el vocabulario del modelo.")
                else:
                    similares = w2v_model.wv.most_similar(palabra.lower(), topn=top_n)
                    sim_df = pd.DataFrame(similares, columns=["Palabra", "Similitud"])
                    sim_df["Similitud"] = sim_df["Similitud"].round(4)
                    st.dataframe(sim_df, use_container_width=True, hide_index=True)

    # --- Analogías ---
    with sub2:
        st.caption("A es a B como C es a ___ → completá con positive=[B, C], negative=[A]")
        col1, col2, col3 = st.columns(3)
        with col1:
            palabra_a = st.text_input("A (negative)", value="men")
        with col2:
            palabra_b = st.text_input("B (positive)", value="prostate_cancer")
        with col3:
            palabra_c = st.text_input("C (positive)", value="women")

        if st.button("Resolver analogía"):
            faltantes = [w for w in [palabra_a, palabra_b, palabra_c] if w.lower() not in VOCAB]
            if faltantes:
                st.warning(f"No están en el vocabulario: {', '.join(faltantes)}")
            else:
                resultado = w2v_model.wv.most_similar(
                    positive=[palabra_b.lower(), palabra_c.lower()],
                    negative=[palabra_a.lower()]
                )
                st.success(f"**{palabra_a}** es a **{palabra_b}** como **{palabra_c}** es a **{resultado[0][0]}**")
                st.dataframe(pd.DataFrame(resultado, columns=["Candidato", "Score"]).round(4),
                             use_container_width=True, hide_index=True)

    # --- Visualización 3D ---
    with sub3:
        import plotly.graph_objs as go
        from sklearn.decomposition import PCA

        palabra_centro = st.text_input("Palabra central para visualizar", value="glaucoma", key="viz_word")
        n_vecinos = st.slider("Cantidad de vecinos a mostrar", 10, 100, 50, key="viz_n")

        if st.button("Generar visualización"):
            if palabra_centro.lower() not in VOCAB:
                st.warning(f"'{palabra_centro}' no está en el vocabulario.")
            else:
                neighbors = w2v_model.wv.most_similar(palabra_centro.lower(), topn=n_vecinos)
                words = [palabra_centro.lower()] + [n[0] for n in neighbors]
                vectors = np.array([w2v_model.wv[w] for w in words])

                pca = PCA(n_components=3)
                vectors_3d = pca.fit_transform(vectors)

                trace = go.Scatter3d(
                    x=vectors_3d[1:, 0], y=vectors_3d[1:, 1], z=vectors_3d[1:, 2],
                    mode='markers+text', text=words[1:],
                    marker=dict(size=5, color='#64748b', opacity=0.8),
                    textposition="top center", name="Vecinos"
                )
                trace_center = go.Scatter3d(
                    x=[vectors_3d[0, 0]], y=[vectors_3d[0, 1]], z=[vectors_3d[0, 2]],
                    mode='markers+text', text=[words[0]],
                    marker=dict(size=12, color='#e11d48'),
                    textposition="top center", name="Centro"
                )

                fig = go.Figure(data=[trace, trace_center])
                fig.update_layout(
                    title=f"Espacio semántico alrededor de '{palabra_centro}'",
                    scene=dict(xaxis_title="PCA 1", yaxis_title="PCA 2", zaxis_title="PCA 3"),
                    margin=dict(l=0, r=0, b=0, t=50),
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Trabajo de Laboratorio 2 · IA Generativa · Word2Vec + Agente híbrido · 2025
</div>
""", unsafe_allow_html=True)