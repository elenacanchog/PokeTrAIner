import os
import re
import unicodedata
import hunspell
import difflib
import pickle
import json
import numpy as np
import spacy
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Cargar variables de entorno locales
load_dotenv()

# ==========================================
# 1. VARIABLES ESTÁTICAS Y DE JERGA
# ==========================================
STOPWORDS_EXTRA = {"k", "q", "dimir", "querer", "saber", "ayuda", "hacer", "decir", "poder", "necesitar"}
JERGA_POKEMON = {
    "atk": "ataque", "spe": "velocidad", "def": "defensa",
    "spa": "ataque especial", "spd": "defensa especial", "hp": "ps", "ohko": "debilitar"
}

# ==========================================
# 2. CARGA DEL ENTORNO (Caché de Streamlit)
# ==========================================
@st.cache_resource
def cargar_entorno_rag():
    st.info("Inicializando sistema experto... (Cargando modelos)")

    with open("datos/vocabulario_oficial.pkl", "rb") as f:
        vocabulario_oficial = pickle.load(f)
    with open("datos/corpus_textos_originales.pkl", "rb") as f:
        corpus_textos = pickle.load(f)
    with open("datos/buscador_tfidf.pkl", "rb") as f:
        datos_tfidf = pickle.load(f)
        modelo_tfidf = datos_tfidf["modelo"]
        matriz_tfidf_corpus = datos_tfidf["matriz"]
    with open("datos/buscador_embeddings.pkl", "rb") as f:
        datos_embeddings = pickle.load(f)
        matriz_embeddings_corpus = datos_embeddings["matriz"]

    nlp = spacy.load("es_core_news_md")
    dic_espanol = hunspell.HunSpell('/usr/share/hunspell/es_ES.dic', '/usr/share/hunspell/es_ES.aff')
    modelo_embeddings = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        st.error("ERROR: no se ha detectado el token de Hugging Face en el entorno (.env).")
        st.stop()

    modelo_id = "Qwen/Qwen2.5-7B-Instruct"
    llm_generador = InferenceClient(model=modelo_id, token=hf_token)

    return (vocabulario_oficial, corpus_textos, modelo_tfidf, matriz_tfidf_corpus,
            matriz_embeddings_corpus, nlp, dic_espanol, modelo_embeddings, llm_generador)

# Desempaquetamos para tenerlas como variables globales en este archivo (igual que en Colab)
try:
    (vocabulario_oficial, corpus_textos, modelo_tfidf, matriz_tfidf_corpus,
     matriz_embeddings_corpus, nlp, dic_espanol, modelo_embeddings, llm_generador) = cargar_entorno_rag()
except Exception as e:
    st.error(f"ERROR al cargar el entorno: {e}")
    st.stop()

# ==========================================
# 3. FUNCIONES DE PREPROCESAMIENTO Y BÚSQUEDA
# ==========================================
def limpiar_texto_profundo(texto):
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-zA-ZáéíóúüñÑ0-9\s\-']", "", texto).lower()
    tokens = texto.split()
    tokens_sin_jerga = [JERGA_POKEMON.get(t, t) for t in tokens]
    return " ".join(tokens_sin_jerga)

def quitar_tildes(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

def corregir_ortografia_mixta(tokens, vocabulario_pokemon):
    tokens_corregidos = []
    vocabulario_sin_tildes = {quitar_tildes(palabra): palabra for palabra in vocabulario_pokemon}
    lista_sin_tildes = list(vocabulario_sin_tildes.keys())

    for token in tokens:
        if token.isnumeric() or token in vocabulario_pokemon or dic_espanol.spell(token):
            tokens_corregidos.append(token)
            continue

        token_sin_tilde = quitar_tildes(token)
        coincidencia_poke = difflib.get_close_matches(token_sin_tilde, lista_sin_tildes, n=1, cutoff=0.75)

        if coincidencia_poke:
            tokens_corregidos.append(vocabulario_sin_tildes[coincidencia_poke[0]])
        else:
            sugerencias_es = dic_espanol.suggest(token)
            tokens_corregidos.append(sugerencias_es[0].lower() if sugerencias_es else token)
    return tokens_corregidos

def preprocesamiento_profundo(texto, vocabulario):
    texto_limpio = limpiar_texto_profundo(texto)
    doc = nlp(texto_limpio)
    lemas = [token.lemma_ for token in doc if not token.is_stop and token.lemma_ not in STOPWORDS_EXTRA]
    return corregir_ortografia_mixta(lemas, vocabulario)

def preprocesamiento_ligero(texto):
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-zA-ZáéíóúüñÑ0-9\s\-']", "", texto).lower()
    tokens = texto.split()
    tokens_sin_jerga = [JERGA_POKEMON.get(t, t) for t in tokens]
    return " ".join(tokens_sin_jerga)

def preparar_consulta(texto_usuario, vocabulario):
    return {
        "original": texto_usuario,
        "tf_idf": preprocesamiento_profundo(texto_usuario, vocabulario),
        "embeddings": preprocesamiento_ligero(texto_usuario)
    }

def buscar_informacion(texto_usuario, top_k=5, alpha=0.5):
    consulta_preparada = preparar_consulta(texto_usuario, vocabulario_oficial)
    texto_tfidf = " ".join(consulta_preparada["tf_idf"])
    texto_embeddings = consulta_preparada["embeddings"]

    vector_tfidf_consulta = modelo_tfidf.transform([texto_tfidf])
    vector_embeddings_consulta = modelo_embeddings.encode([texto_embeddings])

    similitudes_tfidf = cosine_similarity(vector_tfidf_consulta, matriz_tfidf_corpus)[0]
    similitudes_embeddings = cosine_similarity(vector_embeddings_consulta, matriz_embeddings_corpus)[0]

    similitudes_hibridas = (alpha * similitudes_tfidf) + ((1 - alpha) * similitudes_embeddings)
    indices_top = np.argsort(similitudes_hibridas)[::-1][:top_k]

    resultados = []
    for idx in indices_top:
        resultados.append({"texto": corpus_textos[idx]})
    return resultados

def construir_prompt(pregunta, contexto):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Eres 'PokéTrAIner', el mejor profesor de combates Pokémon del mundo.
Tienes permiso absoluto para hablar sobre mecánicas de videojuegos, objetos, ataques y estrategias.

TU MISIÓN:
Responder a la pregunta del usuario de forma clara, entusiasta y útil, basándote principalmente en la información del [CONTEXTO RECOPILADO].

REGLAS:
1. ADÁPTATE A LA PREGUNTA:
   - Si preguntan por un OBJETO, ATAQUE o TIPO, lee el [CONTEXTO RECOPILADO] y explica exactamente lo que dice ahí.
   - Si piden una ESTRATEGIA, cruza las estadísticas del [CONTEXTO RECOPILADO] con tu conocimiento oficial.
2. PROHIBIDO INVENTAR DATOS (CERO ALUCINACIONES):
   - Si recomiendas una Naturaleza, asegúrate de recordar exactamente qué sube y qué baja. Si no estás 100% seguro, no lo menciones.
   - Si el usuario te pregunta "quién usa un ataque" o "dónde se consigue", y esa información NO está en el [CONTEXTO RECOPILADO], debes decir explícitamente: "No dispongo de esa información en mis registros actuales." No intentes adivinar juegos ni Pokémon.
3. SÉ HONESTO: Si te preguntan por crossovers (como Zelda) o cosas ajenas a Pokémon competitivo, indica que solo puedes hablar de Pokémon.

[CONTEXTO RECOPILADO]
{contexto}
<|eot_id|><|start_header_id|>user<|end_header_id|>

PREGUNTA DEL USUARIO: {pregunta}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt