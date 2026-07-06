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
# --- IMPORTS DE LANGCHAIN ---
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from langchain_classic.tools.render import render_text_description
from langchain_classic.agents import AgentExecutor

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

    # hf_token = os.environ.get("HF_TOKEN")
    # if not hf_token:
    #     st.error("ERROR: no se ha detectado el token de Hugging Face en el entorno (.env).")
    #     st.stop()
    
    # Cargamos nueva API Key de Groq
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("ERROR: no se ha detectado el GROQ_API_KEY en el entorno (.env).")
        st.stop()

    modelo_id = "llama-3.3-70b-versatile"
    #Otros modelos usados:
    # Qwen/Qwen2.5-7B-Instruct
    # mistralai/Mixtral-8x7B-Instruct-v0.1
    # Qwen/Qwen2.5-72B-Instruct

    # Usamos el conector oficial de LangChain
    # llm_generador = ChatOpenAI(
    #     model=modelo_id,                                     
    #     base_url="https://api-inference.huggingface.co/v1/", # El túnel
    #     api_key=hf_token,
    #     temperature=0.1,
    #     max_retries=3,
    #     timeout=180.0,
    #     max_tokens=1024,
    #     top_p=0.9
    # )

    llm_generador = ChatGroq(
        model_name=modelo_id,
        api_key=groq_api_key,
        temperature=0.1,
        max_tokens=1024,
        max_retries=3,
        timeout=180.0,
        model_kwargs={"top_p": 0.9} # Empaquetamos top_p aquí para limpiar el warning
    )

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
    # La caja del sistema: SOLO TUS REGLAS PURAS
    instrucciones_sistema = """Eres 'PokéTrAIner', el mejor profesor de combates Pokémon del mundo.
Tienes permiso absoluto para hablar sobre mecánicas de videojuegos, objetos, ataques y estrategias.

TU MISIÓN:
Responder a la pregunta del usuario de forma clara, entusiasta y útil, basándote principalmente en la información del [CONTEXTO RECOPILADO].

REGLAS:
1. ADÁPTATE A LA PREGUNTA:
   - Si preguntan por un OBJETO, ATAQUE o TIPO, lee el [CONTEXTO RECOPILADO] y explica exactamente lo que dice ahí.
   - Si piden una ESTRATEGIA, cruza las estadísticas del [CONTEXTO RECOPILADO] con tu conocimiento oficial.
2. PROHIBIDO INVENTAR DATOS:
   - Si recomiendas una Naturaleza, asegúrate de recordar exactamente qué sube y qué baja. Si no estás 100% seguro, no lo menciones.
   - Si el usuario te pregunta "quién usa un ataque" o "dónde se consigue",
   y esa información NO está en el [CONTEXTO RECOPILADO], debes decir explícitamente:
   "No dispongo de esa información en mis registros actuales." No intentes adivinar juegos ni Pokémon.
3. SÉ HONESTO: si te preguntan por crossovers (como Zelda) o cosas ajenas a Pokémon competitivo, indica que solo puedes hablar de Pokémon.
"""

    # La caja del usuario: EL CONTEXTO Y LA PREGUNTA
    mensaje_usuario = f"""[CONTEXTO RECOPILADO]
{contexto}

PREGUNTA DEL USUARIO: {pregunta}"""

    # Las dos cajas separadas y ordenadas
    mensajes = [
        {"role": "system", "content": instrucciones_sistema},
        {"role": "user", "content": mensaje_usuario}
    ]

    return mensajes



# ==========================================
# HERRAMIENTAS DEL AGENTE (LANGCHAIN)
# ==========================================

@tool
def herramienta_pokedex(consulta: str) -> str:
    """
    Busca en la base de datos oficial de Pokémon (Pokédex).
    Úsala SIEMPRE que necesites buscar estadísticas, debilidades, tipos, 
    efectos de naturalezas, ataques u objetos para responder al usuario.
    """
    # 1. Usamos TU función original intacta (con top_k=3 para no saturar)
    documentos_recuperados = buscar_informacion(consulta, top_k=5, alpha=0.6)
    
    # 2. Si el buscador no encuentra nada, avisamos al Agente
    if not documentos_recuperados:
        return "No se encontró información en la Pokédex sobre esa consulta."
    
    # 3. Formateamos la salida en un solo texto limpio para que el Agente lo lea
    contexto_unido = "\n- ".join([doc['texto'] for doc in documentos_recuperados])
    return contexto_unido


# ==========================================
# 7. INICIALIZACIÓN DEL AGENTE REACT
# ==========================================

# 1. Metemos tu herramienta en la lista de herramientas disponibles para el agente
herramientas_agente = [herramienta_pokedex]

# 2. Definimos el Prompt del sistema con el formato estricto ReAct
plantilla_react = """Eres 'PokéTrAIner', el mejor profesor de combates Pokémon del mundo.
Tienes permiso absoluto para hablar sobre mecánicas de videojuegos, objetos, ataques y estrategias.

REGLAS:
1. ADÁPTATE A LA PREGUNTA: Usa tus herramientas para buscar datos exactos y construir estrategias basadas en ellos.
2. PROHIBIDO INVENTAR DATOS: Si recomiendas algo, asegúrate de que sea real. Si no encuentras información sobre algo en tus herramientas, di que no dispones de esos registros.
3. SÉ HONESTO: Solo puedes hablar de Pokémon competitivo.
4. BÚSQUEDA ROBÓTICA OBLIGATORIA: Tu buscador es una máquina de palabras clave, NO un humano. 
- En tu "Action Input", usa SIEMPRE una sola etiqueta de categoría y MÁXIMO 1 o 2 palabras clave. 
- PROHIBIDO usar frases conversacionales, conjunciones ("y", "que", "para") o verbos ("aumenta", "sirve").
- PROHIBIDO buscar dos cosas a la vez. Si necesitas ver a Pikachu y luego las naturalezas, haz DOS turnos de búsqueda separados.
- EJEMPLOS CORRECTOS: "[CATEGORÍA: POKÉMON] Pikachu" o "[CATEGORÍA: NATURALEZA] velocidad" o "[CATEGORÍA: OBJETO] fuego".
- EJEMPLOS INCORRECTOS: "naturaleza que aumente la velocidad de Pikachu" o "objetos para ganar a Charizard".

Para responder al usuario, DEBES seguir estrictamente este formato de pensamiento paso a paso:

Thought: Siempre debes pensar qué necesitas hacer para responder. ¿Necesitas buscar en la Pokédex?
Action: La herramienta que vas a usar (debe ser exactamente: herramienta_pokedex).
Action Input: El texto exacto que vas a buscar (¡Aplica la regla 4 estrictamente!).
Observation: El resultado que te devuelve la herramienta (Esto aparecerá automáticamente, no lo inventes).
... (Este ciclo de Thought/Action/Action Input/Observation se puede repetir si necesitas buscar más cosas)

Thought: Ya sé la respuesta final tras analizar los datos recopilados.
Final Answer: La respuesta final y detallada que leerá el usuario en español de forma clara y entusiasta.

Herramientas disponibles:
{tools}

Nombres de las herramientas:
{tool_names}

PREGUNTA DEL USUARIO: {input}

Empieza a pensar ahora:
{agent_scratchpad}"""

prompt_agente = PromptTemplate.from_template(plantilla_react)

# 3. Rellenamos el prompt con la descripción automática de tu herramienta
prompt_agente = prompt_agente.partial(
    tools=render_text_description(herramientas_agente),
    tool_names=", ".join([t.name for t in herramientas_agente]),
)

# 4. EL FRENO CRÍTICO: Le decimos al LLM que se detenga a esperar en cuanto escriba "Observation:"
# Así evitamos que alucine los datos del buscador.
llm_con_stop = llm_generador.bind(stop=["\nObservation:", "Observation:"])

# 5. CONSTRUIMOS EL AGENTE PASO A PASO (LCEL - LangChain Expression Language)
agente_pokedex = (
    {
        # Recoge la pregunta del usuario
        "input": lambda x: x["input"],
        # Recoge el historial de "Thought/Action/Observation" que va acumulando
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt_agente
    | llm_con_stop
    | ReActSingleInputOutputParser()
)

# 6. Creamos el ejecutor final (El bucle While que repite el proceso hasta llegar a Final Answer)
ejecutor_agente = AgentExecutor(
    agent=agente_pokedex, 
    tools=herramientas_agente, 
    verbose=True, 
    handle_parsing_errors=True
)