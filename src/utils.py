import re
import unicodedata
import difflib

# ==========================================
# VARIABLES ESTÁTICAS Y DE JERGA
# ==========================================
STOPWORDS_EXTRA = {"k", "q", "dimir", "querer", "saber", "ayuda", "hacer", "decir", "poder", "necesitar"}
JERGA_POKEMON = {
    "atk": "ataque", "spe": "velocidad", "def": "defensa",
    "spa": "ataque especial", "spd": "defensa especial", "hp": "ps", "ohko": "debilitar"
}

# ==========================================
# FUNCIONES DE PREPROCESAMIENTO
# ==========================================
def proteger_etiquetas(texto):
    texto = re.sub(r"\[CATEGORÍA:\s*POKÉMON\]", "tagpokemon", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\[CATEGORÍA:\s*OBJETO\]", "tagobjeto", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\[CATEGORÍA:\s*MOVIMIENTO\]", "tagmovimiento", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\[CATEGORÍA:\s*ATAQUE\]", "tagmovimiento", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\[CATEGORÍA:\s*TIPO\]", "tagtipo", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\[CATEGORÍA:\s*NATURALEZA\]", "tagnaturaleza", texto, flags=re.IGNORECASE)
    return texto

def limpiar_texto_profundo(texto):
    texto = proteger_etiquetas(texto)
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-zA-ZáéíóúüñÑ0-9\s\-']", "", texto).lower()
    tokens = texto.split()
    return " ".join([JERGA_POKEMON.get(t, t) for t in tokens])

def quitar_tildes(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

def corregir_ortografia_mixta(tokens, vocabulario_pokemon, dic_espanol):
    tokens_corregidos = []
    vocabulario_sin_tildes = {quitar_tildes(palabra): palabra for palabra in vocabulario_pokemon}
    lista_sin_tildes = list(vocabulario_sin_tildes.keys())

    for token in tokens:
        if token.isnumeric():
            tokens_corregidos.append(token)
            continue
        if token in vocabulario_pokemon:
            tokens_corregidos.append(token)
            continue
        if token.startswith("tag"): 
            tokens_corregidos.append(token)
            continue
        if dic_espanol.spell(token):
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

def preprocesamiento_profundo(texto, vocabulario, nlp, dic_espanol):
    texto_limpio = limpiar_texto_profundo(texto)
    doc = nlp(texto_limpio)
    lemas = [token.lemma_ for token in doc if (not token.is_stop and token.lemma_ not in STOPWORDS_EXTRA) or token.text.startswith("tag")]
    return corregir_ortografia_mixta(lemas, vocabulario, dic_espanol)

def preprocesamiento_ligero(texto):
    texto = proteger_etiquetas(texto)
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-zA-ZáéíóúüñÑ0-9\s\-']", "", texto).lower()
    tokens = texto.split()
    return " ".join([JERGA_POKEMON.get(t, t) for t in tokens])

def preparar_consulta(texto_usuario, vocabulario, nlp, dic_espanol):
    return {
        "original": texto_usuario,
        "tf_idf": preprocesamiento_profundo(texto_usuario, vocabulario, nlp, dic_espanol),
        "embeddings": preprocesamiento_ligero(texto_usuario)
    }