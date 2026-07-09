import os
import re
import json
import time
import pickle
import unicodedata
import difflib
import numpy as np
import spacy
import hunspell
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import preparar_consulta

# ==============================================================================
# 1. INICIALIZACIÓN DE MOTORES LINGÜÍSTICOS (Globales)
# ==============================================================================
print("Cargando motor lingüístico spaCy...")
nlp = spacy.load("es_core_news_md")

print("Cargando diccionario oficial de la RAE (Hunspell)...")
# NOTA: Esta ruta ('/usr/share/hunspell/...') funciona en Linux/Colab. 
# Si usas Windows, tendrás que descargar los diccionarios .dic y .aff en tu carpeta del proyecto
# y cambiar esta ruta por algo como: dic_espanol = hunspell.HunSpell('es_ES.dic', 'es_ES.aff')
dic_espanol = hunspell.HunSpell('/usr/share/hunspell/es_ES.dic', '/usr/share/hunspell/es_ES.aff')


# ==============================================================================
# 2. FUNCIONES DE CARGA Y TRANSFORMACIÓN DE DATOS
# ==============================================================================
def cargar_vocabulario_real():
    vocabulario = []
    # Actualizamos las rutas a la carpeta 'datos/'
    archivos_corpus = [
        "datos/corpus_pokemon.json",
        "datos/corpus_tipos.json",
        "datos/corpus_movimientos.json",
        "datos/corpus_naturalezas.json",
        "datos/corpus_objetos.json"
    ]

    for archivo in archivos_corpus:
        try:
            with open(archivo, "r", encoding="utf-8") as f:
                datos = json.load(f)
                vocabulario.extend(datos.keys())
        except FileNotFoundError:
            print(f"No se encontró el archivo {archivo}.")

    palabras_clave = [
        "stat", "estadística", "debilidad", "fuerte", "eficaz",
        "naturaleza", "objeto", "ataque", "movimiento", "equipo",
        "pokemon", "focus", "ps"
    ]
    vocabulario.extend(palabras_clave)
    return list(set(vocabulario))

def transformar_json_a_parrafos():
    documentos_texto = []
    
    # 1. POKÉMON
    try:
        with open("datos/corpus_tipos.json", "r", encoding="utf-8") as f_tipos:
            tipos_data = json.load(f_tipos)
        with open("datos/corpus_pokemon.json", "r", encoding="utf-8") as f:
            pokemon_data = json.load(f)
            parrafos_pokemon = []
            for nombre, info in pokemon_data.items():
                tipos = ", ".join(info["tipos"])
                stats = info["estadisticas_base"]
                nombre_en = info.get("nombre_ingles", nombre).replace("-", " ")
                
                # 1. CÁLCULO MATEMÁTICO DE DEBILIDADES (Multiplicadores)
                multiplicadores = {}
                for atacante in tipos_data.keys():
                    multiplicadores[atacante] = 1.0  # Daño neutro por defecto
                    
                for t in info["tipos"]:
                    if t in tipos_data:
                        for atacante in tipos_data[t]["debil_a"]: multiplicadores[atacante] *= 2.0
                        for atacante in tipos_data[t]["resistente_a"]: multiplicadores[atacante] *= 0.5
                        for atacante in tipos_data[t]["inmune_a"]: multiplicadores[atacante] *= 0.0
                
                # 2. CLASIFICACIÓN TÁCTICA
                debil_x4 = [t for t, mult in multiplicadores.items() if mult >= 4.0]
                debil_x2 = [t for t, mult in multiplicadores.items() if mult == 2.0]
                resistencias = [t for t, mult in multiplicadores.items() if 0 < mult <= 0.5]
                inmunidades = [t for t, mult in multiplicadores.items() if mult == 0.0]
                
                txt_x4 = f"Debilidad extrema (Daño x4): {', '.join(debil_x4)}. " if debil_x4 else ""
                txt_x2 = f"Debilidades (Daño x2): {', '.join(debil_x2)}. " if debil_x2 else ""
                txt_res = f"Resistencias: {', '.join(resistencias)}. " if resistencias else ""
                txt_inm = f"Inmunidades (Daño nulo): {', '.join(inmunidades)}. " if inmunidades else ""
                
                texto_defensivo = f"{txt_x4}{txt_x2}{txt_res}{txt_inm}".strip()
                if not texto_defensivo: 
                    texto_defensivo = "Ninguna debilidad o resistencia destacable."
                
                # 3. ROL TÁCTICO (Identificar su mejor estadística)
                stat_max_nombre = max(stats, key=stats.get)
                
                # 4. CREACIÓN DEL TEXTO CON ETIQUETA SEMÁNTICA
                parrafo = (
                    f"[CATEGORÍA: POKÉMON] El Pokémon {nombre.capitalize()} (en inglés: {nombre_en}) es de tipo {tipos}. "
                    f"Perfil defensivo: {texto_defensivo} "
                    f"Estadísticas base: PS: {stats.get('hp', stats.get('ps', 0))}, "
                    f"Ataque: {stats.get('attack', stats.get('ataque', 0))}, "
                    f"Defensa: {stats.get('defense', stats.get('defensa', 0))}, "
                    f"Ataque Especial: {stats.get('special-attack', stats.get('ataque especial', 0))}, "
                    f"Defensa Especial: {stats.get('special-defense', stats.get('defensa especial', 0))}, "
                    f"Velocidad: {stats.get('speed', stats.get('velocidad', 0))}. "
                    f"Su estadística más alta es {stat_max_nombre}, lo que define su rol táctico."
                )
                parrafos_pokemon.append(parrafo)
            
            with open("datos/corpus_pokemon.txt", "w", encoding="utf-8") as f_txt:
                for p in parrafos_pokemon: f_txt.write(p + "\n")
            documentos_texto.extend(parrafos_pokemon)
    except FileNotFoundError: print("No se encontró corpus_pokemon.json")

    # 2. TIPOS
    try:
        with open("datos/corpus_tipos.json", "r", encoding="utf-8") as f:
            tipos_data = json.load(f)
            parrafos_tipos = []
            for tipo, relaciones in tipos_data.items():
                supereficaz = ", ".join(relaciones["supereficaz_contra"]) if relaciones["supereficaz_contra"] else "ninguno"
                poco_eficaz = ", ".join(relaciones["poco_eficaz_contra"]) if relaciones["poco_eficaz_contra"] else "ninguno"
                inmune_contra = ", ".join(relaciones["inmune_contra"]) if relaciones["inmune_contra"] else "ninguno"
                debil_a = ", ".join(relaciones["debil_a"]) if relaciones["debil_a"] else "ninguno"
                resistente_a = ", ".join(relaciones["resistente_a"]) if relaciones["resistente_a"] else "ninguno"
                inmune_a = ", ".join(relaciones["inmune_a"]) if relaciones["inmune_a"] else "ninguno"
                nombre_en = relaciones.get("nombre_ingles", tipo).replace("-", " ")
                
                parrafo = (
                    f"[CATEGORÍA: TIPO] Tabla de tipos {tipo.capitalize()} (conocido en inglés como tipo {nombre_en}): "
                    f"Es supereficaz (hace el doble de daño) contra: {supereficaz}. "
                    f"Es poco eficaz (hace la mitad de daño) contra: {poco_eficaz}. "
                    f"No hace daño (inmune en ataque) contra: {inmune_contra}. "
                    f"Es débil (recibe el doble de daño) ante: {debil_a}. "
                    f"Es resistente (recibe la mitad de daño) a: {resistente_a}. "
                    f"Es inmune (no recibe daño) a: {inmune_a}."
                )
                parrafos_tipos.append(parrafo)
            
            with open("datos/corpus_tipos.txt", "w", encoding="utf-8") as f_txt:
                for p in parrafos_tipos: f_txt.write(p + "\n")
            documentos_texto.extend(parrafos_tipos)
    except FileNotFoundError: print("No se encontró corpus_tipos.json")

    # 3. MOVIMIENTOS
    try:
        with open("datos/corpus_movimientos.json", "r", encoding="utf-8") as f:
            mov_data = json.load(f)
            parrafos_movs = []
            for nombre, info in mov_data.items():
                potencia = info["potencia"] if info["potencia"] else "no aplica"
                precision = f"{info['precision']}%" if info["precision"] else "no falla"
                nombre_en = info.get("nombre_ingles", nombre).replace("-", " ")
                parrafo = (
                    f"[CATEGORÍA: MOVIMIENTO] El ataque {nombre.capitalize()} (en inglés: {nombre_en}) es de tipo {info['tipo']} "
                    f"y pertenece a la categoría {info['categoria']}. "
                    f"Tiene una potencia de {potencia}, una precisión de {precision} y cuenta con {info['pp']} Puntos de Poder (PP). "
                    f"Efecto en combate: {info['efecto']}"
                )
                parrafos_movs.append(parrafo)
            with open("datos/corpus_movimientos.txt", "w", encoding="utf-8") as f_txt:
                for p in parrafos_movs: f_txt.write(p + "\n")
            documentos_texto.extend(parrafos_movs)
    except FileNotFoundError: print("No se encontró corpus_movimientos.json")

    # 4. NATURALEZAS
    try:
        with open("datos/corpus_naturalezas.json", "r", encoding="utf-8") as f:
            nat_data = json.load(f)
            parrafos_nats = []
            for nombre, info in nat_data.items():
                sube, baja = info["sube_stat"], info["baja_stat"]
                nombre_en = info.get("nombre_ingles", nombre).replace("-", " ")
                if sube == "ninguna" and baja == "ninguna":
                    parrafo = f"[CATEGORÍA: NATURALEZA] La naturaleza {nombre.capitalize()} (en inglés {nombre_en}) es neutra. No aumenta ni disminuye ninguna estadística."
                else:
                    parrafo = f"[CATEGORÍA: NATURALEZA] La naturaleza {nombre.capitalize()} (en inglés {nombre_en}) modifica las estadísticas incrementando la {sube} y reduciendo la {baja}."
                parrafos_nats.append(parrafo)
            with open("datos/corpus_naturalezas.txt", "w", encoding="utf-8") as f_txt:
                for p in parrafos_nats: f_txt.write(p + "\n")
            documentos_texto.extend(parrafos_nats)
    except FileNotFoundError: print("No se encontró corpus_naturalezas.json")

    # 5. OBJETOS
    try:
        with open("datos/corpus_objetos.json", "r", encoding="utf-8") as f:
            obj_data = json.load(f)
            parrafos_objs = []
            for nombre, info in obj_data.items():
                nombre_en = info.get("nombre_ingles", nombre).replace("-", " ")
                parrafo = f"[CATEGORÍA: OBJETO] El objeto competitivo {nombre.capitalize()} (en inglés: {nombre_en}) funciona así: {info['descripcion']}"
                parrafos_objs.append(parrafo)
            with open("datos/corpus_objetos.txt", "w", encoding="utf-8") as f_txt:
                for p in parrafos_objs: f_txt.write(p + "\n")
            documentos_texto.extend(parrafos_objs)
    except FileNotFoundError: print("No se encontró corpus_objetos.json")

    return documentos_texto

# ==============================================================================
# 3. FUNCIÓN DE TESTEO DE BÚSQUEDA
# (Nota: Toda la lógica de preprocesamiento de texto se ha movido a src/utils.py)
# ==============================================================================

def buscar_informacion_test(texto_usuario, modelo_tfidf, matriz_tfidf, modelo_embeddings, matriz_embeddings, corpus_textos, vocabulario, top_k=5, alpha=0.3):
    """
    Versión aislada del buscador solo para usarla en los tests del bloque main.
    """
    consulta_preparada = preparar_consulta(texto_usuario, vocabulario, nlp, dic_espanol)
    texto_tfidf = " ".join(consulta_preparada["tf_idf"])
    texto_embeddings = consulta_preparada["embeddings"]

    vector_tfidf_consulta = modelo_tfidf.transform([texto_tfidf])
    vector_embeddings_consulta = modelo_embeddings.encode([texto_embeddings])

    similitudes_tfidf = cosine_similarity(vector_tfidf_consulta, matriz_tfidf)[0]
    similitudes_embeddings = cosine_similarity(vector_embeddings_consulta, matriz_embeddings)[0]

    similitudes_hibridas = (alpha * similitudes_tfidf) + ((1 - alpha) * similitudes_embeddings)
    indices_top = np.argsort(similitudes_hibridas)[::-1][:top_k]

    resultados = []
    for idx in indices_top:
        resultados.append({
            "texto": corpus_textos[idx],
            "puntuacion_total": similitudes_hibridas[idx],
            "score_tfidf": similitudes_tfidf[idx],
            "score_embed": similitudes_embeddings[idx]
        })
    return resultados

# ==============================================================================
# 4. BLOQUE PRINCIPAL (Se ejecuta con `python src/buscadores.py`)
# ==============================================================================
if __name__ == "__main__":
    
    # 1. Cargar el vocabulario y guardarlo
    print("\n[FASE 2] Inicializando Vocabulario...")
    vocabulario_oficial = cargar_vocabulario_real()
    print(f"Vocabulario cargado con {len(vocabulario_oficial)} términos.")
    with open('datos/vocabulario_oficial.pkl', 'wb') as f:
        pickle.dump(vocabulario_oficial, f)

    # 2. Generar el corpus en texto plano
    print("\n[FASE 2] Transformando JSONs a texto plano...")
    corpus_parrafos = transformar_json_a_parrafos()
    print(f"Corpus unificado creado en memoria con {len(corpus_parrafos)} fragmentos.")

    # 3. Preprocesar para modelos
    print("\n[FASE 3] Entrenando Motores de Búsqueda...")
    tiempo_inicio = time.time()
    
    print("1/4 Procesando textos para TF-IDF (Lematización profunda)...")
    corpus_tfidf = []
    for i, texto in enumerate(corpus_parrafos):
        lemas = preprocesamiento_profundo(texto, vocabulario_oficial)
        corpus_tfidf.append(" ".join(lemas))

    print("2/4 Procesando textos para Embeddings (Limpieza ligera)...")
    corpus_embeddings = [preprocesamiento_ligero(texto) for texto in corpus_parrafos]

    print("3/4 Entrenando modelo TF-IDF (Búsqueda Dispersa)...")
    modelo_tfidf = TfidfVectorizer()
    matriz_tfidf = modelo_tfidf.fit_transform(corpus_tfidf)

    print("4/4 Calculando Embeddings (Búsqueda Densa)...")
    modelo_embeddings = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    matriz_embeddings = modelo_embeddings.encode(corpus_embeddings, show_progress_bar=True)

    # 4. Guardar archivos .pkl finales en la carpeta datos
    print("\nGuardando los modelos en la carpeta 'datos/'...")
    with open("datos/corpus_textos_originales.pkl", "wb") as f:
        pickle.dump(corpus_parrafos, f)
        
    with open("datos/buscador_tfidf.pkl", "wb") as f:
        pickle.dump({"modelo": modelo_tfidf, "matriz": matriz_tfidf}, f)
        
    with open("datos/buscador_embeddings.pkl", "wb") as f:
        pickle.dump({"matriz": matriz_embeddings}, f)

    print(f"Buscadores listos en {time.time() - tiempo_inicio:.2f} segundos.\n")
    
    # 5. Ejecutar Test Automático
    print("=== INICIANDO TEST AUTOMÁTICO DE LOS MODELOS GUARDADOS ===")
    preguntas_trampa = [
        "¿Cuánto base attack tiene garchomp?",
        "Dime un ataque de tipo planta que siempre haga golpe crítico.",
        "k hace la tabla duende para arceus??"
    ]
    for i, pregunta in enumerate(preguntas_trampa):
        print(f"\nPRUEBA {i+1}: '{pregunta}'")
        respuestas = buscar_informacion_test(
            pregunta, modelo_tfidf, matriz_tfidf, modelo_embeddings, matriz_embeddings, 
            corpus_parrafos, vocabulario_oficial, top_k=2, alpha=0.3
        )
        for j, res in enumerate(respuestas):
            print(f"[Top {j+1}] Relevancia: {res['puntuacion_total']:.2f}")
            print(f"{res['texto']}")