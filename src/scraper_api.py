import requests
import json
import time
import os

# Asegurarnos de que la carpeta de destino existe
os.makedirs("datos", exist_ok=True)

# Pequeño diccionario estático para tener todos los datos posibles en español
TRADUCCION_TIPOS = {
    "normal": "normal", "fire": "fuego", "water": "agua", "electric": "eléctrico",
    "grass": "planta", "ice": "hielo", "fighting": "lucha", "poison": "veneno",
    "ground": "tierra", "flying": "volador", "psychic": "psíquico", "bug": "bicho",
    "rock": "roca", "ghost": "fantasma", "dragon": "dragón", "dark": "siniestro",
    "steel": "acero", "fairy": "hada"
}

TRADUCCION_CATEGORIA = {
    "status" : "estado", "physical" : "físico", "special" : "especial", "desconocida" : "desconocida"
}

TRADUCCION_STATS = {
    "hp": "ps", "attack": "ataque", "defense": "defensa",
    "special-attack": "ataque especial", "special-defense": "defensa especial", "speed": "velocidad"
}

TRADUCCION_PARADOJA = {
    # FORMAS DEL PASADO
    "great-tusk": "Colmilargo", "scream-tail": "Colagrito", "brute-bonnet": "Furioseta",
    "flutter-mane": "Melenaleteo", "slither-wing": "Reptalada", "sandy-shocks": "Pelarena",
    "roaring-moon": "Bramaluna", "walking-wake": "Ondulagua", "gouging-fire": "Flamariete", "raging-bolt": "Electrofuria",
    # FORMAS DEL FUTURO
    "iron-treads": "Ferrodada", "iron-bundle": "Ferrosaco", "iron-hands": "Ferropalmas",
    "iron-jugulis": "Ferrocuello", "iron-moth": "Ferropolilla", "iron-thorns": "Ferropúas",
    "iron-valiant": "Ferropaladín", "iron-leaves": "Ferroverdor", "iron-crown": "Ferrotesta", "iron-boulder": "Ferromole"
}

TRADUCCION_MANUAL_OBJETOS = {
    "blank-plate": {"nombre": "tabla en blanco", "descripcion": "Una tabla misteriosa que cambia el tipo de Arceus a tipo Normal y potencia sus movimientos de este tipo."},
    "legend-plate": {"nombre": "tabla leyenda", "descripcion": "Una tabla legendaria que cambia el tipo de Arceus en combate para que siempre sea supereficaz contra el objetivo."},
    "booster-energy": {"nombre": "energía potenciadora", "descripcion": "Activa automáticamente la habilidad Paleosíntesis o Carga Cuatridimensional del portador aumentando su estadística más alta."},
    "ability-shield": {"nombre": "escudo habilidad", "descripcion": "Un escudo que protege la habilidad del portador para que no pueda ser cambiada, copiada o anulada por otros Pokémon."},
    "clear-amulet": {"nombre": "amuleto puro", "descripcion": "Un amuleto transparente que evita que las características del portador sean bajadas por movimientos o habilidades de los oponentes."},
    "mirror-herb": {"nombre": "hierba copia", "descripcion": "Una hierba que permite al portador copiar los aumentos de estadísticas que se aplique un rival una sola vez en combate."},
    "punching-glove": {"nombre": "guante de boxeo", "descripcion": "Aumenta la potencia de los movimientos basados en puñetazos y evita el contacto directo con el objetivo (protegiendo de habilidades como Piel Tosca)."},
    "covert-cloak": {"nombre": "capa furtiva", "descripcion": "Una capa que oculta al portador, protegiéndolo completamente de los efectos secundarios de los movimientos enemigos (como retroceso, parálisis o quemaduras)."},
    "loaded-dice": {"nombre": "dado trucado", "descripcion": "Un dado modificado que asegura que los movimientos que golpean varias veces seguidas impacten siempre el mayor número de veces posible."},
    "fairy-feather": {"nombre": "pluma feérica", "descripcion": "Una pluma imbuida de magia que potencia los movimientos de tipo Hada del portador."},
    "wellspring-mask": {"nombre": "máscara fuente", "descripcion": "Una máscara misteriosa que permite a Ogerpon adoptar su Forma Fuente (tipo Agua) y potencia sus ataques de ese tipo."},
    "hearthflame-mask": {"nombre": "máscara horno", "descripcion": "Una máscara misteriosa que permite a Ogerpon adoptar su Forma Horno (tipo Fuego) y potencia sus ataques de ese tipo."},
    "cornerstone-mask": {"nombre": "máscara cimiento", "descripcion": "Una máscara misteriosa que permite a Ogerpon adoptar su Forma Cimiento (tipo Roca) y potencia sus ataques de ese tipo."}
}

TRADUCCION_MANUAL_MOVIMIENTOS = {
    "tera-blast": {"nombre": "teraexplosión", "descripcion": "Si el usuario se ha teracristalizado, este movimiento desata el poder de su Teratipo, atacando por el lado físico o especial según cuál sea más alto."},
    "psicorruido": {"nombre": "psicorruido", "descripcion": "Golpea con ondas psíquicas desagradables que impiden al Pokémon objetivo recuperar PS mediante movimientos o habilidades durante varios turnos."},
    "hidrovapor": {"nombre": "hidrovapor", "descripcion": "Dispara un chorro de agua hirviendo a altísima presión. Su potencia aumenta un 50% si hay clima de sol abrasador en lugar de disminuir."},
    "psicohojas": {"nombre": "psicohojas", "descripcion": "Ataca con una espada de energía psíquica. Su potencia aumenta drásticamente si se encuentra activo el campo eléctrico."},
    "gigaton-hammer": {"nombre": "martillo colosal", "descripcion": "Golpea brutalmente al rival con un martillo descomunal de acero. Debido a su inmenso peso, no puede usarse dos veces seguidas."},
    "make-it-rain": {"nombre": "fiebre dorada", "descripcion": "Lanza una lluvia masiva de monedas de oro brillantes. Reduce el Ataque Especial del usuario tras golpear a los objetivos."},
    "salt-cure": {"nombre": "salazón", "descripcion": "Aplica sal al objetivo causándole daño continuo cada turno. Si el rival es de tipo Agua o Acero, el daño por turno se duplica."},
    "population-bomb": {"nombre": "proliferación", "descripcion": "El usuario ataca en grupo golpeando consecutivamente de 1 a 10 veces seguidas de forma implacable."},
    "rage-fist": {"nombre": "puño furia", "descripcion": "Un golpe cargado de ira. Su potencia base aumenta de forma permanente en 50 puntos cada vez que el usuario recibe un ataque en el combate."},
    "revival-blessing": {"nombre": "plegaria vital", "descripcion": "Una oración sagrada que permite revivir a un Pokémon debilitado de tu equipo, restaurando la mitad de sus PS máximos."},
    "shed-tail": {"nombre": "autotomía", "descripcion": "El usuario crea un sustituto usando la mitad de sus PS máximos y se intercambia inmediatamente por otro Pokémon del equipo."},
    "chilly-reception": {"nombre": "fría acogida", "descripcion": "El usuario cuenta un chiste malo para provocar una ventisca de granizo o nieve y se retira al instante regresando a su Pokéball."},
    "flower-trick": {"nombre": "truco floral", "descripcion": "Lanza una bomba de flores que nunca falla el blanco y siempre resulta en un golpe crítico garantizado."},
    "torch-song": {"nombre": "canto ardiente", "descripcion": "El usuario canta desatando llamas intensas. Cada vez que golpea, incrementa el Ataque Especial del usuario en un nivel."},
    "aqua-step": {"nombre": "danza acuática", "descripcion": "Un ataque fluido y danzante que golpea con fuerza de tipo agua y aumenta la Velocidad del usuario en un nivel tras impactar."}
}

CATEGORIAS_OBJETOS = [
    "held-items", "berries", "mega-stones", "plates", "jewels",
    "species-specific", "type-enhancement", "choice", "bad-held-items"
]

def obtener_nombres_pokemon(limite):
    print(f"Obteniendo la lista de los {limite} Pokémon")
    url = f"https://pokeapi.co/api/v2/pokemon?limit={limite}"
    respuesta = requests.get(url)
    if respuesta.status_code == 200:
        return [poke["name"] for poke in respuesta.json()["results"]]
    return []

def crear_corpus_pokemon(lista_pokemon):
    corpus = {}
    base_url = "https://pokeapi.co/api/v2/pokemon/"
    print(f"\nIniciando la descarga de datos de {len(lista_pokemon)} Pokémon")

    for nombre in lista_pokemon:
        respuesta = requests.get(f"{base_url}{nombre}")
        if respuesta.status_code == 200:
            datos_api = respuesta.json()
            tipos_es = [TRADUCCION_TIPOS.get(tipo['type']['name'], tipo['type']['name']) for tipo in datos_api['types']]
            stats = {stat['stat']['name']: stat['base_stat'] for stat in datos_api['stats']}
            nombre_api = nombre.lower()
            nombre_final = TRADUCCION_PARADOJA.get(nombre_api, nombre_api)

            corpus[nombre_final.lower()] = {
                "tipos": tipos_es,
                "estadisticas_base": stats,
                "nombre_ingles": nombre.lower()
            }
        time.sleep(0.5)

    with open("datos/corpus_pokemon.json", "w", encoding="utf-8") as archivo:
        json.dump(corpus, archivo, indent=4, ensure_ascii=False)
    print("Corpus de Pokémon guardado.")
    return corpus

def crear_corpus_tipos():
    corpus_tipos = {}
    base_url = "https://pokeapi.co/api/v2/type/"
    lista_tipos = list(TRADUCCION_TIPOS.keys())
    print("\nIniciando la descarga de los 18 Tipos...")

    for tipo in lista_tipos:
        respuesta = requests.get(f"{base_url}{tipo}")
        if respuesta.status_code == 200:
            datos_api = respuesta.json()
            relaciones = datos_api['damage_relations']
            nombre_es = TRADUCCION_TIPOS[tipo]

            corpus_tipos[nombre_es] = {
                "supereficaz_contra": [TRADUCCION_TIPOS.get(t['name'], t['name']) for t in relaciones['double_damage_to']],
                "poco_eficaz_contra": [TRADUCCION_TIPOS.get(t['name'], t['name']) for t in relaciones['half_damage_to']],
                "inmune_contra": [TRADUCCION_TIPOS.get(t['name'], t['name']) for t in relaciones['no_damage_to']],
                "debil_a": [TRADUCCION_TIPOS.get(t['name'], t['name']) for t in relaciones['double_damage_from']],
                "resistente_a": [TRADUCCION_TIPOS.get(t['name'], t['name']) for t in relaciones['half_damage_from']],
                "inmune_a": [TRADUCCION_TIPOS.get(t['name'], t['name']) for t in relaciones['no_damage_from']],
                "nombre_ingles": tipo
            }
        time.sleep(0.5)

    with open("datos/corpus_tipos.json", "w", encoding="utf-8") as archivo:
        json.dump(corpus_tipos, archivo, indent=4, ensure_ascii=False)
    print("Corpus de tipos guardado.")
    return corpus_tipos

def obtener_lista_recurso(endpoint, limite):
    print(f"Obteniendo lista de {endpoint}...")
    url = f"https://pokeapi.co/api/v2/{endpoint}?limit={limite}"
    respuesta = requests.get(url)
    if respuesta.status_code == 200:
        return [item["name"] for item in respuesta.json()["results"]]
    return []

def crear_corpus_movimientos(lista_movimientos):
    corpus = {}
    base_url = "https://pokeapi.co/api/v2/move/"
    print(f"\nDescargando datos de {len(lista_movimientos)} movimientos...")

    for mov in lista_movimientos:
        respuesta = requests.get(f"{base_url}{mov}")
        if respuesta.status_code == 200:
            datos = respuesta.json()
            id_movimiento = datos['id']

            if (622 <= id_movimiento <= 658) or (757 <= id_movimiento <= 774) or (datos['type']['name'] == 'shadow'):
                continue

            if mov in TRADUCCION_MANUAL_MOVIMIENTOS:
                nombre_es = TRADUCCION_MANUAL_MOVIMIENTOS[mov]["nombre"]
                descripcion_es = TRADUCCION_MANUAL_MOVIMIENTOS[mov]["descripcion"]
            else:
                nombre_es = next((n['name'].lower() for n in datos['names'] if n['language']['name'] == 'es'), mov)
                entradas_es = [f['flavor_text'] for f in datos['flavor_text_entries'] if f['language']['name'] == 'es']
                descripcion_es = entradas_es[-1].replace("\n", " ") if entradas_es else "Sin descripción."

            if descripcion_es == "Sin descripción." and mov in TRADUCCION_MANUAL_MOVIMIENTOS:
                descripcion_es = TRADUCCION_MANUAL_MOVIMIENTOS[mov]["descripcion"]
            elif descripcion_es == "Sin descripción.":
                descripcion_es = f"Movimiento de tipo {datos['type']['name']} que inflige daño en combate."

            corpus[nombre_es] = {
                "tipo": TRADUCCION_TIPOS.get(datos['type']['name'], datos['type']['name']),
                "categoria": TRADUCCION_CATEGORIA.get(datos['damage_class']['name'] if datos['damage_class'] else "desconocida", "desconocida"),
                "potencia": datos['power'],
                "precision": datos['accuracy'],
                "pp": datos['pp'],
                "efecto": descripcion_es,
                "nombre_ingles": mov
            }
        time.sleep(0.1)

    with open("datos/corpus_movimientos.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=4, ensure_ascii=False)
    print("Corpus de movimientos guardado.")
    return corpus

def crear_corpus_naturalezas(lista_naturalezas):
    corpus = {}
    base_url = "https://pokeapi.co/api/v2/nature/"
    print(f"\nDescargando datos de {len(lista_naturalezas)} naturalezas...")

    for nat in lista_naturalezas:
        respuesta = requests.get(f"{base_url}{nat}")
        if respuesta.status_code == 200:
            datos = respuesta.json()
            nombre_es = next((n['name'].lower() for n in datos['names'] if n['language']['name'] == 'es'), nat)
            sube = datos['increased_stat']['name'] if datos['increased_stat'] else None
            baja = datos['decreased_stat']['name'] if datos['decreased_stat'] else None

            corpus[nombre_es] = {
                "sube_stat": TRADUCCION_STATS.get(sube, "ninguna"),
                "baja_stat": TRADUCCION_STATS.get(baja, "ninguna"),
                "nombre_ingles": nat
            }
        time.sleep(0.3)

    with open("datos/corpus_naturalezas.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=4, ensure_ascii=False)
    print("Corpus de naturalezas guardado.")
    return corpus

def crear_corpus_objetos(lista_objetos):
    corpus = {}
    base_url = "https://pokeapi.co/api/v2/item/"
    print(f"\nDescargando datos de {len(lista_objetos)} objetos...")

    for obj in lista_objetos:
        respuesta = requests.get(f"{base_url}{obj}")
        if respuesta.status_code == 200:
            datos = respuesta.json()
            categoria_actual = datos['category']['name']

            if categoria_actual not in CATEGORIAS_OBJETOS:
                continue

            if obj in TRADUCCION_MANUAL_OBJETOS:
                nombre_es = TRADUCCION_MANUAL_OBJETOS[obj]["nombre"]
                descripcion_es = TRADUCCION_MANUAL_OBJETOS[obj]["descripcion"]
            else:
                nombre_es = next((n['name'].lower() for n in datos['names'] if n['language']['name'] == 'es'), obj)
                entradas_es = [f['text'] for f in datos['flavor_text_entries'] if f['language']['name'] == 'es']
                descripcion_es = entradas_es[-1].replace("\n", " ") if entradas_es else "Sin descripción."

            corpus[nombre_es] = {
                "descripcion": descripcion_es,
                "nombre_ingles": obj
            }
        time.sleep(0.1)

    with open("datos/corpus_objetos.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=4, ensure_ascii=False)
    print("Corpus de objetos guardado.")
    return corpus

if __name__ == "__main__":
    print("=== INICIANDO DESCARGA TOTAL DEL CORPUS ===")
    
    # Hemos descomentado estas líneas para que al ejecutar el archivo, descargue todo.
    lista_nombres = obtener_nombres_pokemon(1025)
    mi_corpus_poke = crear_corpus_pokemon(lista_nombres)
    
    mi_corpus_tipos = crear_corpus_tipos()
    
    lista_movimientos = obtener_lista_recurso("move", 1000)
    mi_corpus_mov = crear_corpus_movimientos(lista_movimientos)
    
    lista_naturalezas = obtener_lista_recurso("nature", 25)
    mi_corpus_nat = crear_corpus_naturalezas(lista_naturalezas)
    
    lista_objetos = obtener_lista_recurso("item", 2176)
    mi_corpus_obj = crear_corpus_objetos(lista_objetos)
    
    print("\n✅ Todos los corpus se han generado correctamente en la carpeta 'datos/'.")