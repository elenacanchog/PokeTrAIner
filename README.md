# 🏆 PokéTrAIner: Tu Asistente Táctico de Pokémon Competitivo

PokéTrAIner es un asistente de Inteligencia Artificial especializado en Pokémon competitivo, construido con arquitectura **RAG (Retrieval-Augmented Generation)**. Su objetivo es ayudar a los entrenadores a planificar estrategias cruzando datos reales de estadísticas, debilidades, naturalezas, movimientos y objetos.

A diferencia de un LLM tradicional que puede "alucinar" o inventar reglas de los videojuegos, PokéTrAIner utiliza un motor de búsqueda híbrido para consultar su propia Pokédex (Base de Conocimiento) antes de emitir cualquier respuesta.

---

## ⚙️ ¿Cómo funciona bajo el capó? La Arquitectura del Proyecto

El flujo de trabajo de PokéTrAIner está diseñado en varias capas para garantizar precisión en los datos y fluidez conversacional:

### 1. Interfaz de Usuario Inmersiva (Frontend)
Desarrollada con **Streamlit**, pero fuertemente inyectada con CSS y HTML personalizados para romper la rígida estructura de bloques tradicional de la plataforma. El resultado es una interfaz dinámica de "ventana de chat" (con scroll interno), una barra de entrada de texto flotante y un diseño de fondo anclado inspirado en el universo Pokémon, ofreciendo una experiencia de usuario (UX) impecable y libre de distracciones visuales.

### 2. Procesamiento de Lenguaje Natural (NLP) Avanzado
Antes de que la pregunta del usuario llegue a los motores de búsqueda, pasa por un riguroso túnel de limpieza y normalización:
* **Corrección Ortográfica Híbrida:** Utiliza `Hunspell` (el motor de diccionarios de la RAE) combinado con la librería `difflib` para corregir errores tipográficos basándose en el vocabulario oficial. Esto asegura que palabras mal escritas como "Picachu" o "atake" se transformen en "Pikachu" y "ataque" antes de ejecutar la búsqueda.
* **Lematización y Limpieza:** `spaCy` (con el modelo de lenguaje `es_core_news_md`) desglosa la oración, elimina "palabras vacías" (stopwords) y extrae la raíz de los verbos para que el buscador no se confunda con conjugaciones complejas.
* **Protección de Etiquetas Semánticas:** Un sistema de expresiones regulares (RegEx) blinda palabras clave internas del sistema (como `[CATEGORÍA: POKÉMON]`) para evitar que los motores NLP las destruyan durante la limpieza.

### 3. Motor de Búsqueda Híbrido (Retrieval)
Para recuperar la información correcta de los documentos locales (`.pkl`), el sistema no confía en un solo método, sino que cruza dos tecnologías de búsqueda con un peso matemático equilibrado (Alpha = 0.6):
* **Búsqueda Dispersa (TF-IDF):** Utilizando `Scikit-Learn`, busca coincidencias exactas de palabras clave. Es una técnica vital para no fallar al buscar nombres propios muy específicos como "Garchomp" o "Terremoto".
* **Búsqueda Densa (Embeddings):** Mediante `SentenceTransformers` (modelo `paraphrase-multilingual-MiniLM`), transforma los textos en vectores matemáticos. Esto permite al sistema entender el *contexto* y la intención de la pregunta, incluso si el usuario utiliza sinónimos o frases abstractas.

### 4. El Cerebro Orquestador: Agente ReAct con LangChain
El corazón lógico del sistema es un **Agente ReAct** (Reasoning and Acting) construido con la infraestructura de **LangChain**. Su comportamiento emula el pensamiento analítico humano:
1.  **Piensa (Thought):** Analiza la consulta del usuario y decide qué datos le faltan para poder responder de forma fiable.
2.  **Actúa (Action):** Utiliza su herramienta integrada (`herramienta_pokedex`) para lanzar consultas automatizadas al motor de búsqueda híbrido.
3.  **Observa (Observation):** Lee e ingiere los datos extraídos de la base de conocimiento local.
4.  **Responde (Final Answer):** Una vez verifica que tiene todos los datos reales necesarios, redacta la respuesta definitiva.

### 5. Generación de Lenguaje (LLM)
La redacción y el razonamiento final corren a cargo del modelo `Llama-3.3-70b-versatile`, conectado a través de la API ultrarrápida de **Groq**. El modelo está configurado con una temperatura muy baja (`0.1`) para priorizar la lógica analítica y evitar la divagación creativa, asegurando respuestas directas, precisas y en tiempos casi instantáneos.

---

## 📊 Origen de los Datos

Todo el conocimiento base, estadísticas, movimientos, objetos y naturalezas con los que se ha construido el corpus de este proyecto provienen de la magnífica [PokéAPI](https://pokeapi.co/). Un agradecimiento gigante a su comunidad por mantener esta base de datos estructurada, viva y accesible de forma gratuita.

---

## ⚠️ Limitaciones Conocidas (Capa Gratuita)

Este proyecto funciona con tecnologías de alto rendimiento pero se apoya en servicios en su capa gratuita (Free Tier). Por tanto, existen algunas limitaciones a tener en cuenta durante su uso:

* **Límites de la API (Rate Limits):** La API de Groq ofrece una velocidad increíble, pero tiene un límite estricto de *Tokens por Minuto (TPM)* y *Tokens por Día (TPD)*.
    * *Pausas tácticas:* Si el asistente procesa demasiada información de golpe, te pedirá amablemente que esperes un par de minutos antes de hacer otra consulta.
    * *Descanso diario:* Si usas el asistente de forma intensiva durante el día, es posible que agote su energía total y te pida volver al día siguiente.
* **Base de datos limitada a competitivo:** El asistente está entrenado estrictamente para analizar mecánicas de combate. No responderá (por instrucciones del sistema) a preguntas sobre lore de la serie, crossovers o dónde capturar Pokémon específicos en la aventura.

---

## 🛠️ Guía del Desarrollador (Instalación en Local)

Si quieres clonar este repositorio y ejecutar a PokéTrAIner en tu propio ordenador, sigue estos pasos:

### 1. Requisitos Previos
Asegúrate de tener instalado **Python 3.9+** en tu sistema.
Para la corrección ortográfica, es posible que necesites tener los diccionarios de Hunspell instalados en tu sistema operativo (p. ej., `sudo apt-get install hunspell-es` en Linux).

### 2. Clonar el repositorio
```bash
git clone [https://github.com/elenacanchog/PokeTrAIner](https://github.com/elenacanchog/PokeTrAIner)
cd poketrainer

```

### 3. Crear un Entorno Virtual

Es altamente recomendable aislar las dependencias del proyecto usando `venv`:

**En Windows:**

```bash
python -m venv venv
venv\Scripts\activate

```

**En macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 4. Instalar Dependencias

Instala todas las librerías necesarias con el archivo de requisitos:

```bash
pip install -r requirements.txt

```

### 5. Configurar Variables de Entorno

1. Copia el archivo de ejemplo: `cp .env.example .env` (o renómbralo manualmente).
2. Abre el archivo `.env` y añade tu clave de la API de Groq:

```env
GROQ_API_KEY="tu_clave_secreta_aqui"

```

### 6. (Opcional) Generar o Actualizar los Modelos RAG

Si has modificado los archivos `.json` de la carpeta `datos/` y necesitas actualizar el conocimiento del asistente, ejecuta el motor de indexación:

```bash
python src/buscadores.py

```

*Esto generará nuevos archivos `.pkl` con los vectores TF-IDF y los Embeddings listos para usarse.*

### 7. ¡Arrancar la Aplicación!

Inicia el servidor local de Streamlit:

```bash
streamlit run app.py

```

Se abrirá automáticamente una pestaña en tu navegador web apuntando a `http://localhost:8501`.

---

## 🎓 Origen del Proyecto

Este sistema experto nació originalmente como el Trabajo Final del curso **"PLN y Modelos de Lenguaje para la Empresa"**, perteneciente a la oferta formativa del **PGTD** (Programa de Generación de Talento Digital). 

En su primera iteración, fue desarrollado en grupo a través de un [Notebook de Google Colab](https://drive.google.com/file/d/1u95KtSoX_fOCAU5mLL92pdlJWLpiOUw1/view?usp=sharing). Tras finalizar el curso, decidí rescatar ese núcleo funcional y llevar el sistema experto al siguiente nivel a modo de reto personal. El objetivo era construir esta arquitectura completa, añadir una interfaz web inmersiva y pulir los motores de recuperación (RAG) para seguir aprendiendo y profundizando en el mundo del Procesamiento de Lenguaje Natural.

---

## 🤝 Contribuciones y Licencia

Este es un proyecto **100% Open Source**, creado por pura pasión por aprender y experimentar con IA. 

Si eres desarrollador, entusiasta del Machine Learning o simplemente un entrenador Pokémon y ves alguna forma de mejorar el código, hacer más eficientes las búsquedas o refinar la interfaz, **¡las sugerencias y Pull Requests son más que bienvenidas!** Estoy totalmente abierta a críticas constructivas y posibles mejoras para seguir creciendo profesionalmente.

