import re
import base64
import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from PIL import Image

# IMPORTANTE: Importamos únicamente el Agente (él ya tiene las herramientas dentro)
from src.motor_rag import ejecutor_agente

# ==========================================
# FUNCIÓN PARA LEER IMÁGENES LOCALES EN CSS
# ==========================================
def obtener_base64(ruta_imagen):
    try:
        with open(ruta_imagen, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"No se pudo cargar la imagen de fondo: {e}")
        return ""

# ==========================================
# 1. CONFIGURACIÓN VISUAL Y DISEÑO POKÉMON
# ==========================================
url_icono_chat = "iconos/icono_chat.png"    
url_icono_user = "iconos/icono_user.png"
ruta_fondo = "iconos/fondo_limpio.png" 
ruta_logo = "iconos/logo.png"

st.set_page_config(page_title="PokeTrAIner - RAG", page_icon=url_icono_chat, layout="wide")

fondo_base64 = obtener_base64(ruta_fondo)

# Quitamos la 'f' inicial y usamos un string normal puro. 
# Luego reemplazamos la palabra clave con nuestra imagen en base64.
estilos_css = """
<style>
    /* 1. ELIMINAR FRANJAS BLANCAS Y CABECERA */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    footer {
        display: none !important;
    }

    /* 2. FONDO TOTAL: Anclado abajo para proteger los pies de los personajes */
    .stApp {
        background-image: url("data:image/png;base64,IMAGEN_FONDO_AQUI");
        background-size: cover !important;
        background-position: center bottom !important; 
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
    }

    /* 🚨 LA SOLUCIÓN: Matamos el fondo blanco asqueroso de la barra de Streamlit 🚨 */
    [data-testid="stBottom"], 
    .stChatInputContainer {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
    }

    /* 3. CAJA CENTRAL DEL CHAT */
    .block-container {
        max-width: 800px !important; 
        padding-top: 1rem !important;
        padding-bottom: 120px !important; 
        margin-top: 2rem;
        position: relative; 
        z-index: 1;
    }

    /* LA CAJA BLANCA: Ahora es FIJA y llega hasta el suelo absoluto para envolver la barra */
    .block-container::before {
        content: "";
        position: fixed; /* FIJA EN LA PANTALLA */
        top: 350px; 
        bottom: 0px; /* Pegada al suelo de la pantalla */
        left: 0;
        right: 0;
        margin: 0 auto; /* La mantiene centrada */
        max-width: 800px;
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 20px 20px 0px 0px; /* Bordes redondos arriba, rectos abajo */
        box-shadow: 0px 10px 30px rgba(0,0,0,0.15);
        z-index: -1; 
    }

    /* 4. LA BARRA DE ESCRIBIR: Centrada y metida dentro de la caja */
    [data-testid="stBottom"] {
        max-width: 800px !important;
        margin: 0 auto !important; 
        left: 0; 
        right: 0;
        padding-bottom: 30px !important; /* Despegamos la barra del suelo visualmente */
    }
    
    div[data-testid="stChatInput"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px;
        border: 2px solid #3B4CCA !important;
        box-shadow: 0px -5px 15px rgba(0,0,0,0.1) !important;
    }

    /* 5. ESTILO DE LOS BOCADILLOS DE CHAT */
    .stChatMessage.user {
        background-color: rgba(235, 245, 251, 0.95) !important; 
        border-radius: 12px 12px 0px 12px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
    }
    .stChatMessage.assistant {
        background-color: rgba(254, 249, 231, 0.95) !important; 
        border-radius: 12px 12px 12px 0px; 
        border-left: 5px solid #FFDE00 !important;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
    }
    .stChatMessage p { color: #1A1A1A !important; }
</style>
""".replace("IMAGEN_FONDO_AQUI", fondo_base64)

st.markdown(estilos_css, unsafe_allow_html=True)

# ==========================================
# 1.5. CABECERA (LOGO + TEXTO)
# ==========================================
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    try:
        # Usamos width="stretch" para cumplir con las nuevas reglas de Streamlit
        st.image(ruta_logo, width="stretch")
    except:
        pass

# Añadimos un poco de margen superior (margin-top) para separarlo del logo
st.markdown("<p style='text-align: center; font-size: 1.5rem; font-weight: 900; color: #3B4CCA; text-shadow: 1px 1px 0px #FFDE00; margin-top: 15px;'>Tu asistente táctico</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-weight: bold; color: #2C3E50; margin-bottom: 20px; padding: 0 20px;'>Soy tu asistente IA experto en Pokémon Competitivo. Analizaré tus preguntas cruzando datos de mi Pokédex.</p>", unsafe_allow_html=True)

# 1. CLASE PARA MEDIR LOS TOKENS EN SEGUNDO PLANO
class RastreadorTokens(BaseCallbackHandler):
    def __init__(self):
        self.tokens_totales = 0

    def on_llm_end(self, response, **kwargs):
        """Atrapa la respuesta del LLM antes de que llegue al usuario para contar los tokens"""
        try:
            # Buscamos los metadatos de uso en la respuesta de LangChain/Groq
            for generation_list in response.generations:
                for generation in generation_list:
                    # Método moderno de LangChain
                    if hasattr(generation, 'message') and hasattr(generation.message, 'usage_metadata'):
                        if generation.message.usage_metadata:
                            self.tokens_totales += generation.message.usage_metadata.get('total_tokens', 0)
                    # Método clásico de respaldo
                    elif response.llm_output and "token_usage" in response.llm_output:
                        self.tokens_totales += response.llm_output["token_usage"].get("total_tokens", 0)
        except Exception:
            pass # Si falla el conteo, simplemente lo ignoramos para no romper la app

# ==========================================
# 2. BUCLE DE CHAT (Interfaz)
# ==========================================
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for mensaje in st.session_state.mensajes:
    if mensaje["role"] == "assistant":
        with st.chat_message("assistant", avatar=url_icono_chat):
            st.markdown(mensaje["content"])
    else:
        with st.chat_message("user", avatar=url_icono_user):
            st.markdown(mensaje["content"])

if prompt_usuario := st.chat_input("¿Qué te interesa saber?"):
    st.session_state.mensajes.append({"role": "user", "content": prompt_usuario})
    
    with st.chat_message("user", avatar=url_icono_user):
        st.markdown(prompt_usuario)

    with st.chat_message("assistant", avatar=url_icono_chat):
        with st.spinner("Analizando datos..."):
            try:
                # Instanciamos el rastreador de tokens
                medidor = RastreadorTokens()

                # 1. Le pasamos la pregunta al Agente y le conectamos el medidor
                respuesta_bruta = ejecutor_agente.invoke(
                    {"input": prompt_usuario},
                    config={"callbacks": [medidor]}
                )
                
                # 2. LangChain devuelve un diccionario. La respuesta final está en la clave 'output'
                respuesta_llm = respuesta_bruta["output"]
                
                # 3. Mostramos la respuesta en la interfaz
                st.markdown(respuesta_llm)

                # Imprimimos el consumo de tokens justo debajo de la respuesta
                print(f"🔋 Tokens consumidos en esta consulta: {medidor.tokens_totales}")
                
                # 4. Guardamos en el historial
                st.session_state.mensajes.append({"role": "assistant", "content": respuesta_llm})

            except Exception as e:
                error_str = str(e)
                
                # Si el error es por llegar al límite de la API de Groq
                if "429" in error_str or "Rate limit" in error_str:
                    
                    # CASO A: Límite Diario (TPD) agotado. 
                    if "tokens per day (TPD)" in error_str or "TPD" in error_str:
                        # Extraemos el tiempo exacto de reseteo de la primera consulta del día anterior
                        match = re.search(r"Please try again in ([a-zA-Z0-9\.]+)\.", error_str)
                        tiempo_crudo = match.group(1) if match else "unas horas"
                        
                        # Formateamos el texto para un español natural
                        tiempo_bonito = re.sub(r'\.\d+s', ' segundos', tiempo_crudo) 
                        tiempo_bonito = tiempo_bonito.replace('h', ' horas, ').replace('m', ' minutos y ').replace('s', ' segundos')
                        
                        st.warning(f"¡Uf! He agotado toda mi energía táctica por hoy. 😴 \n\nPor favor, déjame descansar y vuelve a preguntarme en: **{tiempo_bonito}**.")
                        
                    # CASO B: Límite por Minuto (TPM) agotado debido a la acumulación de búsquedas en esta consulta.
                    elif "tokens per minute (TPM)" in error_str or "TPM" in error_str:
                        # Extraemos el tiempo de espera corto
                        match = re.search(r"Please try again in ([a-zA-Z0-9\.]+)\.", error_str)
                        tiempo_crudo = match.group(1) if match else "unos segundos"
                        
                        tiempo_bonito = re.sub(r'\.\d+s', ' segundos', tiempo_crudo) 
                        tiempo_bonito = tiempo_bonito.replace('h', ' horas, ').replace('m', ' minutos y ').replace('s', ' segundos')
                        
                        st.warning(f"¡Voy muy rápido! El servidor necesita coger aire un momento por la cantidad de datos cruzados. 💨 \n\nPor favor, vuelve a intentarlo en: **{tiempo_bonito}**.")
                    
                    # CASO C: Otros límites desconocidos de la API
                    else:
                        st.warning("¡Uf! Estoy muy cansado de tanto pensar, necesito descansar la mente. 😴 \n\nVuelve a intentarlo en un ratito.")
                        
                # Si es un error interno del código, LangChain, o límite de iteraciones (Agent stopped...)
                else:
                    st.error(f"Lo siento, el Agente encontró un obstáculo en su razonamiento: {error_str}")