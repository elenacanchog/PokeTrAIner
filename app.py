import re
import base64
import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
import streamlit.components.v1 as components
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
    [data-testid="stHeader"] { display: none !important; }
    footer { display: none !important; }

    /* 2. FONDO TOTAL: Anclado abajo */
    .stApp {
        background-image: url("data:image/png;base64,IMAGEN_FONDO_AQUI");
        background-size: cover !important;
        background-position: center bottom !important; 
        background-attachment: fixed !important;
    }
    [data-testid="stAppViewContainer"] { background-color: transparent !important; }

    /* 3. LA CAJA BLANCA (AHORA ES INDEPENDIENTE DEL TEXTO) */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed; 
        top: 220px; 
        bottom: 60px;
        left: 0;
        right: 0;
        margin: 0 auto; 
        width: 100%;
        max-width: 800px;
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 20px 20px 20px 20px; 
        box-shadow: 0px 10px 30px rgba(0,0,0,0.15);
        z-index: 0; 
    }

    /* 4. LA VENTANA DE MENSAJES (SCROLL) */
    .block-container {
        position: fixed !important; 
        top: 220px !important;
        bottom: 120px !important; /* El texto se corta justo por encima de la barra de escribir */
        left: 0 !important;
        right: 0 !important;
        margin: 0 auto !important; 
        width: 100% !important;
        max-width: 800px !important; 
        background: transparent !important; /* El fondo lo pone el paso 3 */
        overflow-y: auto !important; 
        padding: 20px 60px !important;
        z-index: 1;
        scrollbar-width: none; 
    }
    .block-container::-webkit-scrollbar { display: none; }

    /* 5. LA BARRA DE ESCRIBIR: Flotando en el hueco de abajo */
    [data-testid="stBottom"], 
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        max-width: 800px !important;
        left: 0 !important;
        right: 0 !important;
        margin: 0 auto !important;
        padding-bottom: 30px !important; /* Despegada del suelo para que quede dentro de la caja */
    }
    
    div[data-testid="stChatInput"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px !important;
        border: 3px solid #FF8C00 !important; 
        box-shadow: 0px 5px 15px rgba(255, 140, 0, 0.2) !important;
    }

    /* 6. ESTILOS DE LOS BOCADILLOS */
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
# 1.5. CABECERA FIJA (LOGO)
# ==========================================
# Convertimos el logo a base64 igual que el fondo
logo_base64 = obtener_base64(ruta_logo)

# Inyectamos el logo anclado al cielo (position: fixed) para que NUNCA se mueva
html_cabecera = f"""
<div style="position: fixed; top: -20px; left: 0; right: 0; margin: 0 auto; width: 100%; max-width: 800px; text-align: center; z-index: 100; pointer-events: none;">
    <img src="data:image/png;base64,{logo_base64}" style="width: 70%; max-width: 450px; margin-bottom: 5px;">
</div>
"""
st.markdown(html_cabecera, unsafe_allow_html=True)

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
                        st.warning("¡Uf! He agotado toda mi capacidad táctica por hoy. 😴 \n\nEl Centro Pokémon me está curando. Por favor, vuelve a visitarme **más tarde**.")
                        
                    # CASO B: Límite por Minuto (TPM) agotado.
                    elif "tokens per minute (TPM)" in error_str or "TPM" in error_str:
                        st.warning("¡Voy muy rápido! Necesito coger aire un momento por la enorme cantidad de datos que he cruzado. 💨 \n\nPor favor, espera **un par de minutos** y vuelve a intentarlo.")
                    
                    # CASO C: Otros límites desconocidos de la API
                    else:
                        st.warning("¡Uf! Estoy muy cansado de tanto pensar, necesito descansar la mente. 😴 \n\nVuelve a intentarlo en un ratito.")
                        
                # Si es un error interno del código, LangChain, o límite de iteraciones
                else:
                    st.error(f"Lo siento, encontré un obstáculo en mi razonamiento: {error_str}")


# ==========================================
# 3. AUTO-SCROLL MÁGICO PARA LA VENTANA
# ==========================================
# Si hay mensajes en el chat, inyectamos un script invisible para forzar el scroll
if len(st.session_state.mensajes) > 0:
    cantidad_mensajes = len(st.session_state.mensajes)
    
    components.html(
        f"""
        <script>
            // Buscamos nuestra caja blanca (block-container) a través del padre del iframe
            var caja = window.parent.document.querySelector('.block-container');
            if (caja) {{
                // La deslizamos suavemente hasta el fondo
                caja.scrollTo({{
                    top: caja.scrollHeight,
                    behavior: 'smooth'
                }});
            }}
        </script>
        """,
        height=0 
    )