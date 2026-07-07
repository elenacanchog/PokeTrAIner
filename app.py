import re
import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler
from PIL import Image

# IMPORTANTE: Importamos únicamente el Agente (él ya tiene las herramientas dentro)
from src.motor_rag import ejecutor_agente

# ==========================================
# 1. CONFIGURACIÓN VISUAL Y DISEÑO POKÉMON
# ==========================================
url_icono_chat = "iconos/icono_chat.png"
url_icono_user = "iconos/icono_user.png"

st.set_page_config(page_title="PokeTrAIner - RAG", page_icon=url_icono_chat, layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #FDFEFE; }
    .stChatInputContainer { border-bottom: 2px solid #3B4CCA; }
    .stChatInputContainer button { background-color: #FFDE00; color: #3B4CCA; }
    .stChatMessage.user { background-color: #EBF5FB; border-radius: 10px 10px 0px 10px; }
    .stChatMessage.assistant { background-color: #FEF9E7; border-radius: 10px 10px 10px 0px; border-left: 5px solid #FFDE00; }
    h1, h2, h3 { color: #3B4CCA !important; font-family: 'Arial Black', sans-serif; }
</style>
""", unsafe_allow_html=True)

url_banner = "iconos/banner_f.png"

try:
    st.image(url_banner, width="stretch")
except Exception as e:
    st.warning("No se pudo cargar el banner con el enlace.")

st.title("PokéTrAIner: tu asistente táctico")
st.markdown("Soy tu asistente IA experto en **Pokémon Competitivo**. Analizaré tus preguntas cruzando datos de mi Pokédex.")

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