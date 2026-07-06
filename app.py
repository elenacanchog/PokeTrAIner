import streamlit as st
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
                # 1. Le pasamos directamente la pregunta cruda del usuario al Agente
                respuesta_bruta = ejecutor_agente.invoke(
                    {"input": prompt_usuario}
                )
                
                # 2. LangChain devuelve un diccionario. La respuesta final está en la clave 'output'
                respuesta_llm = respuesta_bruta["output"]
                
                # 3. Mostramos la respuesta en la interfaz
                st.markdown(respuesta_llm)
                
                # 4. Guardamos en el historial
                st.session_state.mensajes.append({"role": "assistant", "content": respuesta_llm})
                
                # NOTA TEMPORAL: Por ahora quitamos el expander de "fuentes consultadas", 
                # porque el agente gestiona las búsquedas internamente. Lo añadiremos más adelante si quieres.

            except Exception as e:
                st.error(f"Lo siento, el Agente encontró un obstáculo en su razonamiento: {e}")