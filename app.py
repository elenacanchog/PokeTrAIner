import streamlit as st
from PIL import Image

# IMPORTANTE: Aquí importamos tu lógica desde la carpeta src
from src.motor_rag import buscar_informacion, construir_prompt, llm_generador

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
                # Usamos la función importada de tu motor
                documentos_recuperados = buscar_informacion(prompt_usuario, top_k=3, alpha=0.8)
                contexto_unido = "\n- ".join([doc['texto'] for doc in documentos_recuperados])

                # Usamos la función importada de tu motor
                final_prompt = construir_prompt(prompt_usuario, contexto_unido)
                
                # # Llamamos a la API usando el modelo cargado en el motor
                # respuesta_llm = llm_generador.text_generation(
                #     prompt=final_prompt,
                #     max_new_tokens=600,
                #     temperature=0.2,
                #     repetition_penalty=1.1
                # ).strip()

                final_prompt = construir_prompt(prompt_usuario, contexto_unido)
                
                # Como final_prompt ya es una lista [system, user], se la pasamos directa a la API
                respuesta_api = llm_generador.chat_completion(
                    messages=final_prompt,
                    max_tokens=1024,
                    temperature=0.1,
                    top_p=0.9
                )
                
                # Extraemos el texto de la respuesta (equivalente a tu salida_raw[0]['generated_text'])
                respuesta_llm = respuesta_api.choices[0].message.content.strip()

                st.markdown(respuesta_llm)

                with st.expander("Ver fuentes de Pokédex consultadas"):
                    for i, f in enumerate(documentos_recuperados):
                        st.write(f"   [{i+1}] {f['texto']}")

                st.session_state.mensajes.append({"role": "assistant", "content": respuesta_llm})

            except Exception as e:
                st.error(f"Lo siento, ocurrió un error en la base de datos: {e}")