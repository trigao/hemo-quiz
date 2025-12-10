import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import random

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="HemoQuiz", page_icon="🩸", layout="centered")

# --- ESTILOS CSS (Para parecer app mobile) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .css-1v0mbdj {margin-top: -50px;}
    </style>
    """, unsafe_allow_html=True)

# --- BANCO DE DADOS ---
BANCO_QUESTOES = [
    {"url": "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00002.jpg", "resposta": "Neutrófilo", "dica": "Múltiplos lobos conectados."},
    {"url": "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00006.jpg", "resposta": "Eosinófilo", "dica": "Granulação grossa e brilhante."},
    {"url": "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00022.jpg", "resposta": "Linfócito", "dica": "Núcleo redondo, escuro, quase sem citoplasma."},
    {"url": "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00012.jpg", "resposta": "Monócito", "dica": "Núcleo irregular/dobrado (rim)."},
    {"url": "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00109.jpg", "resposta": "Neutrófilo", "dica": "Segmentação clara."}
]

# --- FUNÇÕES ---
@st.cache_data # Cache para não baixar a mesma imagem toda hora
def baixar_e_processar(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        img_pil = Image.open(BytesIO(response.content))
        img_np = np.array(img_pil)
        
        if len(img_np.shape) == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[-1] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        # Processamento
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        texture = clahe.apply(gray)
        
        blurred = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blurred, 50, 130)
        edges = cv2.dilate(edges, None, iterations=1)
        edges_inv = cv2.bitwise_not(edges)
        
        return img_np, texture, edges_inv
    except:
        return None, None, None

def nova_rodada():
    st.session_state.questao_atual = random.choice(BANCO_QUESTOES)
    st.session_state.respondido = False
    st.session_state.msg_resultado = ""
    st.session_state.cor_msg = "blue"

# --- INICIALIZAÇÃO DE ESTADO ---
if 'acertos' not in st.session_state: st.session_state.acertos = 0
if 'erros' not in st.session_state: st.session_state.erros = 0
if 'questao_atual' not in st.session_state: nova_rodada()

# --- INTERFACE ---
st.title("🩸 HemoQuiz")
col1, col2 = st.columns(2)
col1.metric("Acertos", st.session_state.acertos)
col2.metric("Erros", st.session_state.erros)

item = st.session_state.questao_atual
original, textura, bordas = baixar_e_processar(item['url'])

if original is not None:
    # Abas para ver as visões
    tab1, tab2, tab3 = st.tabs(["👁️ Original", "🔬 Textura (Grânulos)", "📐 Forma (Núcleo)"])
    with tab1: st.image(original, use_container_width=True)
    with tab2: st.image(textura, use_container_width=True, caption="Foco na granulação")
    with tab3: st.image(bordas, use_container_width=True, caption="Foco na lobulação")
    
    st.write("---")
    st.subheader("Que célula é esta?")
    
    # Se ainda não respondeu, mostra botões
    if not st.session_state.respondido:
        opcoes = ["Neutrófilo", "Linfócito", "Monócito", "Eosinófilo", "Basófilo"]
        cols = st.columns(2)
        for i, opcao in enumerate(opcoes):
            if cols[i % 2].button(opcao):
                if opcao == item['resposta']:
                    st.session_state.acertos += 1
                    st.session_state.msg_resultado = f"✅ CORRETO! É um {item['resposta']}."
                    st.session_state.cor_msg = "green"
                    st.balloons()
                else:
                    st.session_state.erros += 1
                    st.session_state.msg_resultado = f"❌ ERROU! Era um {item['resposta']}.\n💡 Dica: {item['dica']}"
                    st.session_state.cor_msg = "red"
                st.session_state.respondido = True
                st.rerun()
    
    # Se já respondeu, mostra resultado e botão de próximo
    else:
        if st.session_state.cor_msg == "green":
            st.success(st.session_state.msg_resultado)
        else:
            st.error(st.session_state.msg_resultado)
            
        if st.button("Próxima Lâmina ➡️", type="primary"):
            nova_rodada()
            st.rerun()

else:
    st.error("Erro ao baixar imagem. Tentando outra...")
    nova_rodada()
    st.rerun()
