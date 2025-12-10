import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import random

# --- CONFIGURAÇÃO DA PÁGINA (WEB) ---
st.set_page_config(page_title="HemoTreino Pro", page_icon="🩸", layout="centered")

# CSS para botões grandes no celular
st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        margin-bottom: 10px;
    }
    img {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- BANCO DE DADOS (12 IMAGENS) ---
if 'banco_questoes' not in st.session_state:
    st.session_state.banco_questoes = [
        # --- NEUTRÓFILOS ---
        {"url": "https://raw.githubusercontent.com/Ace20/Identify_Blood_Cell/master/Data/test/NEUTROPHIL/_0_660.jpeg", "resposta": "Neutrófilo", "dica": "Múltiplos lobos conectados (3+)."},
        {"url": "https://raw.githubusercontent.com/Ace20/Identify_Blood_Cell/master/Data/test/NEUTROPHIL/_0_928.jpeg", "resposta": "Neutrófilo", "dica": "Clássico segmentado. Citoplasma 'sujo' fino."},
        {"url": "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00109.jpg", "resposta": "Neutrófilo", "dica": "Segmentação nuclear clara."},
        
        # --- EOSINÓFILOS ---
        {"url": "https://raw.githubusercontent.com/Ace20/Identify_Blood_Cell/master/Data/test/EOSINOPHIL/_0_161.jpeg", "resposta": "Eosinófilo", "dica": "Olhe a Textura: Brilha muito, parece areia grossa."},
        {"url": "https://raw.githubusercontent.com/Ace20/Identify_Blood_Cell/master/Data/test/EOSINOPHIL/_0_207.jpeg", "resposta": "Eosinófilo", "dica": "Bilobulado (óculos) + grânulos brilhantes."},
        {"url": "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00006.jpg", "resposta": "Eosinófilo", "dica": "Muitos grânulos grandes, diferente da 'poeira' do neutrófilo."},

        # --- LINFÓCITOS ---
        {"url": "https://raw.githubusercontent.com/Ace20/Identify_Blood_Cell/master/Data/test/LYMPHOCYTE/_0_1052.jpeg", "resposta": "Linfócito", "dica": "Bola escura e densa. Quase sem citoplasma."},
        {"url": "https://raw.githubusercontent.com/Ace20/Identify_Blood_Cell/master/Data/test/LYMPHOCYTE/_0_1993.jpeg", "resposta": "Linfócito", "dica": "Núcleo regular e liso (bola de bilhar)."},
        {"url": "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00020.jpg", "resposta": "Linfócito", "dica": "Pequeno e compacto comparado às hemácias."},

        # --- MONÓCITOS ---
        {"url": "https://raw.githubusercontent.com/Ace20/Identify_Blood_Cell/master/Data/test/MONOCYTE/_0_1399.jpeg", "resposta": "Monócito", "dica": "Núcleo dobrado (rim/feijão). Maior que linfócito."},
        {"url": "https://raw.githubusercontent.com/Ace20/Identify_Blood_Cell/master/Data/test/MONOCYTE/_0_9407.jpeg", "resposta": "Monócito", "dica": "Forma irregular e cromatina mais frouxa."},
        {"url": "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00012.jpg", "resposta": "Monócito", "dica": "Grande, espalhado, núcleo não é redondo perfeito."}
    ]

# --- FUNÇÕES DE PROCESSAMENTO ---
@st.cache_data
def carregar_imagem(url):
    try:
        # 1. Baixar Imagem
        response = requests.get(url, timeout=5)
        img_pil = Image.open(BytesIO(response.content))
        img_np = np.array(img_pil)

        # 2. Corrigir canais de cor (Garante que é RGB)
        if len(img_np.shape) == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[-1] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        
        # 3. Redimensionar para padronizar (SEM CORTAR NADA)
        # Redimensionamos para 400px de largura e altura proporcional para caber na tela do celular
        h, w, _ = img_np.shape
        nova_largura = 400
        nova_altura = int(h * (nova_largura / w))
        img_np = cv2.resize(img_np, (nova_largura, nova_altura))

        # 4. Criar Filtros Daltônicos
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # CLAHE (Textura Exagerada)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
        texture = clahe.apply(gray)
        
        # Bordas (Forma Pura)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 60, 160)
        edges = cv2.dilate(edges, None, iterations=1)
        edges_inv = cv2.bitwise_not(edges) # Inverte para fundo branco

        return img_np, texture, edges_inv
    except Exception as e:
        print(f"Erro: {e}")
        return None, None, None

def proxima_pergunta():
    st.session_state.img_atual = random.choice(st.session_state.banco_questoes)
    st.session_state.respondido = False
    st.session_state.resultado = ""
    st.session_state.cor_resultado = "blue"

# --- INÍCIO DO APP ---
if 'acertos' not in st.session_state: st.session_state.acertos = 0
if 'erros' not in st.session_state: st.session_state.erros = 0
if 'img_atual' not in st.session_state: proxima_pergunta()

# Cabeçalho e Placar
st.title("🩸 HemoTreino Pro")
col_p1, col_p2 = st.columns(2)
col_p1.metric("Acertos", st.session_state.acertos)
col_p2.metric("Erros", st.session_state.erros)

# Processamento da Imagem Atual
original, textura, bordas = carregar_imagem(st.session_state.img_atual['url'])

if original is not None:
    # Abas (Melhor que colunas no celular)
    aba1, aba2, aba3 = st.tabs(["👁️ Original", "🔵 TEXTURA (Grânulos)", "🟢 FORMA (Núcleo)"])
    
    with aba1: st.image(original, use_container_width=True)
    with aba2: 
        st.image(textura, use_container_width=True)
        st.info("Dica: Eosinófilos brilham aqui. Neutrófilos parecem poeira.")
    with aba3: 
        st.image(bordas, use_container_width=True)
        st.info("Dica: Conte os lobos aqui. 1 redondo = Linfócito. 3+ ligados = Neutrófilo.")

    st.divider()

    # Área de Resposta
    if not st.session_state.respondido:
        st.subheader("Qual é a célula?")
        cols = st.columns(2)
        opcoes = ["Neutrófilo", "Linfócito", "Monócito", "Eosinófilo"]
        
        for i, op in enumerate(opcoes):
            if cols[i%2].button(op):
                correta = st.session_state.img_atual['resposta']
                if op == correta:
                    st.session_state.acertos += 1
                    st.session_state.resultado = f"✅ ACERTOU! É um {correta}."
                    st.session_state.cor_resultado = "green"
                    st.balloons()
                else:
                    st.session_state.erros += 1
                    dica = st.session_state.img_atual['dica']
                    st.session_state.resultado = f"❌ ERROU! Era {correta}.\n💡 Motivo: {dica}"
                    st.session_state.cor_resultado = "red"
                
                st.session_state.respondido = True
                st.rerun()
    
    # Área de Resultado
    else:
        if st.session_state.cor_resultado == "green":
            st.success(st.session_state.resultado)
        else:
            st.error(st.session_state.resultado)
        
        if st.button("Próxima Lâmina ➡️", type="primary"):
            proxima_pergunta()
            st.rerun()

else:
    st.warning("Erro ao baixar imagem. Tentando outra...")
    if st.button("Tentar Novamente"):
        proxima_pergunta()
        st.rerun()
