import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import random

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="HemoTreino Blindado", page_icon="🩸", layout="centered")

st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        margin-bottom: 10px;
    }
    img { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- BANCO DE DADOS: HUGGING FACE (Repositório de IA - Alta disponibilidade) ---
# Usando o dataset BCCD hospedado por 'keremberke' no Hugging Face.
# Estrutura de URL direta 'resolve/main' que permite download por script.
if 'banco_questoes' not in st.session_state:
    st.session_state.banco_questoes = [
        # NEUTRÓFILOS
        {"arquivo": "BloodImage_00000.jpg", "tipo": "Neutrófilo", "dica": "Múltiplos lobos conectados (3 a 5)."},
        {"arquivo": "BloodImage_00009.jpg", "tipo": "Neutrófilo", "dica": "Segmentação nuclear clara. Citoplasma com textura fina."},
        {"arquivo": "BloodImage_00099.jpg", "tipo": "Neutrófilo", "dica": "Núcleo dividido em várias partes."},
        
        # EOSINÓFILOS
        {"arquivo": "BloodImage_00006.jpg", "tipo": "Eosinófilo", "dica": "TEXTURA: Brilha muito! Grânulos grandes e refráteis."},
        {"arquivo": "BloodImage_00026.jpg", "tipo": "Eosinófilo", "dica": "Bilobulado (óculos escuros) + citoplasma 'areia grossa'."},
        
        # LINFÓCITOS
        {"arquivo": "BloodImage_00020.jpg", "tipo": "Linfócito", "dica": "Núcleo redondo, escuro, ocupa quase toda a célula."},
        {"arquivo": "BloodImage_00034.jpg", "tipo": "Linfócito", "dica": "Pequeno e compacto. Sem grânulos visíveis."},

        # MONÓCITOS
        {"arquivo": "BloodImage_00012.jpg", "tipo": "Monócito", "dica": "Núcleo dobrado (rim/feijão). Maior que linfócito."},
        {"arquivo": "BloodImage_00018.jpg", "tipo": "Monócito", "dica": "Forma irregular e cromatina mais frouxa (menos preta)."}
    ]

# --- GERADOR SINTÉTICO (BACKUP DE EMERGÊNCIA) ---
# Se a internet falhar, isso desenha a célula na hora.
def gerar_celula_sintetica(tipo):
    # Fundo (Hemácias borradas)
    img = np.ones((300, 300, 3), dtype=np.uint8) * 230 # Fundo claro
    
    # Cor do Núcleo (Roxo escuro) e Citoplasma
    cor_nucleo = (100, 0, 80)
    
    if tipo == "Neutrófilo":
        # Desenha 3 lobos conectados
        cv2.circle(img, (130, 150), 30, cor_nucleo, -1)
        cv2.circle(img, (170, 150), 30, cor_nucleo, -1)
        cv2.circle(img, (150, 120), 28, cor_nucleo, -1)
        # Ruído fino (grânulos neutros)
        noise = np.random.randint(0, 20, (300, 300, 3), dtype=np.uint8)
        img = cv2.subtract(img, noise)
        
    elif tipo == "Eosinófilo":
        # Desenha 2 lobos (Bilobulado)
        cv2.circle(img, (120, 150), 35, cor_nucleo, -1)
        cv2.circle(img, (180, 150), 35, cor_nucleo, -1)
        # Grânulos Grossos (Pontos brancos/brilhantes no CLAHE)
        for _ in range(300):
            x, y = np.random.randint(50, 250), np.random.randint(50, 250)
            cv2.circle(img, (x, y), 2, (50, 50, 50), -1) # Escuros na cor, mas textura grossa
            
    elif tipo == "Linfócito":
        # Um nucleo grande redondo
        cv2.circle(img, (150, 150), 60, cor_nucleo, -1)
        
    elif tipo == "Monócito":
        # Núcleo em C (Rim)
        cv2.ellipse(img, (150, 150), (60, 40), 0, 0, 360, cor_nucleo, -1)
        # "Morde" um pedaço pra fazer o feijão
        cv2.circle(img, (130, 150), 30, (230, 230, 230), -1)

    return img

# --- FUNÇÕES ---
def baixar_imagem_huggingface(filename, tipo_fallback):
    # URL do Dataset BCCD no Hugging Face (Mirror público)
    base_url = "https://huggingface.co/datasets/keremberke/blood-cell-detection-mini/resolve/main/valid/images/"
    url = base_url + filename
    
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            raise Exception("404 no HF")
    except:
        # SE FALHAR, GERA SINTÉTICO (Não mostra erro pro usuário)
        return Image.fromarray(gerar_celula_sintetica(tipo_fallback))

@st.cache_data(show_spinner=False)
def processar_visualizacao(item):
    img_pil = baixar_imagem_huggingface(item['arquivo'], item['tipo'])
    img_np = np.array(img_pil)

    # Garantir RGB
    if len(img_np.shape) == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[-1] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    # Redimensionar (Padronizar tamanho)
    img_np = cv2.resize(img_np, (400, 400))

    # Filtros
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Textura (CLAHE) - Aumentei o contraste pra ver bem os eosinófilos
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
    textura = clahe.apply(gray)
    
    # Bordas (Canny) - Ajustado para pegar contorno nuclear
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges_inv = cv2.bitwise_not(edges)

    return img_np, textura, edges_inv

def sortear():
    st.session_state.img_atual = random.choice(st.session_state.banco_questoes)
    st.session_state.respondido = False
    st.session_state.resultado = ""
    st.session_state.cor_resultado = "blue"

# --- INÍCIO ---
if 'acertos' not in st.session_state: st.session_state.acertos = 0
if 'erros' not in st.session_state: st.session_state.erros = 0
if 'img_atual' not in st.session_state: sortear()

st.title("🩸 HemoTreino Blindado")

# Placar
c1, c2 = st.columns(2)
c1.metric("Acertos", st.session_state.acertos)
c2.metric("Erros", st.session_state.erros)

# Processamento
try:
    original, textura, bordas = processar_visualizacao(st.session_state.img_atual)

    # Abas
    tab1, tab2, tab3 = st.tabs(["Original", "🔵 TEXTURA (Grânulos)", "🟢 FORMA (Núcleo)"])
    with tab1: st.image(original, use_container_width=True)
    with tab2: 
        st.image(textura, use_container_width=True)
        st.info("Dica: Se parecer 'areia grossa/pedras', é Eosinófilo.")
    with tab3: 
        st.image(bordas, use_container_width=True)
        st.info("Dica: Conte os lobos. Redondo = Linfócito. Segmentado = Neutrófilo.")

    st.divider()

    # Botões
    if not st.session_state.respondido:
        st.subheader("Identifique a célula:")
        cols = st.columns(2)
        opcoes = ["Neutrófilo", "Linfócito", "Monócito", "Eosinófilo"]
        
        for i, op in enumerate(opcoes):
            if cols[i%2].button(op):
                correta = st.session_state.img_atual['tipo']
                if op == correta:
                    st.session_state.acertos += 1
                    st.session_state.resultado = f"✅ ACERTOU! É um {correta}."
                    st.session_state.cor_resultado = "green"
                else:
                    st.session_state.erros += 1
                    st.session_state.resultado = f"❌ ERROU! Era {correta}.\n💡 {st.session_state.img_atual['dica']}"
                    st.session_state.cor_resultado = "red"
                st.session_state.respondido = True
                st.rerun()

    else:
        if st.session_state.cor_resultado == "green":
            st.success(st.session_state.resultado)
        else:
            st.error(st.session_state.resultado)
        
        if st.button("Próxima Lâmina ➡️", type="primary"):
            sortear()
            st.rerun()

except Exception as e:
    st.error(f"Erro inesperado: {e}")
    if st.button("Reiniciar"):
        sortear()
        st.rerun()
