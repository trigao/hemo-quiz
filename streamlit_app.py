import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO
import random

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="HemoTreino Final", page_icon="🩸", layout="centered")

# CSS para melhorar aparência no celular
st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        margin-bottom: 10px;
    }
    img {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- BANCO DE DADOS (LINKS WIKIMEDIA ESTÁVEIS) ---
# Usando thumbnails (640px) que são mais leves e nunca mudam de endereço
if 'banco_questoes' not in st.session_state:
    st.session_state.banco_questoes = [
        # --- NEUTRÓFILOS ---
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Neutrophil_with_anthrax.jpg/640px-Neutrophil_with_anthrax.jpg",
            "resposta": "Neutrófilo",
            "dica": "Múltiplos lobos conectados (3 a 5). Citoplasma rosa pálido."
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Hypersegmented_neutrophil.jpg/640px-Hypersegmented_neutrophil.jpg",
            "resposta": "Neutrófilo",
            "dica": "Este está hipersegmentado (+5 lobos), comum em anemias, mas é um Neutrófilo."
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Band_neutrophil.jpg/640px-Band_neutrophil.jpg",
            "resposta": "Neutrófilo",
            "dica": "Neutrófilo jovem (Bastão). Núcleo em forma de C ou U sem separação completa."
        },
        
        # --- EOSINÓFILOS ---
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Eosinophil_G.jpg/640px-Eosinophil_G.jpg",
            "resposta": "Eosinófilo",
            "dica": "Filtro TEXTURA: Veja como brilha! Granulação alaranjada grossa."
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Eosinophile.jpg/640px-Eosinophile.jpg",
            "resposta": "Eosinófilo",
            "dica": "Núcleo bilobulado (óculos escuros) e citoplasma cheio de grânulos."
        },

        # --- LINFÓCITOS ---
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6c/Lymphocyte.jpg/640px-Lymphocyte.jpg",
            "resposta": "Linfócito",
            "dica": "Núcleo enorme, redondo e escuro. Ocupa quase a célula toda."
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Manteaux_lymphocyte.jpg/640px-Manteaux_lymphocyte.jpg",
            "resposta": "Linfócito",
            "dica": "Pequeno, compacto, bordas lisas. Cromatina densa (Luminância escura)."
        },
        
        # --- MONÓCITOS ---
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/Monocyte_2.jpg/640px-Monocyte_2.jpg",
            "resposta": "Monócito",
            "dica": "Núcleo irregular em forma de rim/feijão. Maior que o linfócito."
        },
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/Monocyte_1.jpg/640px-Monocyte_1.jpg",
            "resposta": "Monócito",
            "dica": "Cromatina 'frouxa' (menos preta na Luminância) e núcleo dobrado."
        },

        # --- BASÓFILOS ---
        {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Basophil_%282%29.jpg/640px-Basophil_%282%29.jpg",
            "resposta": "Basófilo",
            "dica": "Grânulos muito escuros cobrindo o núcleo. Parece uma amora."
        }
    ]

# --- FUNÇÕES ---
def baixar_url_com_retry(url):
    """Tenta baixar com headers corretos."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except:
        return None

@st.cache_data(show_spinner=False)
def processar_imagem(url):
    # Baixar
    img_pil = baixar_url_com_retry(url)
    
    if img_pil is None:
        return None, None, None

    img_np = np.array(img_pil)

    # Garantir RGB
    if len(img_np.shape) == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[-1] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
    
    # Redimensionamento suave para caber na tela sem esticar
    h, w, _ = img_np.shape
    # Fixa largura em 400px e ajusta altura proporcionalmente
    nova_w = 400
    nova_h = int(h * (nova_w / w))
    img_np = cv2.resize(img_np, (nova_w, nova_h))

    # Filtros
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Textura (CLAHE forte)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    textura = clahe.apply(gray)
    
    # Bordas
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 160)
    edges = cv2.dilate(edges, None, iterations=1)
    edges_inv = cv2.bitwise_not(edges)

    return img_np, textura, edges_inv

def sortear_nova_laminas():
    """Tenta sortear até achar uma imagem que funcione."""
    tentativas = 0
    while tentativas < 10:
        item = random.choice(st.session_state.banco_questoes)
        # Teste rápido se a imagem baixa, se não, tenta outra
        # (Na prática o st.cache ajuda a não ficar lento)
        img_teste = baixar_url_com_retry(item['url'])
        if img_teste is not None:
            st.session_state.img_atual = item
            st.session_state.respondido = False
            st.session_state.resultado = ""
            st.session_state.cor_resultado = "blue"
            return
        tentativas += 1
    
    st.error("Erro de conexão. Verifique sua internet.")

# --- ESTADO INICIAL ---
if 'acertos' not in st.session_state: st.session_state.acertos = 0
if 'erros' not in st.session_state: st.session_state.erros = 0
if 'img_atual' not in st.session_state: sortear_nova_laminas()

# --- INTERFACE ---
st.title("🩸 HemoTreino Final")

# Placar
col1, col2 = st.columns(2)
col1.metric("Acertos", st.session_state.acertos)
col2.metric("Erros", st.session_state.erros)

# Carregar Imagem
original, textura, bordas = processar_imagem(st.session_state.img_atual['url'])

if original is not None:
    # Abas de Visualização
    tab1, tab2, tab3 = st.tabs(["Original", "🔵 TEXTURA", "🟢 FORMA"])
    
    with tab1: st.image(original, use_container_width=True)
    with tab2: 
        st.image(textura, use_container_width=True)
        st.info("Filtro Textura: Destaca grânulos (brilhantes) e cromatina.")
    with tab3: 
        st.image(bordas, use_container_width=True)
        st.info("Filtro Forma: Destaca lobulação do núcleo.")

    st.divider()

    # Botões de Resposta
    if not st.session_state.respondido:
        st.subheader("O que você vê?")
        cols = st.columns(2)
        opcoes = ["Neutrófilo", "Linfócito", "Monócito", "Eosinófilo", "Basófilo"]
        
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
                    st.session_state.resultado = f"❌ ERROU! Era {correta}.\n💡 Dica: {dica}"
                    st.session_state.cor_resultado = "red"
                
                st.session_state.respondido = True
                st.rerun()
    else:
        # Mostrar Resultado e Botão Próximo
        if st.session_state.cor_resultado == "green":
            st.success(st.session_state.resultado)
        else:
            st.error(st.session_state.resultado)
        
        if st.button("Próxima Lâmina ➡️", type="primary"):
            sortear_nova_laminas()
            st.rerun()
else:
    # Se falhar tudo (raro com esse código novo)
    st.warning("Carregando...")
    sortear_nova_laminas()
    st.rerun()
