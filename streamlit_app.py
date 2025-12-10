import pygame
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import random
import sys

# --- CONFIGURAÇÕES VISUAIS ---
LARGURA_TELA = 1280
ALTURA_TELA = 800
COR_FUNDO = (30, 30, 35)
COR_TEXTO = (230, 230, 230)
COR_BOTAO_PADRAO = (50, 50, 60)
COR_BOTAO_HOVER = (70, 70, 80)
COR_ACERTO = (50, 200, 50)
COR_ERRO = (200, 50, 50)

# --- BANCO DE DADOS (LINKS ESTÁVEIS DO GITHUB) ---
# Mapeamento manual de imagens do BCCD Dataset que funcionaram anteriormente
BANCO_QUESTOES = [
    {
        "url": "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00002.jpg",
        "resposta": "Neutrófilo",
        "dica": "Observe os múltiplos lobos (3 a 5) conectados."
    },
    {
        "url": "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00006.jpg",
        "resposta": "Eosinófilo",
        "dica": "Olhe a granulação grossa e brilhante na visão de Textura."
    },
    {
        "url": "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00022.jpg",
        "resposta": "Linfócito",
        "dica": "Núcleo grande, redondo e escuro. Pouco citoplasma."
    },
    {
        "url": "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00012.jpg",
        "resposta": "Monócito",
        "dica": "Núcleo irregular (dobrado), cromatina mais frouxa que o linfócito."
    },
    {
        "url": "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00109.jpg",
        "resposta": "Neutrófilo",
        "dica": "Segmentação nuclear clara."
    }
]

OPCOES_RESPOSTA = ["Neutrófilo", "Linfócito", "Monócito", "Eosinófilo", "Basófilo"]

# --- CLASSE DE BOTÃO ---
class Botao:
    def __init__(self, texto, x, y, w, h, action_value):
        self.rect = pygame.Rect(x, y, w, h)
        self.texto = texto
        self.action_value = action_value
        self.cor_atual = COR_BOTAO_PADRAO
        self.ativo = True

    def desenhar(self, tela, fonte):
        pygame.draw.rect(tela, self.cor_atual, self.rect, border_radius=8)
        pygame.draw.rect(tela, (100, 100, 100), self.rect, 2, border_radius=8)
        texto_surf = fonte.render(self.texto, True, COR_TEXTO)
        texto_rect = texto_surf.get_rect(center=self.rect.center)
        tela.blit(texto_surf, texto_rect)

    def checar_hover(self, mouse_pos):
        if not self.ativo: return
        if self.rect.collidepoint(mouse_pos):
            self.cor_atual = COR_BOTAO_HOVER
        else:
            self.cor_atual = COR_BOTAO_PADRAO

    def checar_clique(self, mouse_pos):
        if self.ativo and self.rect.collidepoint(mouse_pos):
            return True
        return False

# --- FUNÇÕES DE PROCESSAMENTO ---
def criar_celula_sintetica_backup():
    """Gera uma imagem artificial se a internet falhar."""
    print("-> Usando gerador sintético (Backup)...")
    img = np.ones((300, 300, 3), dtype=np.uint8) * 220
    # Citoplasma
    cv2.circle(img, (150, 150), 80, (200, 200, 255), -1)
    # Núcleo (Simulando Neutrófilo 3 lobos)
    cv2.circle(img, (130, 140), 25, (100, 0, 100), -1)
    cv2.circle(img, (170, 140), 25, (100, 0, 100), -1)
    cv2.circle(img, (150, 170), 25, (100, 0, 100), -1)
    return img

def baixar_imagem(url):
    # User-Agent para evitar bloqueio
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        print(f"Baixando: {url.split('/')[-1]}...")
        response = requests.get(url, headers=headers, timeout=8)
        response.raise_for_status()
        
        img_pil = Image.open(BytesIO(response.content))
        img_np = np.array(img_pil)
        
        # Correções de cor
        if len(img_np.shape) == 2: 
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[-1] == 4: 
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
            
        return img_np
    except Exception as e:
        print(f"Falha no download ({e}). Gerando backup...")
        return criar_celula_sintetica_backup()

def processar_filtros(img_rgb):
    # Redimensionamento inteligente
    h, w, _ = img_rgb.shape
    scale = 300 / w
    novo_h = int(h * scale)
    if novo_h > 300: novo_h = 300
    
    img_mini = cv2.resize(img_rgb, (300, novo_h))
    
    # 1. Cinza
    gray = cv2.cvtColor(img_mini, cv2.COLOR_RGB2GRAY)
    
    # 2. Textura (CLAHE) - Nível ALTO para ver grânulos
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    texture = clahe.apply(gray)
    
    # 3. Bordas (Canny) - Ajustado para núcleos
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 130)
    edges = cv2.dilate(edges, None, iterations=1)
    edges_inv = cv2.bitwise_not(edges) # Inverter para fundo branco

    # Converter para Pygame Surfaces
    def to_surf(array, is_gray=False):
        if is_gray:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        return pygame.image.frombuffer(array.tobytes(), (array.shape[1], array.shape[0]), 'RGB')

    return [
        to_surf(img_mini), 
        to_surf(gray, True), 
        to_surf(texture, True), 
        to_surf(edges_inv, True)
    ]

# --- JOGO PRINCIPAL ---
class HemoQuiz:
    def __init__(self):
        pygame.init()
        self.tela = pygame.display.set_mode((LARGURA_TELA, ALTURA_TELA))
        pygame.display.set_caption("HemoQuiz - Treinamento Daltônico")
        self.clock = pygame.time.Clock()
        self.fonte_grande = pygame.font.SysFont('Arial', 28, bold=True)
        self.fonte_media = pygame.font.SysFont('Arial', 20)
        
        self.acertos = 0
        self.erros = 0
        
        # Configurar Botões
        self.botoes = []
        largura_btn = 180
        espaco = 20
        total_w = len(OPCOES_RESPOSTA) * (largura_btn + espaco)
        inicio_x = (LARGURA_TELA - total_w) // 2 + 10
        
        for i, opcao in enumerate(OPCOES_RESPOSTA):
            x = inicio_x + i * (largura_btn + espaco)
            btn = Botao(opcao, x, ALTURA_TELA - 100, largura_btn, 50, opcao)
            self.botoes.append(btn)
            
        self.btn_proximo = Botao("Próxima Lâmina >>", LARGURA_TELA//2 - 100, ALTURA_TELA - 180, 200, 50, "NEXT")
        self.btn_proximo.ativo = False 

        self.surfaces = []
        self.dados_atuais = None
        self.estado = "CARREGANDO"
        self.msg_resultado = ""
        self.cor_resultado = COR_TEXTO
        
        # Start
        self.carregar_nova_questao()

    def carregar_nova_questao(self):
        self.estado = "CARREGANDO"
        self.desenhar()
        pygame.display.flip()
        
        # Lógica de seleção
        item = random.choice(BANCO_QUESTOES)
        
        # Tenta baixar. Se falhar, vem a célula sintética (não crasha)
        img_np = baixar_imagem(item['url'])
        
        # Se baixou a sintética (backup), ajustamos a resposta para Neutrófilo (padrão do desenho)
        if img_np.mean() == 220 and item['resposta'] != "Neutrófilo":
             # Detectou que é a imagem sintética pelo fundo cinza exato
             self.dados_atuais = {"resposta": "Neutrófilo", "dica": "Imagem gerada por backup (Internet falhou)."}
        else:
            self.dados_atuais = item

        self.surfaces = processar_filtros(img_np)
        self.estado = "JOGANDO"
        self.msg_resultado = ""
        
        self.btn_proximo.ativo = False
        for btn in self.botoes:
            btn.ativo = True
            btn.cor_atual = COR_BOTAO_PADRAO

    def verificar_resposta(self, resposta_usuario):
        correta = self.dados_atuais['resposta']
        
        if resposta_usuario == correta:
            self.acertos += 1
            self.msg_resultado = f"✅ CORRETO! É um {correta}."
            self.cor_resultado = COR_ACERTO
        else:
            self.erros += 1
            self.msg_resultado = f"❌ ERROU... Era um {correta}. Dica: {self.dados_atuais['dica']}"
            self.cor_resultado = COR_ERRO
        
        self.estado = "RESULTADO"
        self.btn_proximo.ativo = True
        
        for btn in self.botoes:
            btn.ativo = False
            if btn.action_value == correta:
                btn.cor_atual = COR_ACERTO
            elif btn.action_value == resposta_usuario:
                btn.cor_atual = COR_ERRO

    def desenhar(self):
        self.tela.fill(COR_FUNDO)
        
        # Placar
        texto_placar = f"Acertos: {self.acertos}  |  Erros: {self.erros}"
        self.tela.blit(self.fonte_media.render(texto_placar, True, (255, 215, 0)), (20, 20))
        
        if self.estado == "CARREGANDO":
            txt = self.fonte_grande.render("Baixando Lâmina do GitHub...", True, COR_TEXTO)
            self.tela.blit(txt, txt.get_rect(center=(LARGURA_TELA//2, ALTURA_TELA//2)))
            return

        # Imagens
        titulos = ["Original", "Luminância", "TEXTURA (Olhe!)", "FORMA (Núcleo)"]
        cores_tit = [COR_TEXTO, COR_TEXTO, (100, 200, 255), (100, 255, 100)]
        
        largura_col = LARGURA_TELA // 4
        y_pos = 80
        
        for i, surf in enumerate(self.surfaces):
            # Centralizar na coluna
            x_centro = i * largura_col + (largura_col - surf.get_width()) // 2
            self.tela.blit(surf, (x_centro, y_pos))
            
            # Título
            t_surf = self.fonte_media.render(titulos[i], True, cores_tit[i])
            self.tela.blit(t_surf, t_surf.get_rect(center=(i * largura_col + largura_col//2, y_pos - 25)))

        # Resultado e Botões
        if self.msg_resultado:
            # Fundo preto para destaque
            res_surf = self.fonte_grande.render(self.msg_resultado, True, self.cor_resultado)
            res_rect = res_surf.get_rect(center=(LARGURA_TELA//2, ALTURA_TELA - 200))
            pygame.draw.rect(self.tela, (0,0,0), res_rect.inflate(30, 20))
            self.tela.blit(res_surf, res_rect)

        if self.estado == "JOGANDO":
            for btn in self.botoes: btn.desenhar(self.tela, self.fonte_media)
        
        if self.estado == "RESULTADO":
            self.btn_proximo.desenhar(self.tela, self.fonte_grande)
            # Mostra botões inativos para ver qual era
            for btn in self.botoes: btn.desenhar(self.tela, self.fonte_media)

    def rodar(self):
        rodando = True
        while rodando:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT: rodando = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.estado == "JOGANDO":
                            for btn in self.botoes:
                                if btn.checar_clique(mouse_pos): self.verificar_resposta(btn.action_value)
                        elif self.estado == "RESULTADO":
                            if self.btn_proximo.checar_clique(mouse_pos): self.carregar_nova_questao()

            if self.estado == "JOGANDO":
                for btn in self.botoes: btn.checar_hover(mouse_pos)
            elif self.estado == "RESULTADO":
                self.btn_proximo.checar_hover(mouse_pos)

            self.desenhar()
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    HemoQuiz().rodar()
