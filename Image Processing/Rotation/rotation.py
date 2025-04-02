from PIL import Image
import numpy as np

# Carrega a imagem
print("Carregando imagem original...")
imagem = Image.open("./image.jpeg").convert("L")

# Converte a imagem para array numpy
matriz_imagem = np.array(imagem)
print("Imagem convertida para matriz!\n")

#============================================================================

# Função para rotacionar a imagem em um ângulo fornecido
def rotacionar(matriz_imagem, angulo):
    # Obtém o tamanho da imagem original
    altura, largura = matriz_imagem.shape
    
    # Converte o ângulo para radianos
    angulo_rad = np.radians(angulo)

    # Calcula os valores de seno e cosseno do ângulo
    cos_a, sen_a = np.cos(angulo_rad), np.sin(angulo_rad)

    # Calcula as novas dimensões da imagem rotacionada
    nova_largura = int(abs(largura * cos_a) + abs(altura * sen_a))
    nova_altura = int(abs(largura * sen_a) + abs(altura * cos_a))

    # Cria uma matriz vazia para armazenar a imagem rotacionada
    imagem_rotacionada = np.zeros((nova_altura, nova_largura), dtype=np.uint8)

    # Calcula os centros das imagens original e rotacionada
    centro_x, centro_y = largura // 2, altura // 2
    novo_centro_x, novo_centro_y = nova_largura // 2, nova_altura // 2

    for y in range(nova_altura):
        for x in range(nova_largura):
            # Calcula a posição correspondente na imagem original
            x_original = int((x - novo_centro_x) * cos_a + (y - novo_centro_y) * sen_a + centro_x)
            y_original = int(-(x - novo_centro_x) * sen_a + (y - novo_centro_y) * cos_a + centro_y)

            # Verifica se a posição original está dentro dos limites da imagem
            if 0 <= x_original < largura and 0 <= y_original < altura:
                imagem_rotacionada[y, x] = matriz_imagem[y_original, x_original]

    return imagem_rotacionada

#============================================================================

# Aplicar rotação de x graus
print("Aplicando rotação...")
matriz_rotacionada = rotacionar(matriz_imagem, 150)
print("Rotação aplicada!\n")

# Salvar a imagem rotacionada
# Converte a matriz para imagem e salva
print("Convertendo matriz para a imagem...")
imagem_rotacionada = Image.fromarray(matriz_rotacionada)
imagem_rotacionada.save("image_rotacionada.jpg")
print("Imagem com rotação salva!")