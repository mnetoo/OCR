from PIL import Image
import subprocess




caminho = "./image.jpeg"
caminho_saida = "resized_image.jpg"

#=================================================================================================

# Função para carregar uma imagem
def load_image(caminho):

    print("Loading image...")

    imagem = Image.open(caminho).convert('L')
    largura, altura = imagem.size
    matriz = [[imagem.getpixel((x, y)) for x in range(largura)] for y in range(altura)]

    print("Image loaded successfully!\n")
    return matriz, largura, altura

#=================================================================================================

# Função para calcular novo tamanho da imagem
def calculate_size(largura, altura, escala):

    print("Calculating new image size...")

    nova_largura = int(largura * escala)
    nova_altura = int(altura * escala)
    print("Values calculated successfully!")

    print(f"New dimension: {nova_largura}x{nova_altura}\n")
    return nova_largura, nova_altura

#=================================================================================================

# Função para encontrar o peso bicúbico
def bicubic_weight(t):

    # Distância do pixel a ser interpolado
    t = abs(t)
    if t < 1:
        return (1.5 * t**3) - (2.5 * t**2) + 1
    elif t < 2:
        return (-0.5 * t**3) + (2.5 * t**2) - (4 * t) + 2
    else:
        return 0

#=================================================================================================

# Função para fazer a interpolação bicúbica
def interpolation(matriz, x, y):

    # obtém a altura (número de linhas) e altura (número de colunas)
    altura = len(matriz)
    largura = len(matriz[0])

    x_base = int(x)
    y_base = int(y)
    resultado = 0.0

    # Loop duplo para percorrer os 16 pixels vizinhos (4x4) ao redor do ponto (x, y)
    for i in range(-1, 3):
        for j in range(-1, 3):

            # Calcula as coordenadas do pixel vizinho, garantindo que não ultrapassem os limites da imagem
            px = min(max(x_base + i, 0), largura - 1)
            py = min(max(y_base + j, 0), altura - 1)
            
            # Calcula os pesos bicúbicos nas direções x e y para o pixel atual
            peso_x = bicubic_weight(x - (x_base + i))
            peso_y = bicubic_weight(y - (y_base + j))

            # Acumula o valor do pixel ponderado pelos pesos
            resultado += matriz[py][px] * peso_x * peso_y

    # Garante que o resultado esteja no intervalo [0, 255] e converte para inteiro.
    return int(min(max(resultado, 0), 255))

#=================================================================================================

# Função para fazer o redimensionamento da imagem
def resize_image(matriz, largura, altura, nova_largura, nova_altura):

    print("Resizing the image...")

    # Calcula os fatores de escala em x e y
    fator_x = largura / nova_largura
    fator_y = altura / nova_altura

    # Cria uma nova matriz vazia com as dimensões desejadas
    nova_matriz = [[0 for _ in range(nova_largura)] for _ in range(nova_altura)]

    # Loop duplo para percorrer todos os pixels da nova imagem
    for y in range(nova_altura):
        for x in range(nova_largura):

            # Calcula as coordenadas correspondentes na imagem original
            origem_x = x * fator_x
            origem_y = y * fator_y

            # Chama interpolation para calcular o valor do pixel na nova imagem.
            nova_matriz[y][x] = interpolation(matriz, origem_x, origem_y)

    print("Applied resizing successfully!\n")
    return nova_matriz

#=================================================================================================

# Função para salvar a imagem redimensionada
def save_image(matriz, caminho_saida):

    print("Saving new image...")

    altura = len(matriz)
    largura = len(matriz[0])
    image = Image.new('L', (largura, altura))

    for y in range(altura):
        for x in range(largura):
            image.putpixel((x, y), matriz[y][x])

    image.save(caminho_saida)
    print(f"Image saved as {caminho_saida}")

#=================================================================================================

# Main
matriz_imagem, largura, altura = load_image(caminho)

escala = float(input("Enter the scale value: "))
nova_largura, nova_altura = calculate_size(largura, altura, escala)

# Aplicar o redimensionamento
nova_matriz = resize_image(matriz_imagem, largura, altura, nova_largura, nova_altura)

# Salvar a imagem
save_image(nova_matriz, caminho_saida)




#Abre a imagem original
subprocess.run(["xdg-open", "image.jpeg"])

#Abre a imagem Blur
subprocess.run(["xdg-open", "resized_image.jpg"])