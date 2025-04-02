from PIL import Image
import numpy as np
import subprocess




#Carrega a imagem
print("Carregando imagem original...")
imagem = Image.open("./image.jpeg").convert("L")

#Converte a imagem para array numpy
matriz_imagem = np.array(imagem)
print("Imagem convertida para matriz!\n")


# Verifica a estrutura da matriz
#DEBUG: print("Formato da matriz (altura, largura, canais):", matriz_imagem.shape)
#DEBUG: print("Tipo dos dados:", matriz_imagem.dtype)
#DEBUG: print("Matriz de pixels (amostra):\n", matriz_imagem)


#========================================================================================================


#Função para criar o kernel gaussiano
def criar_kernel(tamanho, sigma):
    kernel = np.zeros((tamanho, tamanho)) # cria uma matriz preenchida com zeros
    centro = tamanho // 2 # calcula o indice central
    for x in range (tamanho):
        for y in range (tamanho):
            # kernel[x, y] =>                       calcula a fórmula gaussiana para cada posição (x, y)
            # np.exp =>                             calcula a função exponencial
            # (x - centro)**2 +(y - centro)**2) =>  distância ao centro
            # 2 * sigma**2 =>                       dispersão do kernel
            kernel[x, y] = np.exp(-((x - centro)**2 +(y - centro)**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


#========================================================================================================


# Função de convolução para imagens monocromáticas
def convolucao(imagem, kernel):
    altura, largura = imagem.shape #        obtém a altura e largura da imagem
    k_altura, k_largura = kernel.shape #    obtém a altura e largura do kernel  
    pad = k_altura // 2 #                   calcula o preenchimento necessário para manter o tamanho da imagem após convolução (bordar artificiais)
    imagem_padded = np.pad(imagem, ((pad, pad), (pad, pad)), mode='reflect') #  adicona o padding a imagem - reflect espelha os pixels da borda
    imagem_suavizada = np.zeros_like(imagem) #  cria uma imagem vazia do mesmo tamanho da imagem original
    
    for y in range(altura):
        for x in range(largura):
            regiao = imagem_padded[y: y + k_altura, x: x + k_largura] #     região que será multiplicada pelo kernel - janela com o mesmo tamanho do kernel
            imagem_suavizada[y, x] = np.sum(regiao * kernel) #              multiplica elemento por elemento a região da imagem pelo kernel      
#                                                                           e armazena como o novo valor do pixel

    return imagem_suavizada.astype(np.uint8) #  Converte a matriz final para valores inteiros de 0 a 255 (uint8), garantindo que a imagem fique no formato correto.


#========================================================================================================


#Cria o kernel gaussiano
print("Criando kernel...")
kernel_gauss = criar_kernel(tamanho = 5, sigma = 5.0)
print("Kernel criado!\n")


#Aplica a convolução na matriz (Blur)
print("Aplicando convoluções...")
matriz_suavizada = matriz_imagem

for i in range(5):  # Aplica o blur 5 vezes
    matriz_suavizada = convolucao(matriz_suavizada, kernel_gauss)

print("Convoluções aplicadas!\n")

#========================================================================================================


#Converte a matriz para imagem e salva
print("Convertendo matriz para a imagem...")
imagem_suavizada = Image.fromarray(matriz_suavizada)
imagem_suavizada.save("image_suavizada.jpg")
print("Imagem com blur salva!")


#Abre a imagem original
subprocess.run(["xdg-open", "image.jpeg"])

#Abre a imagem Blur
subprocess.run(["xdg-open", "image_suavizada.jpg"])