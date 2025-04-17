import numpy as np

# Função de ativação sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    return x * (1 - x)

# Dados de entrada (XOR)
entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])

# Saídas esperadas
saidas = np.array([[0],
                   [1],
                   [1],
                   [0]])

# Inicializa pesos aleatórios
np.random.seed(42)
pesos_entrada_oculta = np.random.rand(2, 2)
pesos_oculta_saida = np.random.rand(2, 1)

# Taxa de aprendizado
alpha = 0.10

# Treinamento
for epoca in range(100000):
    # Camada oculta
    entrada_oculta = np.dot(entradas, pesos_entrada_oculta)
    saida_oculta = sigmoid(entrada_oculta)

    # Camada de saída
    entrada_final = np.dot(saida_oculta, pesos_oculta_saida)
    saida_final = sigmoid(entrada_final)

    # Cálculo do erro
    erro = saidas - saida_final
    if epoca % 1000 == 0:
        print(f'Época {epoca} - Erro: {np.mean(np.abs(erro)):.4f}')

    # Backpropagation
    grad_saida = erro * sigmoid_derivada(saida_final)
    erro_oculta = grad_saida.dot(pesos_oculta_saida.T)
    grad_oculta = erro_oculta * sigmoid_derivada(saida_oculta)

    # Atualização dos pesos
    pesos_oculta_saida += saida_oculta.T.dot(grad_saida) * alpha
    pesos_entrada_oculta += entradas.T.dot(grad_oculta) * alpha

# Testando após o treinamento
print("\nResultados após treinamento:")
for entrada in entradas:
    camada_oculta = sigmoid(np.dot(entrada, pesos_entrada_oculta))
    saida = sigmoid(np.dot(camada_oculta, pesos_oculta_saida))
    print(f"Entrada: {entrada} -> Saída: {saida.round(3)}")
