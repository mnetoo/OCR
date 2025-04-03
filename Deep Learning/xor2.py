import numpy as np

# Ativação: Sigmóide (ativa os neurônios)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da sigmóide (usada no ajuste de pesos)
def sigmoid_derivative(x):
    return x * (1 - x)

# Dados do XOR (entradas e saídas esperadas)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 4 pares de entrada
Y = np.array([[0], [1], [1], [0]])  # Saída esperada

# Inicializar pesos aleatórios
np.random.seed(1)
W1 = np.random.rand(2, 2)  # 2 entradas -> 2 neurônios ocultos
W2 = np.random.rand(2, 1)  # 2 neurônios ocultos -> 1 saída

# Taxa de aprendizado
learning_rate = 0.5

# Número de épocas (quantas vezes treinamos a rede)
epochs = 100000  # Para debug, rodamos menos épocas para ver detalhes

# Treinamento detalhado
for epoch in range(epochs):
    print(f"\nÉPOCA {epoch + 1} -------------------------")
    
    # Forward Pass (Propagação da Informação)
    hidden_input = np.dot(X, W1)  # Multiplicação da entrada pelos pesos da camada oculta
    hidden_output = sigmoid(hidden_input)  # Aplicação da ativação sigmóide

    final_input = np.dot(hidden_output, W2)  # Multiplicação pela camada de saída
    final_output = sigmoid(final_input)  # Aplicação da ativação sigmóide

    # Cálculo do erro
    error = Y - final_output  

    print("Pesos antes da atualização:")
    print("W1:\n", W1)
    print("W2:\n", W2)

    print("\nSaída da camada oculta:")
    print(hidden_output)

    print("\nSaída final (antes do ajuste):")
    print(final_output)

    print("\nErro:")
    print(error)

    # Backpropagation (Ajuste dos Pesos)
    d_output = error * sigmoid_derivative(final_output)  # Ajuste da saída
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(hidden_output)  # Ajuste da oculta

    # Atualizar pesos
    W2 += np.dot(hidden_output.T, d_output) * learning_rate
    W1 += np.dot(X.T, d_hidden) * learning_rate

    print("\nPesos depois da atualização:")
    print("W1:\n", W1)
    print("W2:\n", W2)

# Testando a rede após treinamento
print("\nSaída final após treinamento:")
print(sigmoid(np.dot(sigmoid(np.dot(X, W1)), W2)))