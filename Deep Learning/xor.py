import numpy as np

# Função de ativação sigmóide e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)  # Derivada da sigmóide

# Inicialização dos pesos e bias
np.random.seed(42)
W1 = np.random.uniform(-1, 1, (2, 2))  # Pesos da entrada para a camada oculta (2x2)
b1 = np.random.uniform(-1, 1, (1, 2))  # Bias da camada oculta (1x2)
W2 = np.random.uniform(-1, 1, (2, 1))  # Pesos da camada oculta para a saída (2x1)
b2 = np.random.uniform(-1, 1, (1, 1))  # Bias da camada de saída (1x1)

# Função de feedforward
def feedforward(X):
    global hidden_output, final_output
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)  # Saída da camada oculta
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)  # Saída final
    return final_output

# Conjunto de treinamento (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Teste do feedforward antes do treinamento
output = feedforward(X)
print("Saída antes do treinamento:")
print(output)



# Parâmetros de treinamento
learning_rate = 0.5
epochs = 1000000  # Número de iterações

# Treinamento com backpropagation
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # Cálculo do erro
    error = Y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)  # Ajuste da camada de saída
    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(hidden_output)  # Ajuste da camada oculta

    # Atualização dos pesos e bias
    W2 += np.dot(hidden_output.T, d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += np.dot(X.T, d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Mostrar erro a cada 1000 iterações
    if epoch % 1000 == 0:
        print(f"Época {epoch}, Erro: {np.mean(np.abs(error))}")

# Testando a rede após o treinamento
output = sigmoid(np.dot(sigmoid(np.dot(X, W1) + b1), W2) + b2)
print("\nSaída após treinamento:")
print(output)