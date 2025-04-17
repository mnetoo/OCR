import numpy as np

np.random.seed(3)  # semente randômica

LEARNING_RATE = 0.01

index_list = [0, 1, 2, 3]  # força a aprender o padrão geral e não uma sequência específica

# valores de entrada (1.0 é o bias)
x_train = [np.array([1.0, -1.0, -1.0]),
           np.array([1.0, -1.0, 1.0]),
           np.array([1.0, 1.0, -1.0]),
           np.array([1.0, 1.0, 1.0])]

y_train = [0.0, 1.0, 1.0, 0.0]  # saída esperada


#===============================================================================================================


def neuron_w(input_count):  # input_count -> quantidade de entradas
    weights = np.zeros(input_count + 1)

    for i in range(1, input_count + 1):
        weights[i] = np.random.uniform(-1.0, 1.0)
    return weights


#===============================================================================================================


n_w = [neuron_w(2), neuron_w(2), neuron_w(2)]  # lista com três neurônios (2 entradas + bias)
n_y = [0, 0, 0]  # saída dos neurônios
n_error = [0, 0, 0]  # erro dos neurônios


#===============================================================================================================


def show_learning():
    # Imprime os pesos atuais de todos os neurônios da rede
    print('Current weights:')  # Título para indicar que os pesos serão mostrados

    # Percorre cada neurônio da rede (0, 1 e 2) e seus respectivos vetores de pesos
    for i, w in enumerate(n_w):
        # Imprime os pesos do neurônio atual:
        # w[0] = bias
        # w[1] = peso da primeira entrada
        # w[2] = peso da segunda entrada
        print('neuron', i, ': w0 =', '%5.2f' % w[0],  # w0: peso do bias
              ', w1 =', '%5.2f' % w[1],              # w1: peso da 1ª entrada
              ', w2 =', '%5.2f' % w[2])              # w2: peso da 2ª entrada

    print('----------------')


#===============================================================================================================


def forward_pass(x):
    global n_y  # Usa a lista global que armazena as saídas dos neurônios

    # Calcula a saída do neurônio 0:
    # Faz o produto escalar entre os pesos do neurônio 0 e o vetor de entrada x,
    # e aplica a função de ativação tangente hiperbólica (tanh)
    n_y[0] = np.tanh(np.dot(n_w[0], x))  # Neurônio 0 (camada oculta)

    # Faz o mesmo para o neurônio 1
    n_y[1] = np.tanh(np.dot(n_w[1], x))  # Neurônio 1 (camada oculta)

    # Monta as entradas para o neurônio da camada de saída (neurônio 2):
    # Usa 1.0 como bias, seguido das saídas dos neurônios 0 e 1
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])

    # Calcula o valor de entrada (z) do neurônio 2 (camada de saída)
    z2 = np.dot(n_w[2], n2_inputs)

    # Aplica a função de ativação sigmoide à entrada z2
    # Isso gera a saída final da rede (valor entre 0 e 1)
    n_y[2] = 1.0 / (1.0 + np.exp(-z2))  # Neurônio 2 (camada de saída)



#===============================================================================================================


def backward_pass(y_truth):
    global n_error  # Usa a lista global para armazenar os erros dos neurônios

    # Calcula a derivada da função de erro (loss):
    # É a diferença entre o valor previsto (n_y[2]) e o valor verdadeiro (y_truth),
    # multiplicado por -1 porque a derivada da função de erro quadrático é negativa.
    error_prime = -(y_truth - n_y[2])

    # Calcula a derivada da função de ativação sigmoide:
    # sigmoide'(z) = sigmoide(z) * (1 - sigmoide(z))
    derivative = n_y[2] * (1.0 - n_y[2])

    # Calcula o erro do neurônio da camada de saída (neuron 2)
    n_error[2] = error_prime * derivative

    # Calcula a derivada da função de ativação tanh para o neurônio 0:
    # tanh'(x) = 1 - tanh²(x)
    derivative = 1.0 - n_y[0] ** 2

    # Calcula o erro do neurônio 0 (camada oculta),
    # propagando o erro do neurônio 2 de volta,
    # multiplicado pelo peso da conexão de n_y[0] para n_y[2] e pela derivada da tanh
    n_error[0] = n_w[2][1] * n_error[2] * derivative

    # Faz o mesmo para o neurônio 1 (camada oculta)
    derivative = 1.0 - n_y[1] ** 2
    n_error[1] = n_w[2][2] * n_error[2] * derivative


#===============================================================================================================


def adjust_weights(x):
    global n_w  # Usa a lista global que armazena os pesos dos neurônios

    # Ajusta os pesos do neurônio 0 da camada oculta:
    # x contém as entradas (incluindo o bias em x[0])
    # A fórmula é: w = w - (entrada * taxa_aprendizado * erro)
    n_w[0] -= (x * LEARNING_RATE * n_error[0])

    # Ajusta os pesos do neurônio 1 da camada oculta (mesmo raciocínio)
    n_w[1] -= (x * LEARNING_RATE * n_error[1])

    # Prepara as entradas para o neurônio da camada de saída:
    # a entrada é um vetor com bias (1.0) e as saídas dos dois neurônios ocultos
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])

    # Ajusta os pesos do neurônio da camada de saída:
    # Aqui usamos as saídas dos neurônios ocultos como entrada para o neurônio final
    n_w[2] -= (n2_inputs * LEARNING_RATE * n_error[2])


#===============================================================================================================


# Loop de treinamento da rede
all_correct = False  # Inicialmente, assumimos que nem todos os padrões estão corretos
while not all_correct:  # Continua o treinamento até que todos os padrões estejam classificados corretamente
    all_correct = True  # Vamos assumir que nesta rodada todos serão classificados corretamente

    # Embaralha a ordem dos exemplos de treinamento para evitar que a rede aprenda por repetição de ordem
    np.random.shuffle(index_list)

    # Para cada exemplo embaralhado:
    for i in index_list:
        forward_pass(x_train[i])           # Propagação direta: calcula a saída da rede
        backward_pass(y_train[i])          # Propagação reversa: calcula os erros
        adjust_weights(x_train[i])         # Ajusta os pesos com base no erro
        show_learning()                    # Exibe os pesos atualizados (para debug/observação)

    # Após uma rodada de treinamento completa, verificamos se a rede aprendeu corretamente
    for i in range(len(x_train)):
        forward_pass(x_train[i])  # Faz uma nova previsão com os pesos atualizados

        # Exibe a entrada e a saída do neurônio final
        print('x1 =', '%4.1f' % x_train[i][1], ', x2 =',
              '%4.1f' % x_train[i][2], ', y =',
              '%.4f' % n_y[2])

        # Verifica se houve erro de classificação (usando 0.5 como limiar)
        if (((y_train[i] < 0.5) and (n_y[2] >= 0.5)) or
                ((y_train[i] >= 0.5) and (n_y[2] < 0.5))):
            all_correct = False  # Se algum padrão foi classificado errado, continuamos treinando


#===============================================================================================================