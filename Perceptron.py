import numpy as np

class Perceptron:
    def __init__(self, num_inputs, threshold=100, learning_rate=0.01):
        self.weights = np.zeros(num_inputs + 1)  # +1 para o bias
        self.threshold = threshold
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return 1 if summation > 0 else 0

    def train(self, training_inputs, labels):
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
def generate_data(func, num_inputs):
    # Gera todas as combinações possíveis de entradas booleanas
    inputs = np.array(np.meshgrid(*[[0, 1]] * num_inputs)).T.reshape(-1, num_inputs)
    labels = np.array([func(*input_row) for input_row in inputs])
    return inputs, labels

# Funções AND e OR
and_func = lambda *inputs: int(all(inputs))
xor_func = lambda x, y: int(x != y)
or_func = lambda *inputs: int(any(inputs))

# Exemplo para 3 entradas
inputs, labels = generate_data(and_func, 3)
perceptron = Perceptron(num_inputs=3)
perceptron.train(inputs, labels)

# Testando AND
for input_row in inputs:
    output = perceptron.predict(input_row)
    print(f"Input: {input_row}, Output AND: {output}")

# Exemplo para 4 entradas
inputs, labels = generate_data(or_func, 4)
perceptron = Perceptron(num_inputs=4)
perceptron.train(inputs, labels)
# Testando OR
for input_row in inputs:
    output = perceptron.predict(input_row)
    print(f"Input: {input_row}, Output OR: {output}")


# Testando XOR(não funciona)
inputs, labels = generate_data(xor_func, 2)
perceptron = Perceptron(num_inputs=2)
perceptron.train(inputs, labels)

for input_row in inputs:
    output = perceptron.predict(input_row)
    print(f"Input: {input_row}, Output XOR: {output}")
