import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size + 1) * 2 - 1  # Random weights between -1 and 1
        self.learning_rate = learning_rate

    def activation(self, x):
        return 1 if x >= 0 else -1

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, inputs, targets, max_epochs=1000, stopping_criterion=0.01):
        for epoch in range(max_epochs):
            errors = 0
            for input_vector, target in zip(inputs, targets):
                prediction = self.predict(input_vector)
                error = target - prediction
                errors += int(error != 0)
                self.weights[1:] += self.learning_rate * error * input_vector
                self.weights[0] += self.learning_rate * error
            if errors == 0 or errors / len(targets) < stopping_criterion:
                print(f"Training finished after {epoch + 1} epochs")
                break

def read_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',')
    inputs = data[:, :-1]
    targets = data[:, -1]
    return inputs, targets

def plot_data(inputs, targets, perceptron=None):
    plt.figure()
    plt.scatter(inputs[:, 0], inputs[:, 1], c=targets, cmap='bwr')
    if perceptron is not None:
        xmin, xmax = plt.xlim()
        weights = perceptron.weights
        ymin = (-weights[0] - weights[1] * xmin) / weights[2]
        ymax = (-weights[0] - weights[1] * xmax) / weights[2]
        plt.plot([xmin, xmax], [ymin, ymax], color='black')
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Data and Decision Boundary')
    plt.show()

if __name__ == "__main__":
    # Ejercicio 1: Lectura de datos y entrenamiento del perceptrón
    inputs, targets = read_data("XORtrn.csv")
    perceptron = Perceptron(input_size=2)
    perceptron.train(inputs, targets)

    # Ejercicio 2: Prueba del perceptrón en datos reales
    test_inputs, test_targets = read_data("C:\Users\elcho\OneDrive\Escritorio\blocs notas y pdfs\Sem IA 2\XOR_tst.csv")
    accuracy = np.mean([perceptron.predict(test_input) == test_target for test_input, test_target in zip(test_inputs, test_targets)])
    print(f"Accuracy on test data: {accuracy * 100:.2f}%")

    # Ejercicio 3: Mostrar gráficamente los patrones utilizados y la recta que los separa
    plot_data(inputs, targets, perceptron)