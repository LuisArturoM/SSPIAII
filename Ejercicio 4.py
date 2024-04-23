import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut, KFold
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Cargar los datos
data = pd.read_csv('C:/Users/elcho/OneDrive/Escritorio/blocs notas y pdfs/Sem IA 2/irisbin.csv', header=None)
# Asignar nombres de columnas
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species1', 'species2', 'species3']

# Convertir etiquetas a nombres de especies
def map_species(row):
    if row['species1'] == 1:
        return 'virginica'
    elif row['species2'] == 1:
        return 'versicolor'
    elif row['species3'] == 1:
        return 'setosa'

data['species'] = data.apply(map_species, axis=1)

# Contar el número de instancias de cada especie
species_counts = data['species'].value_counts()

# Obtener las especies y sus recuentos
species = species_counts.index.tolist()
counts = species_counts.values.tolist()

# Separar características y etiquetas
X = data.iloc[:, :-4].values  # Tomamos solo las características
y = data['species'].values

# Realizar PCA para reducir a 2 dimensiones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Graficar la distribución de las clases en dos dimensiones
plt.figure(figsize=(8, 6))
for species in np.unique(y):
    plt.scatter(X_pca[y == species, 0], X_pca[y == species, 1], label=species)
plt.title('Distribucion')
plt.xlabel('C1')
plt.ylabel('C2')
plt.legend()
plt.show()

# Separar características y etiquetas
X = data.iloc[:, :-3].values
y = data.iloc[:, -3:].values

# Codificar las etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.argmax(y, axis=1))

# Normalizar los datos
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42)

# Configurar el MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', max_iter=2000, random_state=42)

# Entrenar el clasificador
mlp.fit(X_train, y_train)

# Validar con leave-one-out
loo = LeaveOneOut()
loo_scores = []
for train_index, test_index in loo.split(X_normalized):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    loo_scores.append(accuracy_score(y_test, y_pred))

# Validar con leave-k-out (k=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
k_scores = []
for train_index, test_index in kf.split(X_normalized):
    X_train, X_test = X_normalized[train_index], X_normalized[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    k_scores.append(accuracy_score(y_test, y_pred))

# Calcular estadísticas
mean_loo = np.mean(loo_scores)
std_loo = np.std(loo_scores)
mean_k = np.mean(k_scores)
std_k = np.std(k_scores)

# Mostrar resultados
print("Leave-One-Out:")
print("Mean Accuracy:", mean_loo)
print("Standard Deviation:", std_loo)
print("\nLeave-K-Out (k=5):")
print("Mean Accuracy:", mean_k)
print("Standard Deviation:", std_k)

