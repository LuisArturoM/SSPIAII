import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

# Ruta del dataset
file_path = "C:/Users/elcho/OneDrive/Escritorio/blocs notas y pdfs/Sem IA 2/zoo.csv"
column_names = ["animal_name", "hair", "feathers", "eggs", "milk", "airborne", "aquatic", 
                "predator", "toothed", "backbone", "breathes", "venomous", "fins", 
                "legs", "tail", "domestic", "catsize", "class_type"]

df = pd.read_csv(file_path, names=column_names)

X = df.drop(columns=["animal_name", "class_type"])
y = df["class_type"]

X = X.apply(pd.to_numeric, errors='coerce')

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.dropna(inplace=True)
y_train = y_train[X_train.index]

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("Classification Report for", name, ":")
    print(classification_report(y_test, y_pred))
    
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity (Recall)": recall,
        "F1 Score": f1
    }

results_df = pd.DataFrame(results).transpose()
print(results_df)