import pickle
import numpy as np
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Importa el tipo de modelo que estás usando desde scikit-learn
# Por ejemplo, si estás usando un modelo de clasificación, podrías tener algo como:
# from sklearn.ensemble import RandomForestClassifier

def cargar_modelo():
    # Cargar el modelo guardado
    model_filename = 'gmm.K17.rc0.01.mod'
    with open(model_filename, 'rb') as model_file:
        model = pickle.load(model_file)
    return model


def cargar_datos_y_etiquetas():
    X = np.load(sys.argv[1])['X']      
    xl = np.load(sys.argv[2])['xl']
    N = X.shape[0]
  
    labs = np.unique(xl).astype(int)
    C = labs.shape[0]
    return X, xl
    
def evaluar_modelo(modelo, datos, etiquetas):
    predicciones = modelo.predict(datos)
    accuracy = accuracy_score(etiquetas, predicciones)
    return accuracy

def main():
    # Verifica si se proporcionan suficientes argumentos de línea de comandos
    if len(sys.argv) != 3:
        print("Uso: python script.py <ruta_modelo_pickle> <ruta_datos_etiquetas>")
        sys.exit(1)

    # Cargar modelo desde el archivo pickle
    modelo = cargar_modelo()
    
    datos, etiquetas = cargar_datos_y_etiquetas()
    # Evaluar el modelo
    accuracy = evaluar_modelo(modelo, datos, etiquetas)
    print("Accuracy del modelo:", accuracy)

if __name__ == "__main__":
    main()
