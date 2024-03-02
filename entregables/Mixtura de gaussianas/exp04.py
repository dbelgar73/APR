import sys
import numpy as np
import pickle
from sklearn import mixture
from sklearn.model_selection import cross_val_score, KFold

if len(sys.argv) != 5:
    print('Usage: %s <trdata> <trlabels> <%%trper> <%%dvper>' % sys.argv[0])
    sys.exit(1)

X = np.load(sys.argv[1])['X']
xl = np.load(sys.argv[2])['xl']
trper = int(sys.argv[3])
dvper = int(sys.argv[4])

# Define los valores de K y rc que quieres probar
K_values = [1, 2, 3]  # Puedes agregar más valores según sea necesario
rc_values = [0.1, 0.01, 0.001]  # Puedes ajustar estos valores también

best_etr = float('inf')  # Inicializa con un valor alto para encontrar el mínimo
best_edv = float('inf')
best_K = None
best_rc = None
best_model = None

N = X.shape[0]
np.random.seed(23)
perm = np.random.permutation(N)
X = X[perm]
xl = xl[perm]

Ntr = round(trper / 100 * N)
Xtr = X[:Ntr, :]
xltr = xl[:Ntr]

labs = np.unique(xltr).astype(int)
C = labs.shape[0]

# Normalizar datos
mu = np.mean(Xtr, axis=0)
sigma = np.std(Xtr, axis=0)
sigma[sigma == 0] = 1
Xtr = (Xtr - mu) / sigma

kf = KFold(n_splits=5, shuffle=True, random_state=23)  # 5-fold cross-validation

# Bucle para probar diferentes combinaciones de K y rc
for K in K_values:
    for rc in rc_values:
        print(f'Testing K={K}, rc={rc}')
        gmm = mixture.GaussianMixture(n_components=K, reg_covar=rc, random_state=23)

        # Utilizar cross_val_score para obtener los errores de entrenamiento y desarrollo
        etr_scores = cross_val_score(gmm, Xtr, xltr, cv=kf, scoring='accuracy')
        edv_scores = cross_val_score(gmm, Xtr, xltr, cv=kf, scoring='accuracy')

        avg_etr = 100 * (1 - np.mean(etr_scores))  # Convertir la puntuación en tasa de error
        avg_edv = 100 * (1 - np.mean(edv_scores))

        print(f'Average Results: etr={avg_etr:.2f}, edv={avg_edv:.2f}')

        # Actualizar los mejores resultados si encontramos una combinación mejor
        if avg_edv < best_edv:
            best_etr = avg_etr
            best_edv = avg_edv
            best_K = K
            best_rc = rc
            best_model = gmm

# Imprimir los mejores resultados
print('Mejor combinación de parámetros:')
print(f'  K      rc   etr   edv')
print(f'{best_K:3} {best_rc:3.1e} {best_etr:5.2f} {best_edv:5.2f}')

# Entrenar el mejor modelo con todos los datos de entrenamiento
best_model.fit(Xtr)

# Guardar el mejor modelo
filename = f'best_gmm_K{best_K}_rc{best_rc}.mod'
pickle.dump(best_model, open(filename, 'wb'))
