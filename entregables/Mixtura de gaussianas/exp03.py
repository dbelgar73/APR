import sys
import math
import numpy as np
import pickle
from sklearn import mixture

if len(sys.argv) != 5:
    print('Usage: %s <trdata> <trlabels> <%%trper> <%%dvper>' % sys.argv[0])
    sys.exit(1)

X = np.load(sys.argv[1])['X']
xl = np.load(sys.argv[2])['xl']
trper = int(sys.argv[3])
dvper = int(sys.argv[4])

# Define los valores de K y rc que quieres probar
K_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12, 13, 14, 15, 16, 17 ,18, 19, 20] #es el que mejor funciona
rc_values = [0, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001] 

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
Ndv = round(dvper / 100 * N)
Xdv = X[N - Ndv:, :]
xldv = xl[N - Ndv:]

labs = np.unique(xltr).astype(int)
C = labs.shape[0]

N, D = Xtr.shape
M = Xdv.shape[0]

# Normalizar datos
mu = np.mean(Xtr, axis=0)
sigma = np.std(Xtr, axis=0)
sigma[sigma == 0] = 1
Xtr = (Xtr - mu) / sigma
Xdv = (Xdv - mu) / sigma
# Bucle para probar diferentes combinaciones de K y rc
for K in K_values:
    for rc in rc_values:
        print(f'Testing K={K}, rc={rc}')
        gtr = np.zeros((C, N))
        gdv = np.zeros((C, M))
        model = []

        for c, lab in enumerate(labs):
            Xtrc = Xtr[xltr == lab]
            Nc = Xtrc.shape[0]
            pc = Nc / N
            gmm = mixture.GaussianMixture(n_components=K, reg_covar=rc, random_state=23)
            gmm.fit(Xtrc)
            gtr[c] = math.log(pc) + gmm.score_samples(Xtr)
            gdv[c] = math.log(pc) + gmm.score_samples(Xdv)
            model.append((pc, gmm))

        idx = np.argmax(gtr, axis=0)
        etr = np.mean(np.not_equal(labs[idx], xltr)) * 100
        idx = np.argmax(gdv, axis=0)
        edv = np.mean(np.not_equal(labs[idx], xldv)) * 100

        print(f'Results: etr={etr:.2f}, edv={edv:.2f}')

        # Actualizar los mejores resultados si encontramos una combinación mejor
        if edv < best_edv:
            best_etr = etr
            best_edv = edv
            best_K = K
            best_rc = rc
            best_model = model

# Imprimir los mejores resultados
print('Mejor combinación de parámetros:')
print(f'  K      rc   etr   edv')
print(f'{best_K:3} {best_rc:3.1e} {best_etr:5.2f} {best_edv:5.2f}')

# Guardar el mejor modelo
filename = f'best_gmm_K{best_K}_rc{best_rc}.mod'
pickle.dump(best_model, open(filename, 'wb'))
