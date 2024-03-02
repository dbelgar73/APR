import sys
import math
import numpy as np
import pickle
from sklearn import mixture
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

if len(sys.argv)!=3:
  print('Usage: %s <trdata> <trlabels>' % sys.argv[0]);
  sys.exit(1);

X= np.load(sys.argv[1])['X'];
xl=np.load(sys.argv[2])['xl'];


K=17; #mejor parametro obtenido
rc=0.01; #mejor parametro obtenido
seed=23;

N=X.shape[0];
np.random.seed(seed); 
perm=np.random.permutation(N);
X=X[perm]; 
xl=xl[perm];

# Normalise data
mu=np.mean(X,axis=0);
sigma=np.std(X,axis=0);
sigma[sigma==0]=1;
X=(X-mu)/sigma;

# Definir número de divisiones para la validación cruzada
num_splits = 5
skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)


# Initialize variables for cross-validation results
etr_list = []
edv_list = []

# Loop over the cross-validation splits
for train_index, dev_index in skf.split(X, xl):
    print("TRAINING.................. \n")
    X_train, X_dev = X[train_index], X[dev_index]
    xl_train, xl_dev = xl[train_index], xl[dev_index]

    # Create and fit the Gaussian Mixture Model
    model = mixture.GaussianMixture(n_components=K, reg_covar=rc, random_state=seed)
    model.fit(X_train)

    # Predict on the training and validation sets
    y_train_pred = model.predict(X_train)
    y_dev_pred = model.predict(X_dev)

    # Calculate accuracy (you might need to use a different metric depending on your problem)
    etr = accuracy_score(xl_train, y_train_pred)
    edv = accuracy_score(xl_dev, y_dev_pred)

    # Append results to the lists
    etr_list.append(etr)
    edv_list.append(edv)

# Calculate average training and validation errors
etr = np.mean(etr_list)
edv = np.mean(edv_list)

# Print the results
print('  K      rc   etr   edv')
print('--- ------- ----- -----')
print(f'{K:3} {rc:3.1e} {etr:5.2f} {edv:5.2f}')

# Save the model to a file
filename = 'gmm.K1.rc0.1.mod'
pickle.dump(model, open(filename, 'wb'))

print('  K      rc   etr   edv')
print('--- ------- ----- -----')
print(f'{K:3} {rc:3.1e} {etr:5.2f} {edv:5.2f}');

filename = 'gmm.K17.rc0.01.mod'
pickle.dump(model, open(filename, 'wb'))
