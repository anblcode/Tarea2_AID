import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmread
import sklearn.linear_model as lm
from sklearn.linear_model import Ridge
#Se prepara la muestra y se cargan los dataset

X_train = csr_matrix(mmread('train.x.nm'))
y_train = np.loadtxt('train.y.dat')
X_dev = csr_matrix(mmread('dev.x.nm'))
y_dev = np.loadtxt('dev.y.dat')
X_test = csr_matrix(mmread('test.x.nm'))
y_test = np.loadtxt('test.y.dat')
#Se entrena el modelo
model = lm.Lasso(fit_intercept=False)
model.set_params(alpha=0.5, max_iter=1500)
model.fit(X_train, y_train)
#Se obtiene R^2
print "R2=%f" % model.score(X_test, y_test)