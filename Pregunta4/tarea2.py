import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmread
import sklearn.linear_model as lm
X_train = csr_matrix(mmread('train.x.nm'))
y_train = np.loadtxt('train.y.dat')
X_dev = csr_matrix(mmread('dev.x.nm'))
y_dev = np.loadtxt('dev.y.dat')
X_test = csr_matrix(mmread('test.x.nm'))
y_test = np.loadtxt('test.y.dat')
model_lasso = lm.Lasso(alpha = 0.5, max_iter=1500, tol=0.001)
model = lm.LinearRegression(fit_intercept = False)
model.fit(X_train,y_train)
model_lasso.fit(X_train, y_train)

print "R2=%f"%model.score(X_test, y_test)
print "R2=%f"%model_lasso.score(X_test, y_test)