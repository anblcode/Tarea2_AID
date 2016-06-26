import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmread
import sklearn.linear_model as lm

data_dir = "./ngrams-deprels-origin.runtime.budget.numscreen.ratings.seasons/"

print "importando X..."
X = csr_matrix( mmread('train.x.nm'))
print "importando y..."
y = np.loadtxt('train.y.dat')

'''
model = lm.LinearRegression(fit_intercept = True)
model.fit (X,y)
print "R2=%f"%model.score(X, y)
'''
model = lm.Lasso(alpha=1.0, max_iter=500, tol=0.001)
model.fit (X,y)
print "R2=%f"%model.score(X, y)



print "importando X_val..."
X_validation = csr_matrix( mmread('dev.x.nm'))
print "importando y_val..."
y_validation = np.loadtxt('dev.y.dat')


#testing

# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((model.predict(X_validation) - y_validation) ** 2))



print "importando X_test..."
X_test = csr_matrix( mmread('test.x.nm'))
print "importando y_test..."
y_test = np.loadtxt('test.y.dat')

print "R2=%f"%model.score(X_test, y_test)
