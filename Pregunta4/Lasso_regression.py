import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import mmread
import sklearn.linear_model as lm
from sklearn.linear_model import Ridge
#Se prepara la muestra y se cargan los dataset
X_train = csr_matrix(mmread('train.x.mm'))
y_train = np.loadtxt('train.y.dat')
X_dev = csr_matrix(mmread('dev.x.mm'))
y_dev = np.loadtxt('dev.y.dat')
X_test = csr_matrix(mmread('test.x.mm'))
y_test = np.loadtxt('test.y.dat')
model = lm.Lasso(fit_intercept=False)
model.set_params(alpha=2, max_iter=3500)
contador=0
x=[]
y=[]
# Se calcula un R^2 para distintos valores de alpha
while contador<3000:
	model.set_params(alpha=contador, max_iter=3000)
	model.fit(X_train, y_train)
	print "R2=%f" % model.score(X_test, y_test)
	x.append(contador)
	y.append(model.score(X_test, y_test))

	contador+=20
#SE plotean los resultados de R^2 en funcion de alpha
plt.plot(x, y, label='R2 en funcion de alpha')
plt.show()