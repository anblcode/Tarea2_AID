import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.linear_model as lm
from sklearn import cross_validation
import pylab
import scipy.stats as stats

		#Pregunta A
url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
df = pd.read_csv(url, sep='\t', header=0)
#elimina la columna con el numero de registro
df = df.drop('Unnamed: 0', axis = 1)
#obtiene los datos de la columna que dice es dato de entrenamiento o de test
istrain_str = df['train']
#pasa la columna de entrenamiento a array, si es de train True, si es test False
istrain = np.asarray([True if s == 'T' else False for s in istrain_str])
#elimina del dataframe columna anteriormente guardada
df = df.drop('train', axis=1)

		#Pregunta B
print '\nDimensiones DataFrame:'
print df.shape
print '\nInfo Dataframe:'
df.info()
print '\nDescripcion DataFrame:'
print df.describe()

		#Pregunta C
scaler = StandardScaler()
#deja la media nula y la varianza unitaria
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
df_scaled['lpsa'] = df['lpsa']

		#pregunta D
#deja en X la data ya preparada, sin la columna lpsa
X = df_scaled.ix[:,:-1]
#guarda el numero de datos TOTALES
N = X.shape[0]
X.insert(0, 'intercept', np.ones(N))
#guarda en y la data de lpsa, nuestra "respuesta"
y = df_scaled['lpsa']
#almacena datos de entrenamiento, predictores
Xtrain = X[istrain]
#almacena datos de entrenamiento, respuesta lpsa
ytrain = y[istrain]
#analogo a lo anterior, pero con datos de test
Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]
#regresion por minimos cuadrados, sin intercepto (datos ya centrados)
linreg = lm.LinearRegression(fit_intercept = False)
#ajuste de modelo lineal con datos de entrenamiento
linreg.fit(Xtrain,ytrain)
#guarda el numero de datos de entrenamiento
Naux = Xtrain.shape[0]

		#pregunta E
#matriz x
xx = Xtrain.as_matrix()
#(xT*x)^-1
b1 = np.linalg.inv(np.mat(xx.T)*np.mat(xx))
#(xT*X)^-1*xT
b2 = np.mat(b1)*np.mat(xx.T)
#(xT*X)^-1*xT*y
b_hat = np.squeeze(np.asarray(b2.dot(ytrain)))
print '\nb gorro'
print b_hat
#se genera el predictor con datos entrenamiento
yhat_train = linreg.predict(Xtrain)
#se obtiene el mean squared error para datos entrenamiento
mse_train = np.mean(np.power(yhat_train - ytrain, 2))
print '\nMSE TRAIN'
print mse_train
#sigma cuadrado utiliza parte del mse, se quita promedio
mse_aux = Naux*mse_train
sigmaCuad = mse_aux/(Naux-Xtrain.shape[1]-1)
#diagonal de la matriz (xT*x)^-1, coeficientes vj
diag = np.diag(b1)

z_score = []
for j in range(0, Xtrain.shape[1]):
	#se utiliza la formula dada en clases
	z_score.append(b_hat[j]/(np.power(sigmaCuad*diag[j],0.5)))
print '\nZ-Score para datos de entrenamiento:'
print z_score

		#pregunta F
#se genera el predictor con datos de prueba
yhat_test = linreg.predict(Xtest)
#se obtiene el mean squared error para datos prueba
mse_test = np.mean(np.power(yhat_test - ytest, 2))
print '\nMSE TEST'
print mse_test
Xm = Xtrain.as_matrix()
ym = ytrain.as_matrix()


ies = [5,10]
for i in ies:
	mse_cv = 0
	k_fold = cross_validation.KFold(len(Xm),i)
	
	for (train, val) in k_fold:
		linreg = lm.LinearRegression(fit_intercept = False)
		linreg.fit(Xm[train], ym[train])
		yhat_val = linreg.predict(Xm[val])
		mse_fold = np.mean(np.power(yhat_val - ym[val], 2))
		mse_cv += mse_fold
	mse_cv = mse_cv / i
	print ('MSE Cross Validation K=',i)
	print mse_cv

		#Pregunta J
#errores de prediccion para datos de entrenamiento
errors = yhat_train - ytrain
stats.probplot(errors,dist="norm",plot=pylab)
pylab.show()



