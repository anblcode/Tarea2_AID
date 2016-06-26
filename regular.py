import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import sklearn.linear_model as lm
import numpy as np
from matplotlib import pyplot as plt
import math
from sklearn import cross_validation

#estilo de graficos
plt.style.use('ggplot')

#preparacion de datos analogo a lss.py
url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
df = pd.read_csv(url, sep='\t', header=0)
df = df.drop('Unnamed: 0', axis = 1)
istrain_str = df['train']
istrain = np.asarray([True if s == 'T' else False for s in istrain_str])
df = df.drop('train', axis=1)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
df_scaled['lpsa'] = df['lpsa']

X = df_scaled.ix[:,:-1]
N = X.shape[0]
X.insert(X.shape[1], 'intercept', np.ones(N))
y = df_scaled['lpsa']

		#Pregunta A
X = X.drop('intercept', axis=1)
Xtrain = X[istrain]
ytrain = y[istrain]
names_regressors = ["Lcavol", "Lweight", "Age", "Lpbh", "Svi", "Lcp", "Gleason", "Pgg45"]
alphas_ = np.logspace(4,-1,base=10)
coefs = []
model = Ridge(fit_intercept=True,solver='svd')
for a in alphas_:
	model.set_params(alpha=a)
	model.fit(Xtrain,ytrain)
	coefs.append(model.coef_)
ax = plt.gca()

for y_arr, label in zip(np.squeeze(coefs).T, names_regressors):
	plt.plot(alphas_, y_arr, label=label)
plt.legend()
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) #reverse axis
plt.xlabel('Alpha')
plt.ylabel('Weights')
plt.title('Regularization Path RIDGE')
plt.axis('tight')
plt.legend(loc=2)
plt.show()

		#Pregunta B

clf = Lasso(fit_intercept=True)
alphas_2 = np.logspace(1,-2,base=10)
coefs = []
for a in alphas_2:
	clf.set_params(alpha=a)
	clf.fit(Xtrain,ytrain)
	coefs.append(clf.coef_)
ax = plt.gca()

for y_arr, label in zip(np.squeeze(coefs).T, names_regressors):
	plt.plot(alphas_2, y_arr, label=label)
plt.legend()
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1]) #reverse axis
plt.xlabel('Alpha')
plt.ylabel('Weights')
plt.title('Regularization Path LASSO')
plt.axis('tight')
plt.legend(loc=2)
plt.show()


		#Pregunta C
Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]
alphas_3 = np.logspace(2,-2,base=10)
coefs = []
model2 = Ridge(fit_intercept=True)
mse_test = []
mse_train = []
for a in alphas_3:
	model2.set_params(alpha=a)
	model2.fit(Xtrain, ytrain)
	yhat_train = model2.predict(Xtrain)
	yhat_test = model2.predict(Xtest)
	mse_train.append(np.mean(np.power(yhat_train - ytrain, 2)))
	mse_test.append(np.mean(np.power(yhat_test - ytest, 2)))
ax = plt.gca()
ax.plot(alphas_3,mse_train,label='train error ridge')
ax.plot(alphas_3,mse_test,label='test error ridge')
plt.legend(loc=2)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()

		#Pregunta D
alphas_4 = np.logspace(2,-2,base=10)
coefs = []
clf2 = Lasso(fit_intercept=True)
mse_test = []
mse_train = []
for a in alphas_4:
	clf2.set_params(alpha=a)
	clf2.fit(Xtrain, ytrain)
	yhat_train = clf2.predict(Xtrain)
	yhat_test = clf2.predict(Xtest)
	mse_train.append(np.mean(np.power(yhat_train - ytrain, 2)))
	mse_test.append(np.mean(np.power(yhat_test - ytest, 2)))
ax = plt.gca()
ax.plot(alphas_4,mse_train,label='Train Error Lasso')
ax.plot(alphas_4,mse_test,label='Test Error Lasso')
plt.legend(loc=2)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])
plt.show()

		#Pregunta E
def MSE(y,yhat): return np.mean(np.power(y-yhat,2))

def best_param(tipoReg, alphas):
	print tipoReg
	Xm = Xtrain.as_matrix()
	ym = ytrain.as_matrix()
	k_fold = cross_validation.KFold(len(Xm),10)
	best_cv_mse = float("inf")

	if tipoReg == "Ridge":
		modelo = Ridge(fit_intercept = True)
	elif tipoReg == "Lasso":
		modelo = Lasso(fit_intercept = True)
	for a in alphas:
		modelo.set_params(alpha=a)
		mse_list_k10 = [
                    MSE(modelo.fit(Xm[train], ym[train]).predict(Xm[val]), ym[val])
                    for train, val in k_fold]
		if np.mean(mse_list_k10) < best_cv_mse:
			best_cv_mse = np.mean(mse_list_k10)
			best_alpha = a
			print "BEST PARAMETER=%f, MSE(CV)=%f"%(best_alpha, best_cv_mse)


best_param("Ridge",alphas_)
best_param("Lasso",alphas_2)

