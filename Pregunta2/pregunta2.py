import pandas as pd
import numpy as np
import sklearn.linear_model as lm
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

def fss(x, y, names_x, test_data, k=10000):
    p = x.shape[1] - 1
    k = min(p, k)
    names_x = np.array(names_x)
    remaining = range(0, p)
    selected = [p]
    training_error = []
    test_error = []
    current_score = 0.0
    best_new_score = 0.0
    while remaining and len(selected) <= k:
        score_candidates = []
        for candidate in remaining:
            #se prepara el modelo con datos de entrenamiento
            model = lm.LinearRegression(fit_intercept=False)
            indexes = selected + [candidate]
            X_train = x[:, indexes]
            model.fit(X_train, y)
            predicted_value_train = model.predict(X_train)
            residual_value_train = predicted_value_train - y
            X_test, y_test = test_data
            X_test = X_test[:, indexes]
            predicted_value_test = model.predict(X_test)
            residual_value_test = predicted_value_test - y_test
            #Se calculan el error cuadratico medio
            mse_train = np.mean(np.power(residual_value_train, 2))
            mse_test = np.mean(np.power(residual_value_test, 2))
            var = (mse_train * X_train.shape[0]) / (X_train.shape[0] - X_train.shape[1] - 1)
            diag_values = np.diag(np.linalg.pinv(np.dot(X_train.T, X_train)))
            #Se obtienen los Zscore
            z_score = np.divide(model.coef_, np.sqrt(np.multiply(var, diag_values)))
            z_score_candidate = z_score[-1]
            score_candidates.append(( mse_train, mse_test, candidate))
        #Se ordena y se agregan los mejores candidatos a la lista seleccionados
        score_candidates.sort()
        z_scor, best_new_score_, mse_test_, best_candidate = score_candidates.pop()
        remaining.remove(best_candidate)
        selected.append(best_candidate)
        training_error.append((len(selected), best_new_score_))
        test_error.append((len(selected), mse_test_))
        print "selected= %s..." % names_x[best_candidate]
        print "totalvars=%d, mse = %f" % (len(indexes), best_new_score_)
    return selected, training_error, test_error

def bss(x, y, names_x, test_data, k=10000):
    p = x.shape[1] - 1
    k = min(p, k)
    names_x = np.array(names_x)
    selected = range(0, p)
    removed = []
    points_training = []
    points_test = []
    while len(selected) >= 1:
        #Se prepara el modelo con datos de entrenamiento
        model = lm.LinearRegression(fit_intercept=False)
        indexes = selected + [p]
        X_train = x[:, indexes]
        model.fit(X_train, y)
        predicted_value_train = model.predict(X_train)
        residual_value_train = predicted_value_train - y
        X_test, y_test = test_data
        X_test = X_test[:, indexes]
        predicted_value_test = model.predict(X_test)
        residual_value_test = predicted_value_test - y_test
        #Se calculan los errores cuadrados medios
        mse_train = np.mean(np.power(residual_value_train, 2))
        mse_test = np.mean(np.power(residual_value_test, 2))
        var = (mse_train * X_train.shape[0]) / (X_train.shape[0] - X_train.shape[1] - 1)
        diag_values = np.diag(np.linalg.pinv(np.dot(X_train.T, X_train)))
        #Se obtienen los Zscore
        z_score = np.divide(model.coef_, np.sqrt(np.multiply(var, diag_values)))
        score_candidates = zip(np.abs(z_score), indexes)
        score_candidates.sort(reverse=True)
        worst_new_z_score, worst_candidate = score_candidates.pop()
        selected.remove(worst_candidate)
        removed.append(worst_candidate)
        points_training.append((len(indexes), mse_train))
        points_test.append((len(indexes), mse_test))
        print "selected= %s..." % names_x[worst_candidate]
        print "totalvars=%d, mse = %f" % (len(indexes), mse_train)
    return removed, points_training, points_test
#Se prepara la muestra
url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'
df = pd.read_csv(url, sep='\t', header=0)
df = df.drop('Unnamed: 0', axis=1)
istrain_str = df['train']
istrain = np.asarray([True if s == 'T' else False for s in istrain_str])
istest = np.logical_not(istrain)
df = df.drop('train', axis=1)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_scaled['lpsa'] = df['lpsa']
X = df_scaled.ix[:, :-1]
N = X.shape[0]
X.insert(X.shape[1], 'intercept', np.ones(N))
y = df_scaled['lpsa']
Xtrain = X[istrain]
ytrain = y[istrain]
Xtest = X[np.logical_not(istrain)]
ytest = y[np.logical_not(istrain)]
Xm_train = Xtrain.as_matrix()
ym_train = ytrain.as_matrix()
Xm_test = Xtest.as_matrix()
ym_test = ytest.as_matrix()
#Se plotean los resultados de FSS
names_regressors = ["Lcavol", "Lweight", "Age", "Lbph", "Svi", "Lcp", "Gleason", "Pgg45"]
selected, points_training, points_test = fss(Xm_train, ym_train, names_regressors, (Xm_test, ym_test))
x_prueba= []
y_prueba= []
for x in points_test:
    x_prueba.append(x[0])
    y_prueba.append(x[1])
x_entrenamiento=[]
y_entrenamiento=[]
for x in points_training:
    x_entrenamiento.append(x[0])
    y_entrenamiento.append(x[1])
plt.plot(x_entrenamiento, y_entrenamiento, label='Set de entrenamiento')
plt.plot(x_prueba, y_prueba, label='Set de prueba')
plt.xlim(max(x_entrenamiento), min(x_entrenamiento))
plt.legend()
plt.show()


#Se plotean los resultados de bss
removed, points_training, points_test = bss(Xm_train, ym_train, names_regressors, (Xm_test, ym_test))
x_prueba_2= []
y_prueba_2= []
for x in points_test:
    x_prueba_2.append(x[0])
    y_prueba_2.append(x[1])
x_entrenamiento_2=[]
y_entrenamiento_2=[]
for x in points_training:
    x_entrenamiento_2.append(x[0])
    y_entrenamiento_2.append(x[1])
plt.plot(x_entrenamiento_2, y_entrenamiento_2, label='Set de entrenamiento')
plt.plot(x_prueba_2, y_prueba_2, label='Set de prueba')
plt.xlim(max(x_entrenamiento_2), min(x_entrenamiento_2))

plt.legend()
plt.show()
