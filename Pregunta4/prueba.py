
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Se cargan los datos
diabetes = datasets.load_diabetes()


# Se eligen las muestras de datos
diabetes_X = diabetes.data[:, np.newaxis]
diabetes_X_temp = diabetes_X[:, :, 2]

# Se dividen los datos en training/testing 
diabetes_X_train = diabetes_X_temp[:-30]
diabetes_X_test = diabetes_X_temp[-30:]

# Se divide la variable explicada en  training/testing 
diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

# Se crean los modelos lineales
regr = linear_model.LinearRegression()
regr2=linear_model.LassoCV(alpha=[0.1,0.2,0.5,1.0,3.0,5.0,10.0])
regr3=linear_model.RidgeCV(alphas=[0.1,0.2,0.5,1.0,3.0,5.0,10.0])

# Se entrenan los modelos
regr.fit(diabetes_X_train, diabetes_y_train)
regr2=linear_model.LassoCV(cv=20).fit(diabetes_X_train, diabetes_y_train)
regr3.fit(diabetes_X_train, diabetes_y_train)


print u'Regresión Mínimos Cuadrados Ordinarios'
# Coeficiente
print'Coeficientes:',regr.coef_
# MSE
print("Residual sum of squares: %.2f"
 % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Varianza Explicada
print('Varianza explicada: %.2f\n' % regr.score(diabetes_X_test, diabetes_y_test))

print u'Regresión Lasso' 
# Coeficientes
print'Coeficientes:', regr2.coef_
# MSE
print("Residual sum of squares: %.2f"
 % np.mean((regr2.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Varianza Explicada
print('Varianza explicada: %.2f\n' % regr2.score(diabetes_X_test, diabetes_y_test))

print u'Regresión Ridge'
# Coeficiente
print'Coeficientes:', regr3.coef_
# MSE
print("Residual sum of squares: %.2f"
 % np.mean((regr3.predict(diabetes_X_test) - diabetes_y_test) ** 2))
# Varianza Explicada
print('Varianza explicada: %.2f\n' % regr3.score(diabetes_X_test, diabetes_y_test))


# Plot 
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
 linewidth=3, label=u'Regresión MCO')
plt.plot(diabetes_X_test, regr2.predict(diabetes_X_test), color='yellow',
 linewidth=3, label=u'Regresión Lasso')
plt.plot(diabetes_X_test, regr3.predict(diabetes_X_test), color='green',
 linewidth=3, label=u'Regresión Ridge')
plt.title(u'Regresióneal por 3 metodos diferentes')
plt.legend()
plt.xticks(())
plt.yticks(())

plt.show()