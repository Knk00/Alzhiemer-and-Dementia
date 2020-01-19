# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
#oasis_cross = pd.read_csv('oasis_cross-sectional.csv')
oasis_long = pd.read_csv('oasis_longitudinal.csv')
oasis_long = oasis_long.drop(columns = [ 'Hand', 'MRI ID'])
oasis_long = oasis_long.set_index('Subject ID')

y = oasis_long['Group'].astype('category')
X = oasis_long.iloc[:, 1:]


from sklearn.preprocessing import LabelEncoder
le =  LabelEncoder()
X.iloc[:, 2] = le.fit_transform(X.iloc[:, 2])

from sklearn.impute import SimpleImputer
imputer_SES = SimpleImputer(missing_values=np.nan, strategy='median')
imputer_MMSE = SimpleImputer(missing_values=np.nan, strategy='mean')
X.iloc[:, 5:6] = imputer_SES.fit_transform(X.iloc[:, 5:6])   
X.iloc[:, 6:7] = imputer_MMSE.fit_transform(X.iloc[:, 6:7])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.svm import SVC
classifier = SVC(C = 10000, kernel= 'linear')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()

from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [1, 10, 100, 1000, 10000], 'kernel' : ['linear']},
               {'C' : [0.5, 1, 10, 100, 1000], 'kernel' : ['rbf', 'poly', 'sigmoid'],
                'gamma' : [0.07, 0.09, 0.1, 0.2, 0.4, 0.5]}]

grid_search = GridSearchCV(estimator=classifier, param_grid= parameters, n_jobs= -1,
                           scoring='accuracy', cv =10)
grid_search.fit(X_train, y_train)
best_acc = grid_search.best_score_
best_par = grid_search.best_params_





from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred, labels=['Demented', 'Nondemented', 'Converted'])
acc = accuracy_score(y_test, y_pred)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()