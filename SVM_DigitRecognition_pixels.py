# Digit Recognition


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import gc

from sklearn.preprocessing import scale

digits = pd.read_csv('C:\\Users\\Sasi\\Anaconda3\\MyFiles\\\SVM\SVM\\train.csv')


numberfour = digits.iloc[3, 1:]
numberfour.shape

numberfour = numberfour.values.reshape(28,28)
plt.imshow(numberfour)

print(numberfour[5:-5,5:-5])

digits.label.astype('category').value_counts()


percentageEachDigit = 100* (digits.label.astype('category').value_counts()/len(digits.index))
print(percentageEachDigit)


digits.isnull().sum()


#Seperate dependent and independent variables
X=digits.iloc[:, 1:]
Y=digits.iloc[:, 0]

#rescale the data 
X=scale(X)
#split the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.10, random_state = 101)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#Model building
from sklearn import svm
from sklearn import metrics

# first we will apply linear SVM
svm_linear = svm.SVC(kernel = 'linear')
svm_linear.fit(x_train, y_train)


#Predict
predictions = svm_linear.predict(x_test)
prediction[:10]


#Finding Confusion Matrix
confusion = metrics.confusion_matrix(y_true=y_test, y_pred=predictions)
confusion

#Finding Accuracy

metrics.accuracy_score(y_true=y_test, y_pred=predictions)

#classwise accuracy report
class_wise = metrics.classification_report(y_true=y_test, y_pred=predictions)
print(class_wise)


#run garbage collector to free up memory
gc.collect()


# Model building
# Now we will apply non-linear SVM - RBF kernel
svm_rbf = svm.SVC(kernel = 'rbf')
svm_rbf.fit(x_train, y_train)


#prediction
predictions= svm_rbf.predict(x_test)
predctions[:10]

#accuracy
metrics.accuracy_score(y_true=y_test, y_pred=predictions)


# GRID SEARCH CROSS VALIDATION
# to optimize value of cost C and choice of kernel gamma
from sklearn.model_selection import GridSearchCV

parameters = {'C':[1, 10, 100],
              'gamma': [1e-2, 1e-3, 1e-4]}

# instantiate a model
svc_grid_search = svm.SVC(kernel="rbf")

# create a classifier to perform grid search
clf = GridSearchCV(svc_grid_search, param_grid=parameters, scoring='accuracy')

#fit
clf.fit(x_train, y_train)

#results

cv_results = pd.DataFrame(clf.cv_results_)
cv_results

# PLOTS
cv_results['param_C'] = cv_results['param_C'].astype('int')

# plotting
plt.figure(figsize=(16,6))

# subplot 1/3
plt.subplot(131)
gamma_01 = cv_results[cv_results['param_gamma']==0.01]
plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='lower right')
plt.xscale('log')
          
# subplot 2/3
plt.subplot(132)
gamma_001 = cv_results[cv_results['param_gamma']==0.001]
plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='lower right')
plt.xscale('log')

# subplot 3/3
plt.subplot(133)
gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]
plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])
plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.0001")
plt.ylim([0.60, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='lower right')
plt.xscale('log')

plt.show()

# FINAL MODEL

#optimal parameters from the plots gamma = 0.001 and C =1
best_C = 1
best_gamma = 0.001

# model
svm_final = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma)

# fit
svm_final.fit(x_train, y_train)

# predict
predictions = svm_final.predict(x_test)

# evaluation: CM
confusion = metrics.confusion_matrix(y_true = y_test, y_pred = predictions)

# measure accuracy
test_accuracy = metrics.accuracy_score(y_true=y_test, y_pred=predictions)
print(test_accuracy,"\n")
print(confusion)


