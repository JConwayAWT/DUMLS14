from sklearn.externals import joblib
from sklearn.datasets import load_svmlight_file
import numpy as np
import winsound
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import accuracy_score


def main():
  training_data, training_labels = load_svmlight_file("dt1.trn.svm")
  validation_data, validation_labels = load_svmlight_file("dt1.vld.svm")

  labels_that_are_1 = []
  labels_that_are_2 = []
  labels_that_are_3 = []
  labels_that_are_4 = []
  labels_that_are_5 = []

  for element in training_labels:
    if element == 1:
      labels_that_are_1.append(1)
      labels_that_are_2.append(0)
      labels_that_are_3.append(0)
      labels_that_are_4.append(0)
      labels_that_are_5.append(0)
    elif element == 2:
      labels_that_are_1.append(0)
      labels_that_are_2.append(1)
      labels_that_are_3.append(0)
      labels_that_are_4.append(0)
      labels_that_are_5.append(0)
    elif element == 3:
      labels_that_are_1.append(0)
      labels_that_are_2.append(0)
      labels_that_are_3.append(1)
      labels_that_are_4.append(0)
      labels_that_are_5.append(0)
    elif element == 4:
      labels_that_are_1.append(0)
      labels_that_are_2.append(0)
      labels_that_are_3.append(0)
      labels_that_are_4.append(1)
      labels_that_are_5.append(0)
    elif element == 5:
      labels_that_are_1.append(0)
      labels_that_are_2.append(0)
      labels_that_are_3.append(0)
      labels_that_are_4.append(0)
      labels_that_are_5.append(1)

  validation_labels_that_are_1 = []
  validation_labels_that_are_2 = []
  validation_labels_that_are_3 = []
  validation_labels_that_are_4 = []
  validation_labels_that_are_5 = []

  for element in validation_labels:
    if element == 1:
      validation_labels_that_are_1.append(1)
      validation_labels_that_are_2.append(0)
      validation_labels_that_are_3.append(0)
      validation_labels_that_are_4.append(0)
      validation_labels_that_are_5.append(0)
    elif element == 2:
      validation_labels_that_are_1.append(0)
      validation_labels_that_are_2.append(1)
      validation_labels_that_are_3.append(0)
      validation_labels_that_are_4.append(0)
      validation_labels_that_are_5.append(0)
    elif element == 3:
      validation_labels_that_are_1.append(0)
      validation_labels_that_are_2.append(0)
      validation_labels_that_are_3.append(1)
      validation_labels_that_are_4.append(0)
      validation_labels_that_are_5.append(0)
    elif element == 4:
      validation_labels_that_are_1.append(0)
      validation_labels_that_are_2.append(0)
      validation_labels_that_are_3.append(0)
      validation_labels_that_are_4.append(1)
      validation_labels_that_are_5.append(0)
    elif element == 5:
      validation_labels_that_are_1.append(0)
      validation_labels_that_are_2.append(0)
      validation_labels_that_are_3.append(0)
      validation_labels_that_are_4.append(0)
      validation_labels_that_are_5.append(1)

  training_labels_1 = np.array(labels_that_are_1)
  training_labels_2 = np.array(labels_that_are_2)
  training_labels_3 = np.array(labels_that_are_3)
  training_labels_4 = np.array(labels_that_are_4)
  training_labels_5 = np.array(labels_that_are_5)

  validation_labels_1 = np.array(validation_labels_that_are_1)
  validation_labels_2 = np.array(validation_labels_that_are_2)
  validation_labels_3 = np.array(validation_labels_that_are_3)
  validation_labels_4 = np.array(validation_labels_that_are_4)
  validation_labels_5 = np.array(validation_labels_that_are_5)

  classifier_1 = classify_1(training_data, training_labels_1)
  classifier_2 = classify_2(training_data, training_labels_2)
  classifier_3 = classify_3(training_data, training_labels_3)
  classifier_4 = classify_4(training_data, training_labels_4)
  classifier_5 = classify_5(training_data, training_labels_5)

  validation_predictions = []

  for index in range(validation_data.shape[0]):
    print 'hello, world'
    best_prediction = get_highest_likelihood_prediction(classifier_1,classifier_2,classifier_3,classifier_4,classifier_5,validation_data, index)
    validation_predictions.append(best_prediction)

##    for linear_svc
##    best_prediction = get_maximum_prediction_value(classifier_1,classifier_2,classifier_3,classifier_4,classifier_5,validation_data,index)
##    validation_predictions.append(best_prediction)

  validation_predictions = np.array(validation_predictions)

  print accuracy_score(validation_labels,validation_predictions)

def get_highest_likelihood_prediction(clf1, clf2, clf3, clf4, clf5, validation_data, index):
  a = clf1.predict_proba(validation_data[index])[0][1]
  b = clf2.predict_proba(validation_data[index])[0][1]
  c = clf3.predict_proba(validation_data[index])[0][1]
  d = clf4.predict_proba(validation_data[index])[0][1]
  e = clf5.predict_proba(validation_data[index])[0][1]
  possibilities = [a,b,c,d,e]
  for index in range(len(possibilities)):
    if np.isnan(possibilities[index]):
      possibilities[index] = 1
  best = max(possibilities)
  if best==possibilities[0]:
    return 1
  elif best==possibilities[1]:
    return 2
  elif best==possibilities[2]:
    return 3
  elif best==possibilities[3]:
    return 4
  else:
    return 5


def get_maximum_prediction_value(clf1, clf2, clf3, clf4, clf5, validation_data, index):
  a = clf1.decision_function(validation_data[index])[0]
  b = clf2.decision_function(validation_data[index])[0]
  c = clf3.decision_function(validation_data[index])[0]
  d = clf4.decision_function(validation_data[index])[0]
  e = clf5.decision_function(validation_data[index])[0]
  highest = max(a,b,c,d,e)
  if highest==a:
    return 1
  elif highest==b:
    return 2
  elif highest==c:
    return 3
  elif highest==d:
    return 4
  elif highest==e:
    return 5


def classify_1(training_data, training_labels):
  #operator = LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
   #  intercept_scaling=1, loss='l1', multi_class='ovr', penalty='l2',
    # random_state=None, tol=0.0001, verbose=0)
  operator = KNeighborsClassifier(algorithm='auto', leaf_size=20, metric='minkowski',
           n_neighbors=50, p=2, weights='distance')
  operator.fit(training_data, training_labels)
  return operator

def classify_2(training_data, training_labels):
##  operator = LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
##     intercept_scaling=1, loss='l1', multi_class='ovr', penalty='l2',
##     random_state=None, tol=0.0001, verbose=0)
  operator = KNeighborsClassifier(algorithm='auto', leaf_size=20, metric='minkowski',
           n_neighbors=50, p=2, weights='distance')
  operator.fit(training_data, training_labels)
  return operator

def classify_3(training_data, training_labels):
##  operator = LinearSVC(C=10, class_weight=None, dual=True, fit_intercept=True,
##     intercept_scaling=1, loss='l1', multi_class='ovr', penalty='l2',
##     random_state=None, tol=0.0001, verbose=0)
  operator = KNeighborsClassifier(algorithm='auto', leaf_size=20, metric='minkowski',
           n_neighbors=5, p=2, weights='distance')
  operator.fit(training_data,training_labels)
  return operator

def classify_4(training_data, training_labels):
##  operator = LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
##     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
##     random_state=None, tol=0.0001, verbose=0)
  operator = KNeighborsClassifier(algorithm='auto', leaf_size=20, metric='minkowski',
           n_neighbors=10, p=2, weights='distance')
  operator.fit(training_data, training_labels)
  return operator

def classify_5(training_data, training_labels):
##  operator = LinearSVC(C=5, class_weight=None, dual=False, fit_intercept=True,
##     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l1',
##     random_state=None, tol=0.0001, verbose=0)
  operator = KNeighborsClassifier(algorithm='auto', leaf_size=20, metric='minkowski',
           n_neighbors=5, p=2, weights='distance')
  operator.fit(training_data, training_labels)
  return operator

main()