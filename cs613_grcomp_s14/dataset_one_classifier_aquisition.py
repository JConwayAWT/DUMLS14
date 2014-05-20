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

  classifier_1 = classify_ones(training_data, training_labels_1, validation_data, validation_labels_1)
  classifier_2 = classify_ones(training_data, training_labels_2, validation_data, validation_labels_2)
  classifier_3 = classify_ones(training_data, training_labels_3, validation_data, validation_labels_3)
  classifier_4 = classify_ones(training_data, training_labels_4, validation_data, validation_labels_4)
  classifier_5 = classify_ones(training_data, training_labels_5, validation_data, validation_labels_5)

  print classifier_1
  print classifier_2
  print classifier_3
  print classifier_4
  print classifier_5

  winsound.Beep(440, 1000)

def classify_ones(training_data, training_labels, validation_data, validation_labels):
  max_score = 0
  best_classifier = None

##  general_svc_parameter_grid = [{
##    "C":[1,10],
##    "kernel":['rbf','poly','sigmoid'],
##    "degree":[2,3,4],
##    "class_weight":['auto'],
##  }]
##
##  for combo in ParameterGrid(general_svc_parameter_grid):
##    try:
##      svc_operator = SVC(**combo)
##      svc_operator.fit(training_data[:2000], training_labels[:2000])
##      score = accuracy_score(validation_labels[:600],svc_operator.predict(validation_data[:600]))
##      print score
##      if score > max_score:
##        max_score = score
##        best_classifier = svc_operator
##        print "New max score ^^^"
##    except:
##      print "Skipping illegal parameter combination"

##  svc_parameter_grid = [{
##    "C":[1, 5, 10],
##    "loss":["l1","l2"],
##    "penalty":["l1","l2"],
##    "dual":[True, False],
##  }]
##  for combo in ParameterGrid(svc_parameter_grid):
##    try:
##      svc_operator = LinearSVC(**combo)
##      svc_operator.fit(training_data, training_labels)
##      score = accuracy_score(validation_labels,svc_operator.predict(validation_data))
##      print score
##      if score > max_score:
##        max_score = score
##        best_classifier = svc_operator
##        print "New max score ^^^"
##    except:
##      print "Skipping illegal parameter combination"

##  knn_parameter_grid = [{
##    "n_neighbors":[5, 10, 30, 50],
##    "weights":['uniform','distance'],
##    "leaf_size":[20,30,60,100]
##  }]
##
##  for combo in ParameterGrid(knn_parameter_grid):
##    try:
##      svc_operator = KNeighborsClassifier(**combo)
##      svc_operator.fit(training_data, training_labels)
##      score = accuracy_score(validation_labels,svc_operator.predict(validation_data))
##      print score
##      if score > max_score:
##        max_score = score
##        best_classifier = svc_operator
##        print "New max score ^^^"
##    except:
##      print "Skipping illegal parameter combination"

  from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron

##  rfc_parameter_grid = [{
##    "C":[1, 5, 10],
##    "n_iter":[5, 10, 20],
##    'fit_intercept':[True,False]
##  }]
##
##  for combo in ParameterGrid(rfc_parameter_grid):
##    try:
##      svc_operator = PassiveAggressiveClassifier(**combo)
##      svc_operator.fit(training_data, training_labels)
##      score = accuracy_score(validation_labels,svc_operator.predict(validation_data))
##      print score
##      if score > max_score:
##        max_score = score
##        best_classifier = svc_operator
##        print "New max score ^^^"
##    except:
##      print "Skipping illegal parameter combination"

##  perceptron_grid = [{
##    "penalty":['l2','l1','elasticnet'],
##    "alpha":[0.0001,0.0002,0.0004],
##    'n_iter':[5,10,20],
##    'class_weight':['auto']
##  }]
##
##  for combo in ParameterGrid(perceptron_grid):
##    try:
##      svc_operator = Perceptron(**combo)
##      svc_operator.fit(training_data, training_labels)
##      score = accuracy_score(validation_labels,svc_operator.predict(validation_data))
##      print score
##      if score > max_score:
##        max_score = score
##        best_classifier = svc_operator
##        print "New max score ^^^"
##    except:
##      print "Skipping illegal parameter combination"

  print "Max score: " + str(max_score)
  print best_classifier
  return [best_classifier, max_score]


main()