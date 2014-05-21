from sklearn.externals import joblib
from sklearn.datasets import load_svmlight_file
import numpy as np
import winsound
import scipy
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import accuracy_score

best_training_accuracies = []

training_data, training_labels = load_svmlight_file("dt1.trn.svm")
validation_data, validation_labels = load_svmlight_file("dt1.vld.svm")

def main():
  classifier12 = get_classifier_for(1, 2)
  classifier13 = get_classifier_for(1, 3)
  classifier14 = get_classifier_for(1, 4)
  classifier15 = get_classifier_for(1, 5)
  classifier23 = get_classifier_for(2, 3)
  classifier24 = get_classifier_for(2, 4)
  classifier25 = get_classifier_for(2, 5)
  classifier34 = get_classifier_for(3, 4)
  classifier35 = get_classifier_for(3, 5)
  classifier45 = get_classifier_for(4, 5)

  classifier1only = get_all_against_for(1)
  classifier2only = get_all_against_for(2)
  classifier3only = get_all_against_for(3)
  classifier4only = get_all_against_for(4)
  classifier5only = get_all_against_for(5)


  predicted_validation_labels = []

  for element in validation_data:
    vote_list = []
    vote_list.append(int(classifier12.predict(element)[0]))
    vote_list.append(int(classifier13.predict(element)[0]))
    vote_list.append(int(classifier14.predict(element)[0]))
    vote_list.append(int(classifier15.predict(element)[0]))
    vote_list.append(int(classifier23.predict(element)[0]))
    vote_list.append(int(classifier24.predict(element)[0]))
    vote_list.append(int(classifier25.predict(element)[0]))
    vote_list.append(int(classifier34.predict(element)[0]))
    vote_list.append(int(classifier35.predict(element)[0]))
    vote_list.append(int(classifier45.predict(element)[0]))

    if int(classifier1only.predict(element)[0]) == 1:
      vote_list.append(1)

    if int(classifier2only.predict(element)[0]) == 1:
      vote_list.append(2)

    if int(classifier3only.predict(element)[0]) == 1:
      vote_list.append(3)

    if int(classifier4only.predict(element)[0]) == 1:
      vote_list.append(4)

    if int(classifier5only.predict(element)[0]) == 1:
      vote_list.append(5)

    one = vote_list.count(1)
    two = vote_list.count(2)
    three = vote_list.count(3)
    four = vote_list.count(4)
    five = vote_list.count(5)

    votes = [one, two, three, four, five]

    if max(votes) == five:
      predicted_validation_labels.append(5)
    elif max(votes) == four:
      predicted_validation_labels.append(4)
    elif max(votes) == one:
      predicted_validation_labels.append(1)
    elif max(votes) == two:
      predicted_validation_labels.append(2)
    else:
      predicted_validation_labels.append(3)

  print accuracy_score(predicted_validation_labels,validation_labels)

  winsound.Beep(880,1000)

  print best_training_accuracies

def get_all_against_for(a):
  max_score = 0
  best_classifier = None

  relevant_training_labels = []
  relevant_validation_labels = []
  for element in training_labels:
    if element==a:
      relevant_training_labels.append(1)
    else:
      relevant_training_labels.append(0)

  for element in validation_labels:
    if element==a:
      relevant_validation_labels.append(1)
    else:
      relevant_validation_labels.append(0)

  current_training_labels = np.array(relevant_training_labels)
  current_validation_labels = np.array(relevant_validation_labels)

  svc_parameter_grid = [{
    "C":[1, 5, 10],
    "loss":["l1","l2"],
    "penalty":["l1","l2"],
    "dual":[True, False],
  }]
  for combo in ParameterGrid(svc_parameter_grid):
    try:
      svc_operator = LinearSVC(**combo)
      svc_operator.fit(training_data, training_labels)
      score = accuracy_score(validation_labels,svc_operator.predict(validation_data))
      print score
      if score > max_score:
        max_score = score
        best_classifier = svc_operator
        print "New max score ^^^"
    except:
      print "Skipping illegal parameter combination"

  return best_classifier


def get_classifier_for(a, b):
  t_data = []; t_labels = []
  v_data = []; v_labels = []
  t_idx = []; v_idx = []

  for index in range(len(training_labels)):
    if training_labels[index] in [a, b]:
      t_idx.append(index)
      t_labels.append(training_labels[index])

  for index in range(len(validation_labels)):
    if validation_labels[index] in [a, b]:
      v_idx.append(index)
      v_labels.append(validation_labels[index])

  t_data = training_data[t_idx]; t_labels = np.array(t_labels)
  v_data = validation_data[v_idx]; v_labels = np.array(v_labels)

  max_score = 0
  best_classifier = None

  svc_parameter_grid = [{
    "C":[1, 5, 10],
    "dual":[True, False],
    "multi_class":["ovr","crammer_singer"],
    "fit_intercept":[True, False],
    "class_weight":['auto']
  }]

  for combo in ParameterGrid(svc_parameter_grid):
    try:
      svc_operator = LinearSVC(**combo)
      svc_operator.fit(t_data, t_labels)
      score = accuracy_score(v_labels, svc_operator.predict(v_data))
      print score
      if score > max_score:
        max_score = score
        best_classifier = svc_operator
        print "(New max score)"
    except:
      print "Illegal parameters skipped"

  knn_parameter_grid = [{
    "n_neighbors":[5, 10, 50],
    "weights":["distance"],
    "algorithm":["auto"]
  }]

  for combo in ParameterGrid(knn_parameter_grid):
    try:
      knn_operator = KNeighborsClassifier(**combo)
      knn_operator.fit(t_data, t_labels)
      score = accuracy_score(v_labels, knn_operator.predict(v_data))
      print score
      if score > max_score:
        max_score = score
        best_classifier = knn_operator
        print "(New max score)"
    except:
      print "Illegal parameters skipped"

  mb_parameter_grid = [{
    'alpha':[0,1,2],
    'fit_prior':[True,False],
  }]

  for combo in ParameterGrid(mb_parameter_grid):
    try:
      svc_operator = MultinomialNB(**combo)
      svc_operator.fit(t_data, t_labels)
      score = accuracy_score(v_labels, svc_operator.predict(v_data))
      print score
      if score > max_score:
        max_score = score
        best_classifier = svc_operator
        print "(New max score)"
    except:
      print "Illegal parameters skipped"

  mb_parameter_grid = [{
    'alpha':[0,1,2],
    'binarize':[True,False],
    'fit_prior':[True,False]
  }]

  for combo in ParameterGrid(mb_parameter_grid):
    try:
      svc_operator = MultinomialNB(**combo)
      svc_operator.fit(t_data, t_labels)
      score = accuracy_score(v_labels, svc_operator.predict(v_data))
      print score
      if score > max_score:
        max_score = score
        best_classifier = svc_operator
        print "(New max score)"
    except:
      print "Illegal parameters skipped"

  print best_classifier, max_score
  best_training_accuracies.append(max_score)
  return best_classifier


main()