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

training_data, training_labels = load_svmlight_file("dt1.trn.svm")
validation_data, validation_labels = load_svmlight_file("dt1.vld.svm")
testing_data, testing_labels = load_svmlight_file("dt1.tst.svm")

def main():

  classifier12 = LinearSVC(C=1, class_weight='auto', dual=True, fit_intercept=False,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)
  classifier13 = LinearSVC(C=1, class_weight='auto', dual=True, fit_intercept=False,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)
  classifier14 = LinearSVC(C=1, class_weight='auto', dual=True, fit_intercept=False,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)
  classifier15 = LinearSVC(C=1, class_weight='auto', dual=True, fit_intercept=False,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)
  classifier23 = LinearSVC(C=1, class_weight='auto', dual=True, fit_intercept=False,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)
  classifier24 = LinearSVC(C=1, class_weight='auto', dual=True, fit_intercept=False,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)
  classifier25 = LinearSVC(C=1, class_weight='auto', dual=True, fit_intercept=False,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)
  classifier34 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           n_neighbors=10, p=2, weights='distance')
  classifier35 = LinearSVC(C=1, class_weight='auto', dual=True, fit_intercept=False,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)
  classifier45 = LinearSVC(C=1, class_weight='auto', dual=True, fit_intercept=True,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)

  classifier12 = train_classifier(classifier12,1,2)
  classifier13 = train_classifier(classifier12,1,3)
  classifier14 = train_classifier(classifier12,1,4)
  classifier15 = train_classifier(classifier12,1,5)
  classifier23 = train_classifier(classifier12,2,3)
  classifier24 = train_classifier(classifier12,2,4)
  classifier25 = train_classifier(classifier12,2,5)
  classifier34 = train_classifier(classifier12,3,4)
  classifier35 = train_classifier(classifier12,3,5)
  classifier45 = train_classifier(classifier12,4,5)

  classifier1only = LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='l1', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)
  classifier2only = LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='l1', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)
  classifier3only = LinearSVC(C=10, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='l1', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)
  classifier4only = LinearSVC(C=1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=0.0001, verbose=0)
  classifier5only = LinearSVC(C=5, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l1',
     random_state=None, tol=0.0001, verbose=0)

  classifier1only = train_classifier_all(classifier1only,1)
  classifier2only = train_classifier_all(classifier2only,2)
  classifier3only = train_classifier_all(classifier3only,3)
  classifier4only = train_classifier_all(classifier4only,4)
  classifier5only = train_classifier_all(classifier5only,5)

  predicted_validation_labels = []
  predicted_test_labels = []

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

  #######################

  for element in testing_data:
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
      predicted_test_labels.append(5)
    elif max(votes) == four:
      predicted_test_labels.append(4)
    elif max(votes) == one:
      predicted_test_labels.append(1)
    elif max(votes) == two:
      predicted_test_labels.append(2)
    else:
      predicted_test_labels.append(3)

  #######################

  print accuracy_score(predicted_validation_labels,validation_labels)

  np.savetxt("dt1.tst.pred",predicted_test_labels,delimiter="\n",fmt="%d")

def train_classifier_all(classifier, a):
  relevant_training_labels = []
  for element in training_labels:
    if element==a:
      relevant_training_labels.append(1)
    else:
      relevant_training_labels.append(0)

  current_training_labels = np.array(relevant_training_labels)
  classifier.fit(training_data,current_training_labels)
  return classifier

def train_classifier(classifier,a,b):
  t_data = []; t_labels = []
  t_idx = []; v_idx = []

  for index in range(len(training_labels)):
    if training_labels[index] in [a, b]:
      t_idx.append(index)
      t_labels.append(training_labels[index])

  t_data = training_data[t_idx]; t_labels = np.array(t_labels)

  classifier.fit(training_data,training_labels)
  return classifier


def get_classifier_for(a, b):
  #all calls to this have been removed...
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
