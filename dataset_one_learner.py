import argparse
import sys, os

import numpy as np
from sklearn.externals import joblib


def main():
  print "Starting..."
  #Best score so far = .56
  #tryLinearSVC(True)

  #Best score so far = .42
  #Alpha = 0.1, fit_prior = False
  #tryMultinomialNaiveBayes(False)

  #Best score so far = .47
  #Alpha = 0.1, Binarize = 0, fit_prior = False
  tryBinomialNaiveBayes(False)

def tryBinomialNaiveBayes(goFast):
  best_score = 0

  from sklearn.datasets import dump_svmlight_file, load_svmlight_file
  if goFast:
    training_data, training_labels = load_svmlight_file("dt1_1500.trn.svm", n_features=253659, zero_based=True)
    validation_data, validation_labels = load_svmlight_file("dt1_1500.vld.svm", n_features=253659, zero_based=True)
    testing_data, testing_labels = load_svmlight_file("dt1_1500.tst.svm", n_features=253659, zero_based=True)
  else:
    training_data, training_labels = load_svmlight_file("dt1.trn.svm")
    validation_data, validation_labels = load_svmlight_file("dt1.vld.svm")
    testing_data, testing_labels = load_svmlight_file("dt1.tst.svm")

  from sklearn.naive_bayes import BernoulliNB

  for alpha_value in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    for binarize_value in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
      for fit_prior_value in [True, False]:
        binary_operator = BernoulliNB(alpha_value,binarize_value,fit_prior_value)
        binary_operator.fit(training_data,training_labels)
        current_score = binary_operator.score(validation_data,validation_labels)

        print "Current test: " + str(alpha_value), str(binarize_value), fit_prior_value
        print "Current score: " + str(current_score)

        if current_score > best_score:
          best_score = current_score
          print "***NEW MAXIMUM SCORE: " + str(best_score)
          print "***NEW MAXIMUM PARAMETERS: " + str(alpha_value), str(binarize_value), fit_prior_value

  print "Best score was " + str(best_score)

def tryMultinomialNaiveBayes(goFast):

  best_score = 0

  from sklearn.datasets import dump_svmlight_file, load_svmlight_file
  if goFast:
    training_data, training_labels = load_svmlight_file("dt1_1500.trn.svm", n_features=253659, zero_based=True)
    validation_data, validation_labels = load_svmlight_file("dt1_1500.vld.svm", n_features=253659, zero_based=True)
    testing_data, testing_labels = load_svmlight_file("dt1_1500.tst.svm", n_features=253659, zero_based=True)
  else:
    training_data, training_labels = load_svmlight_file("dt1.trn.svm")
    validation_data, validation_labels = load_svmlight_file("dt1.vld.svm")
    testing_data, testing_labels = load_svmlight_file("dt1.tst.svm")

  from sklearn.naive_bayes import MultinomialNB

  for alpha_value in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    for fit_prior_value in [True, False]:
      multinomial_operator = MultinomialNB(alpha=alpha_value,fit_prior=fit_prior_value)
      multinomial_operator.fit(training_data,training_labels)
      current_score = multinomial_operator.score(validation_data,validation_labels)

      print "Current test: " + str(alpha_value), fit_prior_value
      print "Current score: " + str(current_score)

      if current_score > best_score:
        best_score = current_score
        print "***NEW MAXIMUM SCORE: " + str(best_score)
        print "***NEW MAXIMUM PARAMETERS: " + str(alpha_value), fit_prior_value

  print "Best score was " + str(best_score)

def tryLinearSVC(goFast):
    #Testing time went from 180s to 30s by using the 1500-line files,
    #though accuracy dropped significantly.  Anyway, just judge your
    #accuracy relative to the baseline for 1500.  It doesn't matter.
    #Test it by passing in False before any submission, though.
    try:

        args_id = 1
        args_d = "."
        n_features = 253659

        if goFast:
          fname_trn = os.path.join(args_d, "dt%d_1500.%s.svm" % (args_id, "trn"))
          fname_vld = os.path.join(args_d, "dt%d_1500.%s.svm" % (args_id, "vld"))
          fname_tst = os.path.join(args_d, "dt%d_1500.%s.svm" % (args_id, "tst"))
        else:
          fname_trn = os.path.join(args_d, "dt%d.%s.svm" % (args_id, "trn"))
          fname_vld = os.path.join(args_d, "dt%d.%s.svm" % (args_id, "vld"))
          fname_tst = os.path.join(args_d, "dt%d.%s.svm" % (args_id, "tst"))

        fname_vld_lbl = os.path.join(args_d, "dt%d.%s.lbl" % (args_id, "vld"))
        fname_tst_lbl = os.path.join(args_d, "dt%d.%s.lbl" % (args_id, "tst"))

        fname_vld_pred = os.path.join(args_d, "dt%d.%s.pred" % (args_id, "vld"))
        fname_tst_pred = os.path.join(args_d, "dt%d.%s.pred" % (args_id, "tst"))

        for fn in (fname_trn, fname_vld, fname_tst):
            if not os.path.isfile(fn):
                print("Missing dataset file: %s " % (fn,))
                sys.exit(1)

        ### reading labels
        from sklearn.datasets import dump_svmlight_file, load_svmlight_file
        data_trn, lbl_trn = load_svmlight_file(fname_trn, n_features=253659, zero_based=True)
        data_vld, lbl_vld = load_svmlight_file(fname_vld, n_features=253659, zero_based=True)
        data_tst, lbl_tst = load_svmlight_file(fname_tst, n_features=253659, zero_based=True)

        ### perform grid search using validation samples
        from sklearn.svm import LinearSVC
        from sklearn.metrics import accuracy_score

        str_formats = (None, "%d", "%d", "%.6f")

        cls_obj=LinearSVC
        metric_obj=accuracy_score

        best_param = None
        best_score = None
        best_svc = None

        for c_value in [1, 10, 50, 100]:
          for loss_value in ['l1','l2']:
            for penalty_value in ['l1','l2']:
              for dual_value in [True, False]:
                for tol_value in [0.0001, 0.001, 0.01]:
                  for multi_class_value in ['ovr','crammer_singer']:
                    for fit_intercept_value in[True, False]:
                      for intercept_scaling_value in [0.01,0.1,1]:
                        try:
                          cls = cls_obj(penalty=penalty_value,loss=loss_value,dual=dual_value,tol=tol_value,C=c_value,multi_class=multi_class_value,fit_intercept=fit_intercept_value,intercept_scaling=intercept_scaling_value,class_weight='auto')
                          cls.fit(data_trn,lbl_trn)

                          one_score = metric_obj(lbl_vld, cls.predict(data_vld))

                          print "Parameters: " + str(c_value), loss_value, penalty_value, dual_value, str(tol_value), multi_class_value, fit_intercept_value, str(intercept_scaling_value)
                          print "Score: " + str(one_score)

                          if ( best_score is None or
                               (args_id < 3 and best_score < one_score) or
                               (args_id == 3 and best_score > one_score) ):
                              print "***NEW MAXIMUM PARAMETER SET:" + str(c_value), loss_value, penalty_value, dual_value, str(tol_value), multi_class_value, fit_intercept_value, str(intercept_scaling_value)
                              print "***NEW MAXIMUM SCORE: " + str(one_score)
                              best_score = one_score
                              best_svc = cls
                        except:
                            print "Illegal parameter combination; moving to next combination."

        pred_vld = best_svc.predict(data_vld)
        pred_tst = best_svc.predict(data_tst)

        print ("Best score for vld: %.6f" % (metric_obj(lbl_vld, pred_vld),))
        print ("Best score for tst: %.6f" % (metric_obj(lbl_tst, pred_tst),))

        np.savetxt(fname_vld_pred, pred_vld, delimiter='\n', fmt=str_formats[args_id])
        np.savetxt(fname_tst_pred, pred_tst, delimiter='\n', fmt=str_formats[args_id])

        np.savetxt(fname_vld_lbl, lbl_vld, delimiter='\n', fmt=str_formats[args_id])
        np.savetxt(fname_tst_lbl, lbl_tst, delimiter='\n', fmt=str_formats[args_id])

    except Exception, exc:
        import traceback
        print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))


if __name__ == '__main__':
  main()



