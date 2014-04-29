import argparse
import sys, os

import numpy as np
from sklearn.externals import joblib


def main():
  tryLinearSVC(True)


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



