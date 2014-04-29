import argparse
import sys, os

import pylab as pl

import numpy as np
from numpy import histogram
#from scipy.stats import itemfreq
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.svm import LinearSVC, SVC, SVR
#from sklearn.manifold import Isomap
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import SparsePCA
## peter LDA requires dense matrix
#from sklearn.lda import LDA
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier

class MyGaussianNB:
	def __init__(self, **kwargs):
		self.pipeline = Pipeline([
			("feature_selection", LinearSVC(penalty="l2",dual=False,tol=1e-3)),
			("classification", GaussianNB())
		])		

	def fit(self, x, y):
		return self.pipeline.fit(x, y)

	def predict(self, x):
		return self.pipeline.predict(x)

#
# The training data must be in dense matrix format
#
class MyEnsemble:
	def __init__(self, **kwargs):
		self.pipeline = Pipeline([
			("feature_selection", LinearSVC(penalty="l2",dual=False,tol=1e-3)),
			("classification", GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0))
		])		

	def fit(self, x, y):
		return self.pipeline.fit(x, y)

	def predict(self, x):
		return self.pipeline.predict(x)


class MyClassifier:
	def __init__(self, **kwargs):

		#self.dict = dict()
		#for key,value in kwargs.iteritems():
		#	self.dict[key] = value

		self.pipeline = Pipeline(
		[
			#("classification", LinearSVC(**kwargs))
			("classification", SGDClassifier(**kwargs))
		])

	def fit(self, x, y):
	
		return self.pipeline.fit(x,y)

	def predict(self,x):
		return self.pipeline.predict(x)

if __name__ == '__main__':
    
    try:
        parser = argparse.ArgumentParser(description='baseline for predicting labels')

        parser.add_argument('-d',
                            default='.',
                            help='Directory with datasets in SVMLight format')

        parser.add_argument('-id', type=int,
                            default=1,
                            choices=[1,2,3],
                            help='Dataset id')

        parser.add_argument('-plotsvd', type=bool,
                            default=False,
                            help='Visualizes SVD-reprojected dataset #1')

        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args()

        n_features = 100 ## dataset id = 3
        if args.id == None or args.id == 1:
            n_features = 253659
        elif args.id == 2:
            n_features = 200
        
        fname_trn = os.path.join(args.d, "dt%d.%s.svm" % (args.id, "trn"))
        fname_vld = os.path.join(args.d, "dt%d.%s.svm" % (args.id, "vld"))
        fname_tst = os.path.join(args.d, "dt%d.%s.svm" % (args.id, "tst"))

        fname_vld_lbl = os.path.join(args.d, "dt%d.%s.lbl" % (args.id, "vld"))
        fname_tst_lbl = os.path.join(args.d, "dt%d.%s.lbl" % (args.id, "tst"))

        fname_vld_pred = os.path.join(args.d, "dt%d.%s.pred" % (args.id, "vld"))
        fname_tst_pred = os.path.join(args.d, "dt%d.%s.pred" % (args.id, "tst"))
        
        for fn in (fname_trn, fname_vld, fname_tst):
            if not os.path.isfile(fn):
                print("Missing dataset file: %s " % (fn,))
                sys.exit(1)
        
        ### reading labels
        from sklearn.datasets import dump_svmlight_file, load_svmlight_file
        data_trn, lbl_trn = load_svmlight_file(fname_trn, n_features=n_features, zero_based=True)
        data_vld, lbl_vld = load_svmlight_file(fname_vld, n_features=n_features, zero_based=True)
        data_tst, lbl_tst = load_svmlight_file(fname_tst, n_features=n_features, zero_based=True)

	# peter print unique labels. Gradient-descent classifier does not seem to support class_weight parameter...?? 
	#print np.unique(lbl_trn)
	#print np.ones(lbl_trn.shape[0], dtype=np.float64, order='C')
	#print np.searchsorted(lbl_trn, '0.')
	#print np.searchsorted(lbl_trn, '1.')
	#print np.searchsorted(lbl_trn, '2.')
	#print np.searchsorted(lbl_trn, '3.')
	#print np.searchsorted(lbl_trn, '4.')
	#print np.searchsorted(lbl_trn, '5.')

	# Plot the principal components using SVD
	if args.plotsvd == True:
		svd = TruncatedSVD(n_components=len(set(lbl_trn)))
		X_r2 = svd.fit(data_trn, lbl_trn).transform(data_trn)
		pl.figure()
		for c, i, target_name in zip("rgb", [1, 2, 3, 4, 5], set(lbl_trn)):
			pl.scatter(X_r2[lbl_trn == i, 0], X_r2[lbl_trn == i, 1], c=c, label=target_name)
		pl.legend()
		pl.title('SVD of Dataset#1 dataset')
		pl.show()


        
        ### perform grid search using validation samples
        dt1_grid = [{'loss': ['modified_huber', 'squared_hinge'],'penalty':['l2'],'class_weight':['auto',None],'alpha':[0.00025],'n_iter':[5,10,15,20,25,30,50,100]}]


        dt2_grid = [{'kernel': ['rbf'], 'C': [1.0, 100.0, 10000.0],
                     'gamma': [0.1, 1.0, 10.0]}]

        dt3_grid = [{'kernel': ['rbf'], 'C': [1.0, 100.0, 10000.0],
                     'gamma': [0.1, 1.0, 10.0]}]

        grids = (None, dt1_grid, dt2_grid, dt3_grid)
        classifiers = (None, MyClassifier, SVC, SVR)
        metrics = (None, accuracy_score, accuracy_score, mean_squared_error)
        str_formats = (None, "%d", "%d", "%.6f")
        #LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=1.0,

        grid_obj=grids[args.id]
        cls_obj=classifiers[args.id]
        metric_obj=metrics[args.id]
        
        best_param = None
        best_score = None
        best_svc = None
        
        for one_param in ParameterGrid(grid_obj):
            cls = cls_obj(**one_param)
            cls.fit(data_trn, lbl_trn)
            one_score = metric_obj(lbl_vld, cls.predict(data_vld))
            
            print ("param=%s, score=%.6f" % (repr(one_param),one_score))
            
            if ( best_score is None or 
                 (args.id < 3 and best_score < one_score) or
                 (args.id == 3 and best_score > one_score) ):
                best_param = one_param
                best_score = one_score
                best_svc = cls
            
        pred_vld = best_svc.predict(data_vld)
        pred_tst = best_svc.predict(data_tst)

	print "\n\nBest configuration: {}".format(repr(best_param))        
        print ("Best score for vld: %.6f" % (metric_obj(lbl_vld, pred_vld),))
        print ("Best score for tst: %.6f" % (metric_obj(lbl_tst, pred_tst),))
        
        np.savetxt(fname_vld_pred, pred_vld, delimiter='\n', fmt=str_formats[args.id])
        np.savetxt(fname_tst_pred, pred_tst, delimiter='\n', fmt=str_formats[args.id])
        
        np.savetxt(fname_vld_lbl, lbl_vld, delimiter='\n', fmt=str_formats[args.id])
        np.savetxt(fname_tst_lbl, lbl_tst, delimiter='\n', fmt=str_formats[args.id])

    except Exception, exc:
        import traceback
        print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))






