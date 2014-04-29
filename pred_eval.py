import argparse
import sys

import numpy as np
from sklearn.externals import joblib

from sklearn.metrics import mean_squared_error, accuracy_score


if __name__ == '__main__':
    
    try:
        parser = argparse.ArgumentParser(description='evaluate predictions')
        parser.add_argument('-gt',
                            required=True,
                            help='Ground truth labels')

        parser.add_argument('-pr',
                            required=True,
                            help='Predicted labels')

        parser.add_argument('-id', type=int,
                            choices=[1,2,3],
                            help='Dataset id')

        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(0)

        args = parser.parse_args()

        ### reading labels
        lbl_gt= np.loadtxt(args.gt)
        lbl_pr= np.loadtxt(args.pr)
        
        metric_obj=None
        metric_name=None
        
        if args.id in [1,2]:
            metric_obj = accuracy_score
            metric_name = 'classification accuracy'
        else:
            metric_obj = mean_squared_error
            metric_name = 'mean squared error'
        
        metric_score = metric_obj(lbl_gt, lbl_pr)
        
        print ("%s: %.5f" % (metric_name,metric_score))
    except Exception, exc:
        import traceback
        print('Exception was raised in %s of %s: %s \n %s ' % (__name__, __file__, str(exc), ''.join(traceback.format_exc())))




