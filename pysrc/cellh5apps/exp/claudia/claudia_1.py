import matplotlib
matplotlib.use('Qt4Agg')
import numpy
from cellh5apps.outlier import OutlierDetection, OutlierDetectionSingleCellPlots, OutlierClusterPlots
from cellh5apps.outlier.learner import OneClassSVM, OneClassSVM_SKL, ClusterGMM, ClusterKM, OneClassMahalanobis, OneClassSVM_SKL_PRECLUSTER, OneClassGMM, OneClassKDE
from cellh5apps.exp import EXP
import time
import pandas

import faulthandler
faulthandler.enable()

if __name__ == "__main__":
    if True:
        od = OutlierDetection("claudia_01", **EXP['claudia_01'])
        od.set_max_training_sample_size(10000)
        od.read_feature(remove_feature=(16,  17,  18,  62,  92, 122, 152, 197, 198, 201, 202), read_classification=False)
        od.pca_run(whiten=True)
        #od.cluster_run(ClusterGMM, max_samples=1000, covariance_type="full", n_components=2)
        
        od_plots = OutlierDetectionSingleCellPlots(od)
        
        feature_set="Object features"

        od.train(classifier_class=OneClassMahalanobis, feature_set=feature_set)
        od.predict2(feature_set=feature_set)
        od.compute_outlyingness()
        od.make_top_hit_list(top=1000)
        
    print 'fini'
       
        
        