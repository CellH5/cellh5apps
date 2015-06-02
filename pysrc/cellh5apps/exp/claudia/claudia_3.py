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
        od = OutlierDetection("claudia_3", **EXP['claudia_03'])
        od.set_max_training_sample_size(10000)
        od.read_feature(object_="secondary__expanded", remove_feature=(23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
                                                                        36,  37,  38,  39,  40,  41,  42,  47,  48,  62,  63,  77,  78,
                                                                        92,  93, 107, 108, 122, 123, 137, 138, 152, 153, 169, 170, 173,
                                                                       174, 177, 178, 181, 182, 189, 190, 191, 193, 194, 195, 197, 198,
                                                                       199, 201, 202, 205, 206, 209, 210, 213, 214), read_classification=False)
        od.pca_run(whiten=True)
        #od.cluster_run(ClusterGMM, max_samples=1000, covariance_type="full", n_components=2)
        
        od_plots = OutlierDetectionSingleCellPlots(od)
        
        feature_set="Object features"

        od.train(classifier_class=OneClassMahalanobis, feature_set=feature_set)
        od.predict2(feature_set=feature_set)
        od.compute_outlyingness()
        od.make_top_hit_list(top=1000)
        
    print 'fini'