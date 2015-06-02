import matplotlib
matplotlib.use('Qt4Agg')
import numpy
from cellh5apps.outlier import OutlierDetection, OutlierDetectionSingleCellPlots, OutlierClusterPlots
from cellh5apps.outlier.learner import OneClassSVM, OneClassSVM_SKL, ClusterGMM, ClusterKM, OneClassMahalanobis, OneClassSVM_SKL_PRECLUSTER, OneClassGMM, OneClassKDE, OneClassGMM2, OneClassRandom
from cellh5apps.exp import EXP

def test_feature_read():
    od = OutlierDetection("mito_check_feature_read_TEST", **EXP['mito_1'])
    od.set_max_training_sample_size(20000)
#     od.read_feature(remove_feature=(0, 62, 92, 122, 152))
    od.read_feature()
    od.pca_run(whiten=True)
if __name__ == "__main__":
    test_feature_read()


        
    print "Fini"

        
        
        