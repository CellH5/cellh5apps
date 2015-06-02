import numpy
from cellh5apps.outlier.tasks import ClassLabelMDS
from cellh5_analysis import CellH5Analysis
from cellh5apps.exp.embo2014.experiments import EXP

if __name__ == "__main__":
    name = "EMBO_2012_Group1_001_subset"
    ca = CellH5Analysis(name , **EXP[name])
    ca.set_read_feature_time_predicate(numpy.equal, 94)
    ca.read_feature(object_="primary__primary")
    ca.pca()
    mds = ClassLabelMDS(ca)
    
    mds.run()
    
    print "***"
    

