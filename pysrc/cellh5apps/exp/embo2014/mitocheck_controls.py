import numpy
import vigra
import matplotlib
import matplotlib.cm
matplotlib.use('Qt4Agg')
    
from matplotlib import pyplot as plt 

from cellh5apps.outlier import OutlierDetection, OutlierDetectionSingleCellPlots, OutlierClusterPlots
from cellh5apps.outlier.learner import OneClassSVM, OneClassMahalanobis, OneClassSVM_SKL, OneClassKDE, OneClassGMM, OneClassSVM_SKL, OneClassSVM_LIBSVM,\
    ClusterGMM, ClusterKM
from cellh5apps.utils.colormaps import YlBlCMap

import faulthandler
faulthandler.enable()

from cellh5apps.exp import EXP
 
# EXP = {'EMBO_2012_Group1_001':
#         {
#         'mapping_files' : {
#            'EMBO_2012_Group1_001': 'F:/embo_course_2014/EMBO_2012_Group1_001.txt',
#         },
#         'cellh5_files' : {
#            'EMBO_2012_Group1_001': 'F:/embo_course_2014/EMBO_2012_Group1_001_all_positions_with_data.ch5',
#         },
# #         'locations' : (
# #             ("A",  8), ("B", 8), ("C", 8), ("D", 8),
# #             ("H", 6), ("H", 7), ("G", 6), ("G", 7),
# #             ("H",12), ("H",13), ("G",12), ("G",13),
# #             ),
# #         'rows' : list("ABCDEFGHIJKLMNOP")[:3],
# #         'cols' : tuple(range(2,4)),
#         'gamma' : 0.001953125,
#         'nu' : 0.01,
#         'pca_dims' : 239,
#         'kernel' :'rbf'
#         }
#    }
# 
# class EMBO2014Plate(object):
# 
#     
#     def evaluate_roc(self, cms, thrhs, split):
#         all_stats = []
#         for cm, t in zip(cms, thrhs):
#             cm_n = self.normalize(cm)       
#             stats = self.get_stats(cm_n, split)
#             all_stats.append(stats)
#             
#         self.plot_roc(all_stats)
#             
#         
#             
#     def plot_roc(self, all_stats):
#         tprs = [stat['tpr'] for stat in all_stats]
#         fprs = [stat['fpr'] for stat in all_stats]
#         
#         fig = plt.figure()
#         ax = plt.subplot(111)
#         plt.plot(fprs, tprs, 'k.-')
#         ax.set_aspect(1)
#         ax.set_title(self.od.classifier.describe())
#         ax.set_xlabel('fpr')
#         ax.set_ylabel('tpr')
#         
#         plt.tight_layout()
#         plt.savefig(self.od.output("roc_%s.png"  % self.od.classifier.describe()))
#         
#         plt.close(fig)
#         
#         
#     def evaluate(self, cm, cms, thrhs, split):
#         cm_n = self.normalize(cm)
#         stats = self.get_stats(cm, split)
#         stats_n = self.get_stats(cm_n, split)
#         
#         output=  "%5.4f\t%5.4f\t%5.4f\t%4.3f\t%4.3f\t%4.3f\t%s\n" % (stats_n['acc'], stats_n['tpr'], stats_n['fpr'], stats_n['pre'], stats_n['f1'], stats_n['f2'], self.od.classifier.describe())
#         with open(self.od.output("test.txt"), "a") as myfile:
#             myfile.write(output)
#         
#         self.export_cm([cm, cm_n], [stats, stats_n])
#                
#     def export_cm(self, cms, stats_l):
#         fig = plt.figure()
#         
#         for c_i, (cm, stats) in enumerate(zip(cms, stats_l)):
#             ax = plt.subplot(1, len(cms), c_i+1)
#             
#             if c_i ==0:
#                 vmax = cm.max()
#                 cm = cm.astype(numpy.int32)
#                 stri = "%d"
#             else:
#                 vmax = 1
#                 stri = "%4.2f"
#                 
#             res = ax.pcolor(cm, cmap=YlBlCMap, vmin=0, vmax=vmax)
#             for i, cas in enumerate(cm):
#                 for j, c in enumerate(cas):
#                     if c>0:
#                         ax.text(j+0.3, i+0.5, stri % c, fontsize=10)
#             ax.set_ylim(0,cm.shape[0])
#             ax.invert_yaxis()
#             ax.set_xticklabels([])
#             ax.set_yticklabels([])
#             
#             ax.set_title("acc %3.2f, tpr %3.2f, fpr %3.2f \npre %3.2f, f1 %3.2f, f2 %3.2f" % (stats['acc'], stats['tpr'], stats['fpr'], stats['pre'], stats['f1'], stats['f2']) + "\n" + self.od.classifier.describe(), fontdict={"size":6})
#             ax.set_aspect(0.7)
#             ax.spines["right"].set_visible(False)
#             ax.spines["top"].set_visible(False)
#             ax.spines["bottom"].set_visible(False)
#             ax.spines["left"].set_visible(False)
#             
#             plt.axis('off')
#         plt.tight_layout()
# 
#         fig.savefig(self.od.output("%s.png" % self.od.classifier.describe()))
#         plt.close(fig)
#         
#         
#         
#     def __init__(self, name, **kwargs):
#         self.od = OutlierDetection(name=name, **kwargs)
#         self.od.read_feature(object_="primary__primary", time_frames=[94])
#         
#     def analyze_roc(self):        
#         # GMM
#         for k in [2,10,100]:
#             self.od.train(classifier_class=OneClassGMM, k=k)
#             self.od.predict()
#             self.od.compute_outlyingness()
#             cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
#             self.evaluate_roc(cms, thrs, 2) 
#         
#         # Mahala
#         self.od.train(classifier_class=OneClassMahalanobis)
#         self.od.predict()
#         self.od.compute_outlyingness()
#         cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
#         self.evaluate_roc(cms, thrs, 2)   
#         
#         # KDE 
#         for bandwidth in numpy.linspace(0.01, 5, 4):
#             self.od.train(classifier_class=OneClassKDE, bandwidth=bandwidth)
#             self.od.predict()
#             self.od.compute_outlyingness()
#             cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
#             self.evaluate_roc(cms, thrs, 2)   
#         
#         # One class svm
#         for nu in [0.01, 0.10, 0.2, 0.99]:
#             self.od.set_nu(nu)  
#             for gamma in numpy.linspace(0.001, 0.2, 4):
#                 self.od.train(classifier_class=OneClassSVM, gamma=gamma, nu=nu, kernel="rbf")
#                 self.od.predict()
#                 self.od.compute_outlyingness()
#                 cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
#                 self.evaluate_roc(cms, thrs, 2)         
#         self.od.write_readme()
#         
#     def show_feature_space(self):
#         fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharey=True, sharex=True)
#         
#         all_ax = (ax1, ax2, ax3, ax4, ax5, ax6)
#         
#         
#         train_data = self.od.get_data(("neg",), "PCA")
#         ax1.scatter(train_data[:, 0], train_data[:, 1], c='green', edgecolor="none")
#         ax1.set_title("Training_data")
#         
#         clf = self.od.classifier
#         matrix = self.od.get_column_as_matrix("PCA")
#         outlier = self.od.get_column_as_matrix("Predictions")
#         classif = self.od.get_column_as_matrix("Object classification label")
#         
#         hpd = self.od.get_column_as_matrix("Hyperplane distance")
#         
#         xmin, xmax = matrix[:,0].min(), matrix[:,0].max()
#         ymin, ymax = matrix[:,1].min(), matrix[:,1].max()
#         xx, yy = numpy.meshgrid(numpy.linspace(xmin, xmax, 100), numpy.linspace(ymin, ymax, 100))
# 
#         #Z = clf.decision_function(numpy.c_[xx.ravel(), yy.ravel()])
#         Z = clf.kde.score_samples(numpy.c_[xx.ravel(), yy.ravel()])
#         Z = Z.reshape(xx.shape)    
#         Z = numpy.exp(Z)
# 
#         ax2.contourf(xx, yy, Z, levels=numpy.linspace(Z.min(), numpy.exp(clf.direct_thresh), 7), cmap=plt.get_cmap('Reds'))
#         ax2.contour(xx, yy, Z, levels=[numpy.exp(clf.direct_thresh)], linewidths=1, colors='red')
#         ax2.contourf(xx, yy, Z, levels=[numpy.exp(clf.direct_thresh), Z.max()], colors='orange')
#         ax2.set_title("Outlier density and cutoff\n%s" % self.od.classifier.describe())
#  
#         ax3.scatter(matrix[outlier==1, 0], matrix[outlier==1, 1], c='green', edgecolor="none")
#         ax3.set_title("Predicted inliers")
#         
#         ax4.set_title("Predicted outliers")
#         ax4.scatter(matrix[outlier==-1, 0], matrix[outlier==-1, 1], c='red', edgecolor="none")
#         
#         
#         ax5.scatter(matrix[classif==0, 0], matrix[classif==0, 1], c='green', edgecolor="none")
#         ax5.scatter(matrix[numpy.logical_and(classif<5, classif>0), 0], matrix[numpy.logical_and(classif<5, classif>0), 1], c='blue', edgecolor="none")
#         ax5.set_title("Supervised: Interphase (green) and mitotic (blue)")
#         ax5.set_xlabel("Principle component 1")
#         ax5.set_ylabel("Principle component 2")
#         
#         ax6.set_title("Supervised: Phenotypes")
#         ax6.scatter(matrix[classif>=5, 0], matrix[classif>=5, 1], c='red', edgecolor="none")
#         ax6.set_xlabel("Principle component 1")
#         ax6.set_ylabel("Principle component 2")
# 
#         for ax in all_ax:
#             ax.set_xlim(xmin, xmax)
#             ax.set_ylim(ymin, ymax)
#         
#         plt.tight_layout()
#         
#         plt.show()
#         
#     def myscatter(self, ax, x_vals, y_vals, bins=100):
#         
#         cmap = matplotlib.cm.jet
#         cmap._init(); 
#         cmap._lut[0,:] = 1
#         
#         xmin, xmax = x_vals.min(), x_vals.max()
#         ymin, ymax = y_vals.min(), y_vals.max()
#         
#         image, x_edges, y_edges = numpy.histogram2d(x_vals, y_vals, bins=bins)
#         
#         image = numpy.flipud(image.swapaxes(1,0))
#         
#         ax.imshow(image, interpolation="nearest", extent=[xmin, xmax, ymin, ymax], cmap=cmap)
#         
#         
#     def show_feature_space2(self):
#         return
#         fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharey=True, sharex=True)
#         
#         all_ax = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9)
#         matrix = self.od.get_column_as_matrix("PCA")
#         outlier = self.od.get_column_as_matrix("Predictions")
#         classif = self.od.get_column_as_matrix("Object classification label")
#         
#         # ax 1, All negative
#         all_neg = self.od.get_data(("neg",), "PCA")
#         self.myscatter(ax1, all_neg[:, 0], all_neg[:, 1])
#         ax1.set_title("All negative positions")
#         
#         # ax 2, training data
#         train_matrix = self.od.last_training_matrix
#         self.myscatter(ax2, train_matrix[:, 0], train_matrix[:, 1])
#         ax2.set_title("Training_data")
#         
#         # ax 3, classifier
#         clf = self.od.classifier
#         #hpd = self.od.get_column_as_matrix("Hyperplane distance") # for SVM
#         xmin, xmax = matrix[:,0].min(), matrix[:,0].max()
#         ymin, ymax = matrix[:,1].min(), matrix[:,1].max()
#         xx, yy = numpy.meshgrid(numpy.linspace(xmin, xmax, 100), numpy.linspace(ymin, ymax, 100))
#         Z = clf.decision_function(numpy.c_[xx.ravel(), yy.ravel()]) # for svm
#         Z = clf.kde.score_samples(numpy.c_[xx.ravel(), yy.ravel()])
#         Z = Z.reshape(xx.shape)    
#         Z = numpy.exp(Z)
#         ax3.contourf(xx, yy, Z, levels=numpy.linspace(Z.min(), numpy.exp(clf.direct_thresh), 7), cmap=plt.get_cmap('Reds'))
#         ax3.contour(xx, yy, Z, levels=[numpy.exp(clf.direct_thresh)], linewidths=1, colors='red')
#         ax3.contourf(xx, yy, Z, levels=[numpy.exp(clf.direct_thresh), Z.max()], colors='orange')
#         ax3.set_title("Outlier density and cutoff\n%s" % self.od.classifier.describe())
#  
#         # ax 4, predicted inliers
#         self.myscatter(ax4, matrix[outlier==1, 0], matrix[outlier==1, 1])
#         ax4.contour(xx, yy, Z, levels=[numpy.exp(clf.direct_thresh)], linewidths=1, colors='red')
#         ax4.set_title("Predicted inliers")
#         
#         # ax 5, predicted outliers
#         self.myscatter(ax5, matrix[outlier==-1, 0], matrix[outlier==-1, 1])
#         ax5.contour(xx, yy, Z, levels=[numpy.exp(clf.direct_thresh)], linewidths=1, colors='red')
#         ax5.set_title("Predicted outliers")
#     
#         # ax 6 wrong outliers
#         self.myscatter(ax6, matrix[numpy.logical_and(outlier==-1, classif<5), 0], matrix[numpy.logical_and(outlier==-1, classif<5), 1])
#         ax6.contour(xx, yy, Z, levels=[numpy.exp(clf.direct_thresh)], linewidths=1, colors='red')
#         ax6.set_title("Wrong outliers")
#         
#         # ax7 Interphase
#         self.myscatter(ax7, matrix[classif==0, 0], matrix[classif==0, 1])
#         ax7.set_title("Supervised: Interphase")
#         ax7.contour(xx, yy, Z, levels=[numpy.exp(clf.direct_thresh)], linewidths=1, colors='red')
#         
#         # ax8 Mitotic
#         self.myscatter(ax8, matrix[numpy.logical_and(classif<5, classif>0), 0], matrix[numpy.logical_and(classif<5, classif>0), 1])
#         ax8.set_title("Supervised: Mitotic")
#         ax8.contour(xx, yy, Z, levels=[numpy.exp(clf.direct_thresh)], linewidths=1, colors='red')
#         
#         #ax 9 Phenotypes
#         self.myscatter(ax9, matrix[classif>=5, 0], matrix[classif>=5, 1])
#         ax9.set_title("Supervised: Phenotypes")
#         ax9.contour(xx, yy, Z, levels=[numpy.exp(clf.direct_thresh)], linewidths=1, colors='red')
#         
#         ax7.set_xlabel("Principle component 1")
#         ax8.set_xlabel("Principle component 1")
#         ax9.set_xlabel("Principle component 1")
#         
#         ax1.set_ylabel("Principle component 2")
#         ax4.set_ylabel("Principle component 2")
#         ax7.set_ylabel("Principle component 2")
# 
# 
#         for ax in all_ax:
#             ax.set_xlim(xmin, xmax)
#             ax.set_ylim(ymin, ymax)
#         
#         plt.tight_layout()
#         
#         plt.savefig(self.od.output("overview_%s.png"  % self.od.classifier.describe()))
#         
#     
#     def analyze(self):        
#         split = 5
#         
# #         # GMM
# #         for k in [2,10,100]:
# #             self.od.train(classifier_class=OneClassGMM, k=k)
# #             self.od.predict()
# #             self.od.compute_outlyingness()
# #             cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
# #             self.evaluate(cm, cms, thrs, split) 
# # #          
# #         # Mahala
# #         self.od.train(classifier_class=OneClassMahalanobis)
# #         self.od.predict()
# #         self.od.compute_outlyingness()
# #         cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
# #         self.evaluate(cm, cms, thrs, split)   
# #          
#         # KDE 
# #         for bandwidth in numpy.linspace(1, 6, 6):
# #             self.od.train(classifier_class=OneClassKDE, bandwidth=bandwidth, feature_set="PCA")
# #             self.od.predict(feature_set="PCA")
# #             self.od.compute_outlyingness()
# #             cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
# #             print "***"* 10, bandwidth
# #             cm_n = self.normalize(cm)
# #             for k,v in self.get_stats(cm,5).items():
# #                 print k, v
# #                 print 
# #             self.evaluate(cm, cms, thrs, split)
# # #                 self.evaluate(cms, thrs, split)     
# #             self.show_feature_space2()      
#           
#         # One class svm
# #         for nu in numpy.linspace(0.9, 0.999, 10):
# 
#         for nu in [0.1, 0.01, 0.05]:
#             for gamma in [4**-k for k in range(2,8)]:
#                 feature_set = "Object features"
#                 feature_set = "PCA"
#                 self.od.train(classifier_class=OneClassSVM_SKL, gamma=gamma, nu=nu, kernel="rbf", feature_set=feature_set)
#                 self.od.predict(feature_set=feature_set)
#                 self.od.compute_outlyingness()
#                 cm, cms, thrs = self.od.get_sl_od_confusion_matrix()
#                 print "***"* 10, nu, gamma
#                 cm_n = self.normalize(cm)
#                 for k,v in self.get_stats(cm,5).items():
#                     print k, v
#                     print 
#                 self.evaluate(cm, cms, thrs, split)
# #                 self.evaluate(cms, thrs, split)     
#                 self.show_feature_space2()    

    
if __name__ == "__main__":    
    #od = OutlierDetection("embo", **EXP['EMBO_2012_Group1_001'])
    od = OutlierDetection("embo", **EXP['EMBO_2012_Group1_001_full'])
    od.set_max_training_sample_size(16000)
    od.read_feature(remove_feature=(16,  17,  18,  62,  92, 122, 152, 173, 174, 177, 178, 197, 198, 201, 202))
    od.pca_run()

    feature_set="PCA"
#             
    od.train(classifier_class=OneClassSVM_SKL, gamma=0.014, nu=0.14, kernel="rbf", feature_set=feature_set)
    od.predict(feature_set=feature_set)
    od.compute_outlyingness()
    
    od_plots = OutlierDetectionSingleCellPlots(od)
    od_plots.evaluate(5)
    
    od_cluster = OutlierClusterPlots(od)
    od_cluster.cluster_on_all(ClusterGMM, feature_names=None, n_components=2, covariance_type="diag")
    od_plots.show_feature_space(5,('neg', 'pos'))