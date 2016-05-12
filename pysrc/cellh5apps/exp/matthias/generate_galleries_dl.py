
import numpy
from cellh5apps.outlier import OutlierDetection
from cellh5apps.exp import EXP
import gzip
import cPickle as pickle
import pandas

import cellh5
cellh5.GALLERY_SIZE = 80

import h5py


    
def generate_random(group, how_many, size):
    od = OutlierDetection("predrug_a_8_int_only", **EXP['matthias_predrug_a8_plates_1_8'])
    
    
    neg_mapping = od.mapping[od.mapping["Group"] == group]
    
    cnt = 0
    
    all_gals = []
    all_klass = []
    
    while True:
        rn = numpy.random.randint(len(neg_mapping))
        plate, well, site, gene = neg_mapping.iloc[rn][["Plate", "Well", "Site", "Gene Symbol"]]
        
        ch5_pos = od.get_ch5_position(plate, well, site)
        
        n_cells = len(ch5_pos.get_object_idx())
        print plate, well, site, gene, n_cells
        if n_cells == 0:
            continue
        gals = ch5_pos.get_gallery_image(range(n_cells), size=size*2)[::2, ::2].T.reshape((n_cells, size,size))
        all_gals.append(gals)
        
        klass= ch5_pos.get_class_label(range(n_cells))
        
        all_klass.append(klass)

        cnt += n_cells
        if cnt > how_many:
            break
        
    all_gals = numpy.concatenate(all_gals)
    all_klass = numpy.concatenate(all_klass)
    
    h = h5py.File('predrug_%s_40x40_%dk.h5' % (group, how_many / 1000), 'w')
    h.create_dataset('galleries', data=all_gals)
    h.create_dataset('labels', data=all_klass)
    h.close()

def generate_cellinder(size):
    od = OutlierDetection("predrug_a_8_int_only", **EXP['matthias_predrug_a8_plates_1_8'])
    
    cellinder_file = r"C:\Users\sommerc\cellh5apps\pysrc\cellh5apps\cellinder\predrug_a8\d_15-02-24-15-37\__anotation.txt"
    
    cellinder = pandas.read_csv(cellinder_file, sep="\t")
    
    
    cnt = 0
    norm_gals = []
    lab = []
    
    for i, row in cellinder.iterrows():
        plate, well, site, gene, ch5_idx, class_label, correct = row[["Plate", "Well", "Site", "Gene", "CellH5_index", "Label", "Correct"]]
        class_label = int(class_label)
        correct = bool(correct)
        
        ch5_pos = od.get_ch5_position(plate, well, site)
        print i,  correct, plate, well, site, gene, ch5_idx, class_label, correct 
        if correct:
            gals = ch5_pos.get_gallery_image(ch5_idx, size=size*2)[::2, ::2].reshape((1,size, size))

            norm_gals.append(gals)
            lab.append(class_label)

        
        
        
    norm_gals = numpy.concatenate(norm_gals)
    lab = numpy.array(lab)
    
    h = h5py.File('predrug_cellinder_40x40.h5', 'w')
    h.create_dataset('galleries', data=norm_gals)
    h.create_dataset('labels', data=lab)
    h.close()


    
def generate_per_class(size=60, max_samples=100000, out_folder='C:/Users/sommerc/src/deep_learning/input_data', sub_sampling=None):
    od = OutlierDetection("gnerate_galleries_for_dl", **EXP['matthias_predrug_a8_plates_1_8'])
    
#     neg_mapping = od.mapping[od.mapping["Group"] == "target"]
    neg_mapping = od.mapping
    
    cnt = 0
    
    all_gals = []
    all_class = []
    
    while True:
        rn = numpy.random.randint(len(neg_mapping))
        plate, well, site, gene = neg_mapping.iloc[rn][["Plate", "Well", "Site", "Gene Symbol"]]
        
        ch5_pos = od.get_ch5_position(plate, well, site)
        
        n_cells = len(ch5_pos.get_object_idx())
        print plate, well, site, gene, "\t", n_cells, "\t", cnt
        if n_cells == 0:
            continue
        
        c_idx = range(n_cells)
        if sub_sampling is not None:
            gals = ch5_pos.get_gallery_image(c_idx, size=size)[::sub_sampling, ::sub_sampling].T.reshape((n_cells, (size/sub_sampling)**2))
        else:
            gals = ch5_pos.get_gallery_image(c_idx, size=size).T.reshape((n_cells, size**2))
            
        
        
        klass= ch5_pos.get_class_label(c_idx)
        
        all_gals.append(gals)
        all_class.append(klass)

        cnt += n_cells
        if cnt > max_samples:
            break
        
    all_gals = numpy.concatenate(all_gals)[:max_samples,:]
    all_class = numpy.concatenate(all_class)[:max_samples]
    
    h = h5py.File('galleries_%03dx%03d.h5', 'w')
    h.create_dataset('galleries', data=all_gals)
    h.create_dataset('labels', data=all_class)
    h.close()
    
    
    
#     pickle.dump(all_gals,  gzip.open('neg_galleries.pkl.gz', 'wb'))
#     pickle.dump(all_class, gzip.open('neg_classlabels.pkl.gz', 'wb'))
        
    
    
if __name__ == "__main__":  
#     generate_random("neg", 50000, 40)
#     generate_random("target", 100000, 40)
    generate_cellinder(40)
        
    print "Fini"

        
        
        