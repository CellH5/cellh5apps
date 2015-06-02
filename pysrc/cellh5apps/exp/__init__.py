from matthias.experiments import EXP as matthias
from sara.experiments import EXP as sara
from embo2014.experiments import EXP as embo
from claudia.experiments import EXP as claudia
from mitocheck.experiments import EXP as mito_1

EXP = {}
for d in [matthias, sara, embo, claudia, mito_1]:
    EXP.update(d)
    
if __name__ == "__main__":
    print "Available experiments:"
    print "----------------------------"
    for e in EXP:
        print e
