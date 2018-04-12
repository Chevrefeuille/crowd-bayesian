import glob
from model import tools
import numpy as np
from shutil import copyfile
    
if __name__ == "__main__":
    has_child = {}
    for annotation_path in glob.iglob('data/annotations/*.dat'):
        with open(annotation_path, 'r') as f:
            for line in f:
                data = list(map(int, line.split()))
                id_A, id_B, min_age = data[0], data[1], data[10]
                if min_age < 15:
                    has_child[id_A] = True
                    has_child[id_B] = True
                else:
                    has_child[id_A] = False
                    has_child[id_B] = False


    for file_path in glob.iglob('data/classes/kazoku/*.dat'):
        if 'threshold' not in file_path:
            data = tools.load_data_scilab(file_path)
            ids = set(data[:,1])
            id_A = min(ids)
            id_B = max(ids)
            if has_child[id_A] or has_child[id_B]:
                subclass = 'child'
            else:
                subclass = 'adults'
            file_name = file_path.split('/')[-1]
            dest_path = 'data/classes/kazoku_{}/{}'.format(subclass, file_name)
            copyfile(file_path, dest_path)
            print('Copying {} to {}'.format(file_name, dest_path))


                
            