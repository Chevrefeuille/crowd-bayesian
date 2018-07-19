import glob
from os import remove

if __name__ == "__main__":
    for file_path in glob.iglob('data/classes/kazoku_*/*.dat', recursive=True):
        print('Removing {}'.format(file_path))
        remove(file_path)