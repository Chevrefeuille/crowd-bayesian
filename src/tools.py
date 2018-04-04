import numpy as np
import h5py 
import math
from os import listdir
import random

def pick_files():
    """
    This function eliminates some files from the analysis 

    The criterion is having at least 60 lines (observation points)
    after thresholding with the following: 
    (ie 60 lines in data file, 60 * 0.5 sec = 30 sec)

    1. Distance between partners 
    2. Velocity of group 
    3. Height of partners
    maybe some others...
    """

def convert(data):
    """
    Convert data to use appropriate units
    """
    data[:, 5:7] /= 1000 # from cm to m
    return data

def load_data_scilab(file_name):
    """
    Load data contained in a given file
    """
    f = h5py.File(file_name, 'r') 
    data = f.get('data') 
    data = np.array(data)
    return data.T

def rotate_vectors(dxAB, dxBA, dyAB, dyBA, vxG, vyG):
    """
    Rotate to vectors to obtain an output vector whose x component is aligned with the group velocity
    """
    dAB = (dxAB**2 + dyAB**2)**.5
    dx_rAB, dy_rAB, dx_rBA, dy_rBA = dxAB, dyAB, dxBA, dyBA

    for j in range(len(dx_rAB)):
        magnitude = (vxG[j]*vxG[j] + vyG[j]*vyG[j])**.5
        if magnitude != 0:
            dx_rAB[j] = (vxG[j]*dxAB[j] +  vyG[j]*dyAB[j]) / magnitude
            dy_rAB[j] = (vyG[j]*dxAB[j] + -vxG[j]*dyAB[j]) / magnitude
 
            dx_rBA[j] = (vxG[j]*dxBA[j] +  vyG[j]*dyBA[j]) / magnitude
            dy_rBA[j] = (vyG[j]*dxBA[j] + -vxG[j]*dyBA[j]) / magnitude

    dABx = [dx_rAB, dx_rBA]
    dABy = [dy_rAB, dy_rBA]
    abs_dABy = np.abs(dABy)

    return [dAB, dABx, dABy, abs_dABy]

def extract_individual_data(data):
    """
    Separate data into different array for each individual
    """
    ids = set(data[:,1])
    id_A = min(ids)
    id_B = max(ids)
    dataA = data[data[:,1] == id_A, :]
    dataB = data[data[:,1] == id_B, :]
    # print(np.shape(dataA))
    return dataA, dataB


def compute_parameters(dataA, dataB):
    """
    Compute the parameters that are used in the Bayesian inference
    """
    dxA, dyA = dataA[:, 2], dataA[:, 3]
    dxB, dyB = dataB[:, 2], dataB[:, 3]
    dxAB, dyAB = dxB - dxA, dyB - dyA
    dxBA, dyBA = -dxAB,  -dyAB
    vxA, vyA = dataA[:, 5], dataA[:, 6]
    vxB, vyB = dataB[:, 5], dataB[:, 6]
    # group velocity
    vxG, vyG = (vxA + vxB) / 2, (vyA + vyB) / 2
    vG = (vxG**2 + vyG**2)**.5
    # velocities difference
    vxDiff = (vxA - vxB) / 2
    vyDiff = (vyA - vyB) / 2
    vDiff = (vxDiff**2 + vyDiff**2)**.5
    # velocities dot product
    vvdotAB = np.arctan2(vyA, vxA) - np.arctan2(vyB, vxB)
    vvdot = vvdotAB % (2 * math.pi)
    vvdot[vvdot > math.pi] = vvdot[vvdot > math.pi] - 2 * math.pi # interval is [-pi, pi]
    vvdot[vvdot < -math.pi] = vvdot[vvdot < -math.pi] + 2 * math.pi
    # velocities/distance dot product
    vddotA = np.arctan2(dyAB, dxAB) - np.arctan2(vyA, vxA)
    vddotB = np.arctan2(dyBA, dxBA) - np.arctan2(vyB, vxB)
    vddot = np.concatenate((vddotA, vddotB), axis=0)
    vddot = vddot % (2 * math.pi)
    vddot[vddot > math.pi] = vddot[vddot > math.pi] - 2 * math.pi # interval is [-pi, pi]
    vddot[vddot < -math.pi] = vddot[vddot < -math.pi] + 2 * math.pi
    # heights
    hA, hB = dataA[:, 4], dataB[:, 4]
    hAvg = (hA + hB) / 2
    hDiff = np.abs(hA - hB)
    if np.mean(hA) < np.mean(hB):
        h_short = hA
        h_tall = hB
    else:
        h_short = hB
        h_tall = hA
    # rotate the vectors
    [dAB, dABx, dABy, abs_dABy] = rotate_vectors(dxAB, dxBA, dyAB, dyBA, vxG, vyG)
    return dAB, vG, vDiff, vvdot, vddot,  hAvg, hDiff, h_short, h_tall


def get_data_set(data_path):
    dry_path = data_path + 'doryo/'
    koi_path = data_path + 'koibito/'
    dry_set = [dry_path + f for f in listdir(dry_path) if 'threshold' in f]
    koi_set = [koi_path + f for f in listdir(koi_path) if 'threshold' in f]
    return dry_set, koi_set

def shuffle_data_set(dry_set, koi_set, train_ratio):
    n_dry, n_koi = len(dry_set), len(koi_set)
    n_dry_train, n_koi_train = round(train_ratio * n_dry), round(train_ratio * n_koi)

    shuffled_dry = random.sample(dry_set, n_dry)
    shuffled_koi = random.sample(koi_set, n_koi)
    
    return shuffled_dry[:n_dry_train], shuffled_dry[n_dry_train:], shuffled_koi[:n_koi_train], shuffled_koi[n_koi_train:]
    