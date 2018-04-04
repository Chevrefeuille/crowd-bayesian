import tools
from constants import *
import numpy as np
import math
import glob

def threshold(data):
    """
    Apply threshold to the various parameters
    """
    dataA, dataB = tools.extract_individual_data(data)
    dataA, dataB = threshold_distance(dataA, dataB)
    dataA, dataB = threshold_velocity(dataA, dataB)
    dataA, dataB = threshold_data_veldiff(dataA, dataB)
    dataA, dataB = threshold_vvdot(dataA, dataB)
    dataA, dataB = threshold_height(dataA, dataB)

    vxA, vyA = dataA[:, 5], dataA[:, 6]
    vxB, vyB = dataB[:, 5], dataB[:, 6]

    data2 = np.concatenate((dataA, dataB), axis=0)
    return np.concatenate((dataA, dataB), axis=0)

def threshold_distance(dataA, dataB):
    """
    Apply a threshold on the distance to the given data
    """
    distAB = ((dataA[:, 2] - dataB[:, 2])**2 + (dataA[:, 3] - dataB[:, 3])**2)**.5
    thresholdA = dataA[distAB < DISTANCE_THRESHOLD, :]
    thresholdB = dataB[distAB < DISTANCE_THRESHOLD, :]

    return thresholdA, thresholdB

def threshold_velocity(dataA, dataB):
    """
    Apply a threshold on the velocity to the given data
    """
    timeA = dataA[:, 0]
    timeB = dataB[:, 0]

    vxA, vyA = dataA[:, 5], dataA[:, 6]
    vxB, vyB = dataB[:, 5], dataB[:, 6]

    vA = (vxA**2 + vyA**2)**.5
    vB = (vxB**2 + vyB**2)**.5

    pre_thresholdA = timeA[vA > VELOCITY_THRESHOLD]
    pre_thresholdB = timeB[vB > VELOCITY_THRESHOLD]
    inter_threshold = np.intersect1d(pre_thresholdA, pre_thresholdB)

    threshold_boolA = np.isin(timeA, inter_threshold)
    threshold_boolB = np.isin(timeB, inter_threshold)

    thresholdA = dataA[threshold_boolA, :]
    thresholdB = dataB[threshold_boolB, :]

    return thresholdA, thresholdB

def threshold_data_veldiff(dataA, dataB):
    """
    Apply a threshold on the velocity difference to the given data
    """
    vxA, vyA = dataA[:, 5], dataA[:, 6]
    vxB, vyB = dataB[:, 5], dataB[:, 6]
    veldiff_x = (vxA - vxB) / 2
    veldiff_y = (vyA - vyB) / 2
    veldiff = (veldiff_x**2 + veldiff_y**2)**.5
    threshold_bool = np.logical_and(VELDIFF_MIN_TOLERABLE < veldiff, veldiff < VELDIFF_MAX_TOLERABLE)
    thresholdA = dataA[threshold_bool, :]
    thresholdB = dataB[threshold_bool, :]

    vxA, vyA = thresholdA[:, 5], thresholdA[:, 6]
    vxB, vyB = thresholdB[:, 5], thresholdB[:, 6]

    veldiff_x = (vxA - vxB) / 2
    veldiff_y = (vyA - vyB) / 2
    veldiff = (veldiff_x**2 + veldiff_y**2)**.5
    return thresholdA, thresholdB

def threshold_vvdot(dataA, dataB):
    """
    Apply a threshold on the dot product of the velocities to the given data
    """
    vxA, vyA = dataA[:, 5], dataA[:, 6]
    vxB, vyB = dataB[:, 5], dataB[:, 6]

    vA = (vxA**2 + vyA**2)**.5
    vB = (vxB**2 + vyB**2)**.5
    vxDiff = (vxA - vxB) / 2
    vyDiff = (vyA - vyB) / 2
    vDiff = (vxDiff**2 + vyDiff**2)**.5

    vvdotAB = np.arctan2(vyA, vxA) - np.arctan2(vyB, vxB)
    # limit vvdot to the interval [-pi, pi]
    vvdot = vvdotAB % (2 * math.pi)
    vvdot[vvdot > math.pi] = vvdot[vvdot > math.pi] - 2 * math.pi # interval is [-pi, pi]
    vvdot[vvdot < -math.pi] = vvdot[vvdot < -math.pi] + 2 * math.pi

    threshold_bool = np.logical_and(VDDOT_MIN_TOLERABLE < vvdot, vvdot < VDDOT_MAX_TOLERABLE)
    thresholdA = dataA[threshold_bool, :]
    thresholdB = dataB[threshold_bool, :]
    
    vxA, vyA = thresholdA[:, 5], thresholdA[:, 6]
    vxB, vyB = thresholdB[:, 5], thresholdB[:, 6]

    vA = (vxA**2 + vyA**2)**.5
    vB = (vxB**2 + vyB**2)**.5
    vxDiff = (vxA - vxB) / 2
    vyDiff = (vyA - vyB) / 2
    vDiff = (vxDiff**2 + vyDiff**2)**.5
    return thresholdA, thresholdB

def threshold_height(dataA, dataB):
    """
    Apply a threshold on the height to the given data
    """
    hA = dataA[:, 4]
    hB = dataB[:, 4]

    condA = np.logical_and(HEIGHT_MIN_TOLERABLE < hA, hA < HEIGHT_MAX_TOLERABLE)
    condB = np.logical_and(HEIGHT_MIN_TOLERABLE < hB, hB < HEIGHT_MAX_TOLERABLE)
    cond = np.logical_and(condA, condB)

    thresholdA = dataA[cond, :]
    thresholdB = dataB[cond, :]

    return thresholdA, thresholdB

if __name__ == "__main__":
    for file_path in glob.iglob('../data/**/*.dat', recursive=True):
        if 'threshold' not in file_path and 'day' in file_path:
            data = tools.load_data_scilab(file_path)
            data = tools.convert(data)
            threshold_data = threshold(data)
            if np.shape(threshold_data[:,0])[0] > 60:
                threshold_file_path = '{}_threshold.dat'.format(file_path.rstrip('.dat'))
                print('Applying thresholds to {} and storing in {}'.format(file_path, threshold_file_path))
                with open(threshold_file_path, 'wb') as outfile:
                    np.save(outfile, threshold_data)