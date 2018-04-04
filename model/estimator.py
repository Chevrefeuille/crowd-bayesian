import numpy as np
from model.constants import *
from model import tools

def find_bins_distance(distances):
    """
    Compute distance histogram
    """
    n_bins = round((D_MAX_TOLERABLE - D_MIN_TOLERABLE) / D_BIN_SIZE)
    edges = np.linspace(D_MIN_TOLERABLE, D_MAX_TOLERABLE, n_bins)
    bins = np.digitize(distances, edges)
    return bins
    
def find_bins_velocity(velocities):
    """
    Compute velocity histogram
    """
    n_bins = round((VEL_MAX_TOLERABLE - VEL_MIN_TOLERABLE) / VEL_BIN_SIZE)
    edges = np.linspace(VEL_MIN_TOLERABLE, VEL_MAX_TOLERABLE, n_bins)
    bins = np.digitize(velocities, edges)
    return bins

def find_bins_velocity_differences(velocity_differences):
    """
    Compute velocity histogram
    """
    n_bins = round((VELDIFF_MAX_TOLERABLE - VELDIFF_MIN_TOLERABLE) / VELDIFF_BIN_SIZE)
    edges = np.linspace(VELDIFF_MIN_TOLERABLE, VELDIFF_MAX_TOLERABLE, n_bins)
    bins = np.digitize(velocity_differences, edges)
    return bins

def find_bins_velocities_dotproduct(velocities_dotproduct):
    """
    Compute velocities dot product histogram
    """
    n_bins = round((VVDOT_MAX_TOLERABLE - VVDOT_MIN_TOLERABLE) / VVDOT_BIN_SIZE)
    edges = np.linspace(VVDOT_MIN_TOLERABLE, VVDOT_MAX_TOLERABLE, n_bins)
    bins = np.digitize(velocities_dotproduct, edges)
    return bins

def find_bins_velocity_distance_dotproduct(velocity_distance_dotproduct):
    """
    Compute velocity/distance dot product histogram
    """
    n_bins = round((VDDOT_MAX_TOLERABLE - VDDOT_MIN_TOLERABLE) / VDDOT_BIN_SIZE)
    edges = np.linspace(VDDOT_MIN_TOLERABLE, VDDOT_MAX_TOLERABLE, n_bins)
    bins = np.digitize(velocity_distance_dotproduct, edges)
    return bins

def find_bins_average_height(average_heights):
    """
    Compute velocity/distance dot product histogram
    """
    n_bins = round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE)
    edges = np.linspace(HEIGHT_MIN_TOLERABLE, HEIGHT_MAX_TOLERABLE, n_bins)
    bins = np.digitize(average_heights, edges)
    return bins

def find_bins_height_diff(height_differences):
    """
    Compute velocity/distance dot product histogram
    """
    n_bins = round((HEIGHTDIFF_MAX_TOLERABLE - HEIGHTDIFF_MIN_TOLERABLE) / HEIGHTDIFF_BIN_SIZE)
    edges = np.linspace(HEIGHTDIFF_MIN_TOLERABLE, HEIGHTDIFF_MAX_TOLERABLE, n_bins)
    bins = np.digitize(height_differences, edges)
    return bins

def find_bins_short_height(short_heights):
    """
    Compute velocity/distance dot product histogram
    """
    n_bins = round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE)
    edges = np.linspace(HEIGHT_MIN_TOLERABLE, HEIGHT_MAX_TOLERABLE, n_bins)
    bins = np.digitize(short_heights, edges)
    return bins

def find_bins_tall_height(tall_heights):
    """
    Compute velocity/distance dot product histogram
    """
    n_bins = round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE)
    edges = np.linspace(HEIGHT_MIN_TOLERABLE, HEIGHT_MAX_TOLERABLE, n_bins)
    bins = np.digitize(tall_heights, edges)
    return bins


def find_bins(file_path):
    data = np.load(file_path)
    dataA, dataB = tools.extract_individual_data(data)
    dAB, vG, vDiff, vvdot, vddot,  hAvg, hDiff, h_short, h_tall = tools.compute_parameters(dataA, dataB)
    binsD = find_bins_distance(dAB)
    binsvG = find_bins_velocity(vG)
    binsvDiff = find_bins_velocity_differences(vDiff)
    binsvvdot = find_bins_velocities_dotproduct(vvdot)
    binsvddot = find_bins_velocity_distance_dotproduct(vddot)
    binsavgH = find_bins_average_height(hAvg)
    binsHDiff = find_bins_height_diff(hDiff)
    binsshortH = find_bins_short_height(h_short)
    binstallH = find_bins_tall_height(h_tall)
    return binsD, binsvG, binsvDiff, binsvvdot, binsvddot, binsavgH, binsHDiff, binsshortH, binstallH

def compute_probability_class(file_path, pdf_dry_d, pdf_koi_d, pdf_dry_grp_v, pdf_koi_grp_v, pdf_dry_veldiff, pdf_koi_veldiff, pdf_dry_heightdiff, pdf_koi_heightdiff, alpha):
    bins = find_bins(file_path)
    p0dry, p0koi = 0.5, 0.5
    n_data = len(bins[0])
    p_post = np.zeros((n_data, 2))
    for j in range(n_data):
        p_likel_dry = pdf_dry_d[bins[0][j]] * pdf_dry_grp_v[bins[1][j]] * pdf_dry_veldiff[bins[2][j]] * pdf_dry_heightdiff[bins[6][j]]
        p_likel_koi = pdf_koi_d[bins[0][j]] * pdf_koi_grp_v[bins[1][j]] * pdf_koi_veldiff[bins[2][j]] * pdf_koi_heightdiff[bins[6][j]]

        if j == 0:
            p_prior_dry = p0dry
            p_prior_koi = p0koi
        else:
            p_prior_dry = alpha * p0dry + (1 - alpha) * p_post[j-1][0]
            p_prior_koi = alpha * p0koi + (1 - alpha) * p_post[j-1][1]

        p_cond_dry = p_likel_dry * p_prior_dry
        p_cond_koi = p_likel_koi * p_prior_koi

        temp = p_cond_dry + p_cond_koi

        p_cond_dry /= temp
        p_cond_koi /= temp

        p_post[j] = [p_cond_dry,p_cond_koi]

    return np.mean(p_post[:,0]), np.mean(p_post[:,1])


def compute_accuracy(dry_set, koi_set, pdf_dry_d, pdf_koi_d, pdf_dry_grp_v, pdf_koi_grp_v, pdf_dry_veldiff, pdf_koi_veldiff, pdf_dry_heightdiff, pdf_koi_heightdiff, alpha):
    dry_wrong, dry_right = 0, 0
    for file_path in dry_set:
        p_dry, p_koi = compute_probability_class(file_path, pdf_dry_d, pdf_koi_d, pdf_dry_grp_v, pdf_koi_grp_v, pdf_dry_veldiff, pdf_koi_veldiff, pdf_dry_heightdiff, pdf_koi_heightdiff, alpha)
        if p_dry < 0.5:
            dry_wrong += 1
        else:
            dry_right += 1

    koi_wrong, koi_right = 0, 0
    for file_path in koi_set: 
        p_dry, p_koi = compute_probability_class(file_path, pdf_dry_d, pdf_koi_d, pdf_dry_grp_v, pdf_koi_grp_v, pdf_dry_veldiff, pdf_koi_veldiff, pdf_dry_heightdiff, pdf_koi_heightdiff, alpha)
        if p_koi < 0.5:
            koi_wrong += 1
        else:
            koi_right += 1
    return dry_wrong, dry_right, koi_wrong, koi_right

    

