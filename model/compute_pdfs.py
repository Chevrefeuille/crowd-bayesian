from model.constants import *
import numpy as np
from model import tools
import math
import glob
import matplotlib.pyplot as plt

def compute_distance_histogram(distances):
    """
    Compute distance histogram
    """
    n_bins = round((D_MAX_TOLERABLE - D_MIN_TOLERABLE) / D_BIN_SIZE) + 1
    edges = np.linspace(D_MIN_TOLERABLE, D_MAX_TOLERABLE, n_bins)
    histog = np.histogram(distances, edges)
    return histog[0]
    
def compute_velocity_histogram(velocities):
    """
    Compute velocity histogram
    """
    n_bins = round((VEL_MAX_TOLERABLE - VEL_MIN_TOLERABLE) / VEL_BIN_SIZE) + 1
    edges = np.linspace(VEL_MIN_TOLERABLE, VEL_MAX_TOLERABLE, n_bins)
    histog = np.histogram(velocities, edges)
    return histog[0]

def compute_velocity_differences_histogram(velocity_differences):
    """
    Compute velocity histogram
    """
    n_bins = round((VELDIFF_MAX_TOLERABLE - VELDIFF_MIN_TOLERABLE) / VELDIFF_BIN_SIZE) + 1
    edges = np.linspace(VELDIFF_MIN_TOLERABLE, VELDIFF_MAX_TOLERABLE, n_bins)
    histog = np.histogram(velocity_differences, edges)
    return histog[0]

def compute_velocities_dotproduct_histogram(velocities_dotproduct):
    """
    Compute velocities dot product histogram
    """
    n_bins = round((VVDOT_MAX_TOLERABLE - VVDOT_MIN_TOLERABLE) / VVDOT_BIN_SIZE) + 1
    edges = np.linspace(VVDOT_MIN_TOLERABLE, VVDOT_MAX_TOLERABLE, n_bins)
    histog = np.histogram(velocities_dotproduct, edges)
    return histog[0]

def compute_velocity_distance_dotproduct_histogram(velocity_distance_dotproduct):
    """
    Compute velocity/distance dot product histogram
    """
    n_bins = round((VDDOT_MAX_TOLERABLE - VDDOT_MIN_TOLERABLE) / VDDOT_BIN_SIZE) + 1
    edges = np.linspace(VDDOT_MIN_TOLERABLE, VDDOT_MAX_TOLERABLE, n_bins)
    histog = np.histogram(velocity_distance_dotproduct, edges)
    return histog[0]

def compute_average_height_histogram(average_heights):
    """
    Compute velocity/distance dot product histogram
    """
    n_bins = round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE) + 1
    edges = np.linspace(HEIGHT_MIN_TOLERABLE, HEIGHT_MAX_TOLERABLE, n_bins)
    histog = np.histogram(average_heights, edges)
    return histog[0]

def compute_height_diff_histogram(height_differences):
    """
    Compute velocity/distance dot product histogram
    """
    n_bins = round((HEIGHTDIFF_MAX_TOLERABLE - HEIGHTDIFF_MIN_TOLERABLE) / HEIGHTDIFF_BIN_SIZE) + 1
    edges = np.linspace(HEIGHTDIFF_MIN_TOLERABLE, HEIGHTDIFF_MAX_TOLERABLE, n_bins)
    histog = np.histogram(height_differences, edges)
    return histog[0]

def compute_short_height_histogram(short_heights):
    """
    Compute velocity/distance dot product histogram
    """
    n_bins = round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE) + 1
    edges = np.linspace(HEIGHT_MIN_TOLERABLE, HEIGHT_MAX_TOLERABLE, n_bins)
    histog = np.histogram(short_heights, edges)
    return histog[0]

def compute_tall_height_histogram(tall_heights):
    """
    Compute velocity/distance dot product histogram
    """
    n_bins = round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE) + 1
    edges = np.linspace(HEIGHT_MIN_TOLERABLE, HEIGHT_MAX_TOLERABLE, n_bins)
    histog = np.histogram(tall_heights, edges)
    return histog[0]

def compute_histograms(file_path):
    """
    Compute the histograms for the given data file
    """
    data = np.load(file_path)
    n_lines = np.shape(data)[0] // 2
    if n_lines < 30:
        hD = np.zeros((round((D_MAX_TOLERABLE - D_MIN_TOLERABLE) / D_BIN_SIZE)))
        hvG = np.zeros((round((VEL_MAX_TOLERABLE - VEL_MIN_TOLERABLE) / VEL_BIN_SIZE)))
        hvDiff = np.zeros((round((VELDIFF_MAX_TOLERABLE - VELDIFF_MIN_TOLERABLE) / VELDIFF_BIN_SIZE)))
        hvvdot = np.zeros((round((VVDOT_MAX_TOLERABLE - VVDOT_MIN_TOLERABLE) / VVDOT_BIN_SIZE)))
        hvddot = np.zeros((round((VDDOT_MAX_TOLERABLE - VDDOT_MIN_TOLERABLE) / VDDOT_BIN_SIZE)))
        havgH = np.zeros((round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE)))
        hHDiff = np.zeros((round((HEIGHTDIFF_MAX_TOLERABLE - HEIGHTDIFF_MIN_TOLERABLE) / HEIGHTDIFF_BIN_SIZE)))
        hshortH = np.zeros((round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE)))
        htallH = np.zeros((round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE)))
    else:
        dataA, dataB = tools.extract_individual_data(data)
        dAB, vG, vDiff, vvdot, vddot,  hAvg, hDiff, h_short, h_tall = tools.compute_parameters(dataA, dataB)
        hD = compute_distance_histogram(dAB)
        hvG = compute_velocity_histogram(vG)
        hvDiff = compute_velocity_differences_histogram(vDiff)
        hvvdot = compute_velocities_dotproduct_histogram(vvdot)
        hvddot = compute_velocity_distance_dotproduct_histogram(vddot)
        havgH = compute_average_height_histogram(hAvg)
        hHDiff = compute_height_diff_histogram(hDiff)
        hshortH = compute_short_height_histogram(h_short)
        htallH = compute_tall_height_histogram(h_tall)
    return hD, hvG, hvDiff, hvvdot, hvddot, havgH, hHDiff, hshortH, htallH
    


def train(dry_set, koi_set):

    hD_cum_dry = np.zeros((round((D_MAX_TOLERABLE - D_MIN_TOLERABLE) / D_BIN_SIZE)))
    hvG_cum_dry = np.zeros((round((VEL_MAX_TOLERABLE - VEL_MIN_TOLERABLE) / VEL_BIN_SIZE)))
    hvDiff_cum_dry = np.zeros((round((VELDIFF_MAX_TOLERABLE - VELDIFF_MIN_TOLERABLE) / VELDIFF_BIN_SIZE)))
    hvvdot_cum_dry = np.zeros((round((VVDOT_MAX_TOLERABLE - VVDOT_MIN_TOLERABLE) / VVDOT_BIN_SIZE)))
    hvddot_cum_dry = np.zeros((round((VDDOT_MAX_TOLERABLE - VDDOT_MIN_TOLERABLE) / VDDOT_BIN_SIZE)))
    havgH_cum_dry = np.zeros((round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE)))
    hHDiff_cum_dry = np.zeros((round((HEIGHTDIFF_MAX_TOLERABLE - HEIGHTDIFF_MIN_TOLERABLE) / HEIGHTDIFF_BIN_SIZE)))
    hshortH_cum_dry = np.zeros((round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE)))
    htallH_cum_dry = np.zeros((round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE)))

    for file_path in dry_set:
        histograms = compute_histograms(file_path)
        hD_cum_dry += histograms[0]
        hvG_cum_dry += histograms[1]
        hvDiff_cum_dry += histograms[2]
        hvvdot_cum_dry += histograms[3]
        hvddot_cum_dry += histograms[4]
        havgH_cum_dry += histograms[5]
        hHDiff_cum_dry += histograms[6]
        hshortH_cum_dry += histograms[7]
        htallH_cum_dry += histograms[8]
    
    pdf_D_dry = hD_cum_dry / sum(hD_cum_dry) / D_BIN_SIZE
    with open('data/pdfs/pdf_D_dry.dat', 'wb') as outfile:
        np.save(outfile, pdf_D_dry)
    pdf_vG_dry = hvG_cum_dry / sum(hvG_cum_dry) / VEL_BIN_SIZE
    with open('data/pdfs/pdf_vG_dry.dat', 'wb') as outfile:
        np.save(outfile, pdf_vG_dry)
    pdf_vDiff_dry = hvDiff_cum_dry / sum(hvDiff_cum_dry) / VEL_BIN_SIZE
    with open('data/pdfs/pdf_vDiff_dry.dat', 'wb') as outfile:
        np.save(outfile, pdf_vDiff_dry)
    pdf_vvdot_dry = hvvdot_cum_dry / sum(hvvdot_cum_dry) / VVDOT_BIN_SIZE
    with open('data/pdfs/pdf_vvdot_dry.dat', 'wb') as outfile:
        np.save(outfile, pdf_vvdot_dry)
    pdf_vddot_dry = hvddot_cum_dry / sum(hvddot_cum_dry) / VDDOT_BIN_SIZE
    with open('data/pdfs/pdf_vddot_dry.dat', 'wb') as outfile:
        np.save(outfile, pdf_vddot_dry)
    pdf_avgH_dry = havgH_cum_dry / sum(havgH_cum_dry) / HEIGHT_BIN_SIZE
    with open('data/pdfs/pdf_avgH_dry.dat', 'wb') as outfile:
        np.save(outfile, pdf_avgH_dry)
    pdf_HDiff_dry = hHDiff_cum_dry / sum(hHDiff_cum_dry) / HEIGHTDIFF_BIN_SIZE
    with open('data/pdfs/pdf_HDiff_dry.dat', 'wb') as outfile:
        np.save(outfile, pdf_HDiff_dry)
    pdf_avgH_dry = havgH_cum_dry / sum(hshortH_cum_dry) / HEIGHT_BIN_SIZE
    with open('data/pdfs/pdf_avgH_dry.dat', 'wb') as outfile:
        np.save(outfile, pdf_avgH_dry)
    pdf_avgH_dry = havgH_cum_dry / sum(htallH_cum_dry) / HEIGHT_BIN_SIZE
    with open('data/pdfs/pdf_avgH_dry.dat', 'wb') as outfile:
        np.save(outfile, pdf_avgH_dry)

    hD_cum_koi = np.zeros((round((D_MAX_TOLERABLE - D_MIN_TOLERABLE) / D_BIN_SIZE)))
    hvG_cum_koi = np.zeros((round((VEL_MAX_TOLERABLE - VEL_MIN_TOLERABLE) / VEL_BIN_SIZE)))
    hvDiff_cum_koi = np.zeros((round((VELDIFF_MAX_TOLERABLE - VELDIFF_MIN_TOLERABLE) / VELDIFF_BIN_SIZE)))
    hvvdot_cum_koi = np.zeros((round((VVDOT_MAX_TOLERABLE - VVDOT_MIN_TOLERABLE) / VVDOT_BIN_SIZE)))
    hvddot_cum_koi = np.zeros((round((VDDOT_MAX_TOLERABLE - VDDOT_MIN_TOLERABLE) / VDDOT_BIN_SIZE)))
    havgH_cum_koi = np.zeros((round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE)))
    hHDiff_cum_koi = np.zeros((round((HEIGHTDIFF_MAX_TOLERABLE - HEIGHTDIFF_MIN_TOLERABLE) / HEIGHTDIFF_BIN_SIZE)))
    hshortH_cum_koi = np.zeros((round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE)))
    htallH_cum_koi = np.zeros((round((HEIGHT_MAX_TOLERABLE - HEIGHT_MIN_TOLERABLE) / HEIGHT_BIN_SIZE)))

    for file_path in koi_set:
        histograms = compute_histograms(file_path)
        hD_cum_koi += histograms[0]
        hvG_cum_koi += histograms[1]
        hvDiff_cum_koi += histograms[2]
        hvvdot_cum_koi += histograms[3]
        hvddot_cum_koi += histograms[4]
        havgH_cum_koi += histograms[5]
        hHDiff_cum_koi += histograms[6]
        hshortH_cum_koi += histograms[7]
        htallH_cum_koi += histograms[8]
    
    pdf_D_koi = hD_cum_koi / sum(hD_cum_koi) / D_BIN_SIZE
    with open('data/pdfs/pdf_D_koi.dat', 'wb') as outfile:
        np.save(outfile, pdf_D_koi)
    pdf_vG_koi = hvG_cum_koi / sum(hvG_cum_koi) / VEL_BIN_SIZE
    with open('data/pdfs/pdf_vG_koi.dat', 'wb') as outfile:
        np.save(outfile, pdf_vG_koi)
    pdf_vDiff_koi = hvDiff_cum_koi / sum(hvDiff_cum_koi) / VEL_BIN_SIZE
    with open('data/pdfs/pdf_vDiff_koi.dat', 'wb') as outfile:
        np.save(outfile, pdf_vDiff_koi)
    pdf_vvdot_koi = hvvdot_cum_koi / sum(hvvdot_cum_koi) / VVDOT_BIN_SIZE
    with open('data/pdfs/pdf_vvdot_koi.dat', 'wb') as outfile:
        np.save(outfile, pdf_vvdot_koi)
    pdf_vddot_koi = hvddot_cum_koi / sum(hvddot_cum_koi) / VDDOT_BIN_SIZE
    with open('data/pdfs/pdf_vddot_koi.dat', 'wb') as outfile:
        np.save(outfile, pdf_vddot_koi)
    pdf_avgH_koi = havgH_cum_koi / sum(havgH_cum_koi) / HEIGHT_BIN_SIZE
    with open('data/pdfs/pdf_avgH_koi.dat', 'wb') as outfile:
        np.save(outfile, pdf_avgH_koi)
    pdf_HDiff_koi = hHDiff_cum_koi / sum(hHDiff_cum_koi) / HEIGHTDIFF_BIN_SIZE
    with open('data/pdfs/pdf_HDiff_koi.dat', 'wb') as outfile:
        np.save(outfile, pdf_HDiff_koi)
    pdf_avgH_koi = havgH_cum_koi / sum(hshortH_cum_koi) / HEIGHT_BIN_SIZE
    with open('data/pdfs/pdf_avgH_koi.dat', 'wb') as outfile:
        np.save(outfile, pdf_avgH_koi)
    pdf_avgH_koi = havgH_cum_koi / sum(htallH_cum_koi) / HEIGHT_BIN_SIZE
    with open('data/pdfs/pdf_avgH_koi.dat', 'wb') as outfile:
        np.save(outfile, pdf_avgH_koi)

    