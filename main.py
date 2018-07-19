from model.bayesian_model import *
import numpy as np
import time

if __name__ == "__main__":

    start_time = time.time()
    
    classes = ['koibito', 'doryo']
    observables = ['v_g', 'd', 'h_diff', 'v_diff']
    bayesian_estimator = BayesianEstimator(cl=classes, obs=observables)

    datasets = get_datasets('data/classes/', classes)
    train_ratio = 0.3
    alphas = [0, 0.5, 1]
    epoch = 20

    #bayesian_estimator.cross_validate(alphas, epoch, train_ratio, datasets)
    # measures = ['KLdiv', 'JSdiv', 'EMD', 'LL']
    # jaccard_dist = bayesian_estimator.cross_validate_global(epoch, train_ratio, datasets, measures)
    bayesian_estimator.cross_validate([0, 0.5, 1], epoch, train_ratio, datasets)
    # print(jaccard_dist)
    
    elapsed_time = time.time() - start_time
    print('Time elapsed  %2.2f sec' %elapsed_time)


    