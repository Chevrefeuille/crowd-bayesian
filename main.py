from model.bayesian_model import *
import numpy as np

if __name__ == "__main__":

    classes = ['koibito', 'doryo', 'kazoku', 'yujin']
    observables = ['v_g', 'd', 'h_diff', 'v_diff']
    bayesian_estimator = BayesianEstimator(cl=classes, obs=observables)

    datasets = get_datasets('data/classes/', classes)
    train_ratio = 0.3
    alphas = [0, 0.5, 1]
    epoch = 50

    bayesian_estimator.cross_validate(alphas, epoch, train_ratio, datasets)
    