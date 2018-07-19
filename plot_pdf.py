from model.bayesian_model import *
import numpy as np

if __name__ == "__main__":

    classes = classes = ['doryo', 'koibito', 'kazoku', 'yujin']
    observables = ['v_g', 'v_diff', 'h_diff', 'd']
    bayesian_estimator = BayesianEstimator(cl=classes, obs=observables)

    datasets = get_datasets('data/classes/', classes)
    train_ratio = 1
    train_sets, tests_sets = shuffle_data_set(datasets, train_ratio)

    bayesian_estimator.train(train_sets=train_sets)
    bayesian_estimator.plot_pdf()
    bayesian_estimator.save_pdfs()
    