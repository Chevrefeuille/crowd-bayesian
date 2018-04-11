import numpy as np
from model import tools
import matplotlib.pyplot as plt
from os import listdir
import random
import operator
from model.constants import TRADUCTION_TABLE, PARAM_NAME_TABLE, PARAM_UNIT_TABLE, PLOT_PARAM_TABLE

def get_datasets(data_path, classes):
    """
    Get the dataset for the given classes
    """
    datasets = {}
    for c in classes:
        class_path = data_path + c +'/'
        class_set = [class_path + f for f in listdir(class_path) if 'threshold' in f]
        datasets[c] = class_set
    return datasets

def shuffle_data_set(datasets, train_ratio):
    """
    Randomly partition the datasets into training and testing sets
    """
    train_sets, test_sets = {}, {}
    for c, dataset in datasets.items():
        n = len(dataset)
        n_train = round(train_ratio * n)
        shuffled_set = random.sample(dataset, n)
        train_sets[c] = shuffled_set[:n_train]
        test_sets[c] = shuffled_set[n_train:]
    
    return train_sets, test_sets

class BayesianEstimator():
    """
    Base class for the estimator
    """
    def __init__(self, cl, obs):
        self.cl = cl
        self.obs = obs
        self.pdfs = {}

    def train(self, train_sets):
        histograms = {}
        # initialize empty histograms
        for o in self.obs:
            histograms[o], self.pdfs[o] = {}, {}
            for c in self.cl:
                histograms[o][c] = tools.initialize_histogram(o)
        # compute histograms for each classes
        for c in self.cl:
            for file_path in train_sets[c]:
                data = np.load(file_path)
                data_A, data_B = tools.extract_individual_data(data)
                obs_data = tools.compute_observables(data_A, data_B)
                for o in self.obs:
                    histograms[o][c] += tools.compute_histogram(o, obs_data[o])
        for o in self.obs:
            for c in self.cl:
                self.pdfs[o][c] = tools.compute_pdf(o,  histograms[o][c])
    
    def compute_probabilities(self, bins, alpha):
        n_data = len(bins[self.obs[0]])
        p_posts, p_prior, p_likes, p_conds = {}, {}, {}, {}
        for c in self.cl:
            p_prior[c] = 1 / len(self.cl)
            p_posts[c] = np.zeros((n_data))
            p_posts[c][0] = p_prior[c]
        for j in range(1, n_data):
            for c in self.cl:
                p_likes[c] = 1
                i = 0
                for o in self.obs:
                    if self.pdfs[o][c][bins[o][j]-1] != 0:
                        p_likes[c] *= self.pdfs[o][c][bins[o][j]-1]
                    # else:
                        # i += 1
                p_prior[c] = alpha * p_posts[c][0] + (1 - alpha) * p_posts[c][j-1]
                p_conds[c] = p_likes[c] * p_prior[c]          
            s = sum(p_conds.values())
            for c in self.cl:
                p_posts[c][j] = p_conds[c] / s 
        mean_ps = {}
        for c in self.cl:
            mean_ps[c] = np.mean(p_posts[c])
        return mean_ps

    def evaluate(self, alpha, test_sets):
        results = {}
        confusion_matrix = {}
        # print('-------------------------------')
        # print('\t Right \t Wrong \t Rate\n')
        t = 0
        for c in self.cl:
            results[c] = {'right': 0, 'wrong': 0}
            # init condusion matrix
            confusion_matrix[c] = {}
            for c_pred in self.cl:
                confusion_matrix[c][c_pred] = 0
            for file_path in test_sets[c]:
                data = np.load(file_path)
                data_A, data_B = tools.extract_individual_data(data)
                obs_data = tools.compute_observables(data_A, data_B)
                # obs_data = tools.shuffle_data(obs_data)
                bins = {}
                for o in self.obs:
                    bins[o] = tools.find_bins(o, obs_data[o])
                mean_p = self.compute_probabilities(bins, alpha)
                # t += i
                class_max = max(mean_p.items(), key=operator.itemgetter(1))[0]
                confusion_matrix[c][class_max] += 1
                if class_max == c:
                    results[c]['right'] += 1
                else:
                    results[c]['wrong'] += 1
                rate = results[c]['right'] / (results[c]['right'] + results[c]['wrong'])
            # print('{}\t {}\t {}\t {}'.format(c, results[c]['right'], results[c]['wrong'], rate))
        # tools.print_confusion_matrix(self.cl, confusion_matrix)
        # print(t)
        return results

    
    def cross_validate(self, alphas, epoch, train_ratio, datasets):
        for a in alphas:
            right_ns = {}
            for c in self.cl:
                right_ns[c] = []
            for epoch in range(20):
                train_sets, tests_sets = shuffle_data_set(datasets, train_ratio)
                self.train(train_sets=train_sets)
                results = self.evaluate(alpha=a, test_sets=tests_sets)
                for c in self.cl:
                    right_ns[c] += [results[c]['right']]
            print('-------------------------------')
            print('alpha = {}'.format(a))
            for c in self.cl:
                mean_succ = np.mean(right_ns[c]) / len(tests_sets[c])
                sdt_succ = np.std(right_ns[c]) / len(tests_sets[c])
                print('{}\t {:.2f}% ± {:.2f}%'.format(c, mean_succ * 100, sdt_succ * 100))

    def plot_pdf(self):
        plt.rcParams['grid.linestyle'] = '--'
        for o in self.obs:
            edges = tools.get_edges(o)
            for c in self.cl:
                plt.plot(edges, self.pdfs[o][c], label=TRADUCTION_TABLE[c], linewidth=3)
            plt.xlabel('{}({})'.format(PARAM_NAME_TABLE[o], PARAM_UNIT_TABLE[o]))
            plt.ylabel('p({})'.format(PARAM_NAME_TABLE[o]))
            plt.xlim(PLOT_PARAM_TABLE[o])
            plt.legend()
            plt.grid()
            plt.show()



            

               
