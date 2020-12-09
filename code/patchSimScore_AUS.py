import pandas as pd
import itertools
import numpy as np
from scipy.stats import norm
from scipy.stats import truncnorm
from joblib import Parallel, delayed
import multiprocessing
import subprocess
import random
import os
from sklearn.utils.extmath import cartesian
import sys
from datetime import datetime
import argparse

def get_epicurve(filename):
    sim_epicurve = pd.read_csv(filename,index_col=0,header=None,delimiter=' ')
    return sim_epicurve

def get_county_score(filename, i, scaling_df, data_horizon, adj_ground_truth, d, c, log=False):
    w = int(d/7)
    a = adj_ground_truth.sum()

    scaling_factor = scaling_df[c][i]
    sim_epicurve = get_epicurve(filename)
    sim_epicurve = sim_epicurve.loc[['NSW', 'Qld', 'SA', 'Tas', 'Vic', 'WA', 'NT'],:]
    sim_epicurve = sim_epicurve.loc[[c],:]
    sim_epicurve = sim_epicurve.iloc[:,range((data_horizon-1)*7)]
    sim_epicurve = sim_epicurve.groupby(sim_epicurve.columns // d, axis=1).sum()
    sim_epicurve = sim_epicurve * scaling_factor
    sim_epicurve_vec = sim_epicurve.to_numpy().reshape(1*sim_epicurve.shape[1])
    if log:
        sim_epicurve_vec = np.log(sim_epicurve_vec + 1)

    m = max(adj_ground_truth.loc[c, range(data_horizon)])
    beta = []
    for i in range(data_horizon):
        val = adj_ground_truth.loc[c, i]
        for j in np.linspace(0, 0.9, num=10):
            if (val >= (j * m)) & (val <= ((j+0.1) * m)):
                beta.append(j+0.1)
    beta = np.array(beta)
    beta = beta[range(data_horizon)]

    if m < 1:
        beta = 1
    adj_ground_truth_vec = adj_ground_truth.loc[c,range(data_horizon)]

    if log:
        adj_ground_truth_vec = np.log(adj_ground_truth_vec + 1)

    #define sd
    alpha = (data_horizon - np.array(range(data_horizon))) / float(data_horizon)

    gt_th = []
    for i in range(data_horizon):
        if 0.008*adj_ground_truth_vec[i] < 2:
            gt_th.append(2)
        else:
            gt_th.append(0.008*adj_ground_truth_vec[i])

    gt_th = np.array(gt_th)
    sd = gt_th * (0.9 ** beta)/(0.9**alpha)

    adj_ground_truth_vec = list(adj_ground_truth_vec)
    p = np.log(norm.pdf(sim_epicurve_vec, adj_ground_truth_vec, sd))
    p = np.array(p)

    return np.exp(p.sum())

def get_score(work_dir, data_horizon, adj_ground_truth, d, c, log):
    cells=pd.read_csv(work_dir+'cells.csv')
    scaling_df=pd.read_csv(work_dir+'scaling.csv')
    score = np.zeros(len(cells))
    num_cores = 40
    score = Parallel(n_jobs=num_cores)(delayed(get_county_score)(work_dir+'outs/cell_{}.out'.format(cell_id), int(cell_id), scaling_df, data_horizon, adj_ground_truth, d, c, log) for cell_id in cells.index)
    return score/sum(score)


def write_score(work_dir, data_horizon, adj_ground_truth, d, log):
    th=0.0
    states = ['NSW', 'Qld', 'SA', 'Vic', 'WA']
    cells = pd.read_csv(work_dir + 'cells.csv')
    scores = 0
    for state in states:
	    scores = scores + np.log(get_score(work_dir, data_horizon, adj_ground_truth, d, state, True))

    scores = np.exp(scores)
    scores = scores/sum(scores)

    cells['weight'] = scores
    if log:
        cells.to_csv(work_dir+'cells_{}_{}_{}_log.csv'.format(str(data_horizon), d, str(th)))
    else:
        cells.to_csv(work_dir+'cells_{}_{}_log.csv'.format(str(data_horizon), d))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--season')
	parser.add_argument('-ddir', '--work_dir')
	parser.add_argument('-d', '--data_horizon')
	parser.add_argument('-cumd', '--cum_days')

	args = parser.parse_args()

	datadir = 'data/surveillance'
	ground_truth = pd.read_csv(datadir+'aus_flupositive_2016.csv')
	ground_truth = ground_truth.pivot(index='State', columns='Date', values='Count')
	ground_truth.columns = range(len(ground_truth.columns))

	cells = pd.read_csv(args.work_dir + 'cells.csv', index_col=0)
	write_score(args.work_dir, int(args.data_horizon), ground_truth, int(args.cum_days), log=True)


if __name__ == '__main__':
    main()
