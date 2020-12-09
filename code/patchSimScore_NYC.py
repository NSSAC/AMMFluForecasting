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

def get_county_score(filename, data_horizon, adj_ground_truth, d, c, log=False):
    w = int(d/7)
    a = adj_ground_truth.sum()

    sim_epicurve = get_epicurve(filename)
    sim_epicurve = sim_epicurve.loc[[36005, 36047, 36061, 36081, 36085],:] ##NYC County FIPS codes
    sim_epicurve = sim_epicurve.loc[[c],:]
    sim_epicurve = sim_epicurve.iloc[:,range((data_horizon-1)*7)]
    sim_epicurve = sim_epicurve.groupby(sim_epicurve.columns // d, axis=1).sum() ## Convert to weekly
    sim_epicurve_vec = sim_epicurve.as_matrix().reshape(1*sim_epicurve.shape[1])
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
    if m < 1:
        beta = 1

    adj_ground_truth_vec = adj_ground_truth.loc[c,range(data_horizon)]
    if log:
        adj_ground_truth_vec = np.log(adj_ground_truth_vec + 1)
    #define sd
    alpha = (data_horizon - np.array(range(data_horizon))) / float(data_horizon)
    sd = 1 * (0.9 ** beta)/(0.9**alpha)

    adj_ground_truth_vec = list(adj_ground_truth_vec)
    p = np.log(norm.pdf(sim_epicurve_vec, adj_ground_truth_vec, sd))
    p = np.array(p)

    return np.exp(p.sum())

def get_score(work_dir, data_horizon, adj_ground_truth, d, c, s, log):
    cells=pd.read_csv(work_dir+'cells.csv')
    score = np.zeros(len(cells))

    num_cores = multiprocessing.cpu_count()
    score = Parallel(n_jobs=num_cores)(delayed(get_county_score)(work_dir+'outs/cell_{}.out'.format(cell_id), data_horizon, adj_ground_truth, d, c, log) for cell_id in cells.index)

    return score/sum(score)


def write_score(work_dir, data_horizon, adj_ground_truth, d, log):
    th=0.0
    counties = [36005,36047,36061,36081,36085]
    cells = pd.read_csv(work_dir + 'cells.csv')
    scores = 0
    for county in counties:
	    scores = scores + np.log(get_score(work_dir, data_horizon, adj_ground_truth, d, county, True))
    scores = np.exp(scores)
    scores = scores/sum(scores)
    cells['weight'] = scores/sum(scores)
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
	county_fips = pd.DataFrame({'FIPS':[36005,36047,36061,36081,36085],
								'Borough':['Bronx','Brooklyn','Manhattan','Queens','Staten Island'],
                       'color':['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']})
	##ED_ILIplus is weekly
	ground_truth = pd.read_csv(datadir+'nyc_ED_ILIplus_{}.csv'.format(args.season),index_col=0)
	ground_truth.columns = range(len(ground_truth.columns))

	ED_frac = pd.read_csv(datadir+'nyc_ED_frac_{}.csv'.format(args.season),index_col=0)
	adj_ground_truth = ground_truth.divide(ED_frac.ED_frac.values,axis=0)

	write_score(args.work_dir, int(args.data_horizon), adj_ground_truth, int(args.cum_days), log=True)

if __name__ == "__main__":
    main()
