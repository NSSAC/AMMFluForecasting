import pandas as pd
import itertools
import numpy as np
from scipy.stats import norm
import scipy.stats
from joblib import Parallel, delayed
import multiprocessing
import subprocess
import random
import os
from sklearn.utils.extmath import cartesian
import sys
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--season')
parser.add_argument('-wd', '--work_dir')
parser.add_argument('-l', '--label')

args = parser.parse_args()

datadir = 'data/surveillance'
county_fips = pd.DataFrame({'FIPS':[36005,36047,36061,36081,36085],
                            'Borough':['Bronx','Brooklyn','Manhattan','Queens','Staten Island'],
                        'color':['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']})


def read_ground_truth(season):
    ground_truth = pd.read_csv(datadir+'nyc_ED_ILIplus_{}.csv'.format(season),index_col=0)
    ground_truth.columns = range(len(ground_truth.columns))

    ED_frac = pd.read_csv(datadir+'nyc_ED_frac_{}.csv'.format(season),index_col=0)
    adj_ground_truth = ground_truth.divide(ED_frac.ED_frac.values,axis=0)

    return ED_frac, adj_ground_truth

def get_epicurve(filename, end_week=30, d=7):
    sim_epicurve = pd.read_csv(filename,index_col=0,header=None,delimiter=' ')
    sim_epicurve = sim_epicurve.loc[[36005, 36047, 36061, 36081, 36085],:]
    sim_epicurve = sim_epicurve.iloc[:,range((end_week-1)*7)]
    sim_epicurve = sim_epicurve.groupby(sim_epicurve.columns // d, axis=1).sum()

    return sim_epicurve

def get_ids(score_file):
    cells = pd.read_csv(score_file)
    ids = np.random.choice(cells.id, 1000, True, cells.weight)

    return ids

def get_best_training(work_dir, b_ids, adj_ground_truth, data_horizon):
    s = {}
    for b_id in b_ids:
        sim_epicurve = get_epicurve('{}outs/cell_{}.out'.format(work_dir, b_id))
#         s[b_id] = abs(sim_epicurve.iloc[:,range(data_horizon)] - adj_ground_truth.iloc[:,range(data_horizon)]).sum().sum()
        s[b_id] = np.mean(np.mean(abs(sim_epicurve.iloc[:,range(data_horizon)] - adj_ground_truth.iloc[:,range(data_horizon)])/adj_ground_truth.iloc[:,range(data_horizon)]))

    return s

def get_pred_score(work_dir, b_id, adj_ground_truth, data_horizon, look_ahead=4):
    sim_epicurve = get_epicurve('{}outs/cell_{}.out'.format(work_dir, b_id))
    a = np.mean(np.mean(abs(sim_epicurve.iloc[:,range(data_horizon+1, data_horizon+look_ahead)] - adj_ground_truth.iloc[:,range(data_horizon+1, data_horizon+look_ahead)])/adj_ground_truth.iloc[:,range(data_horizon+1, data_horizon+look_ahead)]))

    return a

def get_peak(work_dir, b_ids):
    ptime = {}
    psize = {}
    k = 0
    for b_id in b_ids:
        sim_epicurve = get_epicurve('{}outs/cell_{}.out'.format(work_dir, b_id))
        ptime[k] = sim_epicurve.idxmax(1)
        psize[k] = sim_epicurve.max(1)
        k = k + 1

    return pd.DataFrame(ptime), pd.DataFrame(psize)

def get_onset(work_dir, b_ids, adj_ground_truth, onset_th=0.1):
    on = []
    s = adj_ground_truth.max(1) * onset_th
    for b_id in b_ids:
        on_c = []
        for f in [36005,36047,36061,36081,36085]:
            sim_epicurve = get_epicurve('{}outs/cell_{}.out'.format(work_dir, b_id))
            ss = sim_epicurve.loc[f,:] > s[f]
            on_c.append(ss.idxmax())
        on.append(on_c)

    on_df = pd.DataFrame(on)
    on_df.columns = [36005,36047,36061,36081,36085]

    return on_df

season = int(args.season)
work_dir = args.work_dir
label = args.label

ED_frac, adj_ground_truth = read_ground_truth(season)
peak_time = {}
peak_size = {}
onset = {}
for data_horizon in [10,12,14,16,18,20,22,24,26]:
	ids = get_ids('{}cells_{}_7_0.0_log.csv'.format(work_dir, data_horizon))
	ptime_df, psize_df = get_peak(work_dir, ids)
	on_df = get_onset(work_dir, ids, adj_ground_truth, onset_th=0.1)

	peak_time[data_horizon] = ptime_df.transpose()
	peak_size[data_horizon] = psize_df.transpose()
	onset[data_horizon] = on_df


d = {}
d['peakT'] = peak_time
d['peakS'] = peak_size
d['onsetT'] = onset

np.save('{}_{}_seasonal_target.npy'.format(season,label), d)
