import pandas as pd
import itertools
import numpy as np
from scipy.stats import norm
from joblib import Parallel, delayed
import multiprocessing
import subprocess
import random
import os
from sklearn.utils.extmath import cartesian
import sys
from datetime import datetime

import argparse

def get_weekly_epicurve(filename, d):
    sim_epicurve = pd.read_csv(filename,index_col=0,header=None,delimiter=' ')
    return sim_epicurve.groupby(sim_epicurve.columns // d, axis=1).sum()

def get_ids(work_dir, data_horizon, d, nsamp, log, th=None):

    if log:
        if th is None:
            cells = pd.read_csv('{}cells_{}_{}_log.csv'.format(work_dir, data_horizon, d))
        else:
            cells = pd.read_csv('{}cells_{}_{}_{}_log.csv'.format(work_dir, data_horizon, d, th))
    else:
        cells = pd.read_csv('{}cells_{}_{}.csv'.format(work_dir, data_horizon, d))
    ids = np.random.choice(cells.id, nsamp, True, cells.weight)
    if log:
        if th is None:
            np.savetxt('{}id_{}_{}_log.txt'.format(work_dir, data_horizon, d), ids)
        else:
            np.savetxt('{}id_{}_{}_{}_log.txt'.format(work_dir, data_horizon, d, th), ids)
    else:
        np.savetxt('{}id_{}_{}.txt'.format(work_dir, data_horizon, d), ids)
    return ids

def get_mape(work_dir, data_horizon, d, ids, ground_truth, ED_frac, log, th=None, look_ahead=4):

    FIPS=[36005,36047,36061,36081,36085]

    forecast_horizon = data_horizon + look_ahead
    forecast_horizon = 30
    best_ids = np.unique(ids)
    dict_d = {}
    for b_id in best_ids:
       	best_cell = work_dir+'outs/cell_{}.out'.format(b_id)
       	sim_epicurve = get_weekly_epicurve(best_cell, 7)
       	sim_epicurve = sim_epicurve.multiply(ED_frac.ED_frac.values,axis=0)

        dict_d[b_id] = sim_epicurve.iloc[:,range(forecast_horizon)]

    ## compute summaries
    dff = {}
    dfp = pd.DataFrame(columns=[0.05, 0.275, 0.5, 0.725, 0.95], index=FIPS)

    for fips in FIPS:
        dd = []
        dfc = pd.DataFrame(columns=[0.05, 0.275, 0.5, 0.725, 0.95], index=range(dict_d[b_id].shape[1]))
        for k in dict_d.keys():
            dd.append(dict_d[k].loc[fips])
        df = pd.DataFrame(dd, index=dict_d.keys())
        for c in dfc.columns:
            dfc[c] = df.loc[ids,:].quantile(q=c, axis=0)

        dff[fips] = dfc
        dfp.loc[fips,:] = df.loc[ids,:].idxmax(axis=1).quantile([0.05, 0.275, 0.5, 0.725, 0.95])

    pred_df = pd.DataFrame(columns=range(data_horizon, forecast_horizon), index=FIPS)
    for f in FIPS:
        pred_df.loc[f,:] = dff[f].loc[range(data_horizon, forecast_horizon), 0.5]
    diff_df = abs(pred_df - ground_truth.loc[:,range(data_horizon, forecast_horizon)])
    mape_df = diff_df.divide(ground_truth.iloc[:,range(data_horizon, forecast_horizon)])

    if log:
        if th is None:
            np.save('{}forecast_{}_{}_log.npy'.format(work_dir, data_horizon, d), dff)
            np.save('{}peak_forecast_{}_{}_log.npy'.format(work_dir, data_horizon, d), dfp)
        else:
            np.save('{}forecast_{}_{}_{}_log_30.npy'.format(work_dir, data_horizon, d, th), dff)
            np.save('{}peak_forecast_{}_{}_{}_log_30.npy'.format(work_dir, data_horizon, d, th), dfp)
    else:
        np.save('{}forecast_{}_{}.npy'.format(work_dir, data_horizon, d), dff)
        np.save('{}peak_forecast_{}_{}.npy'.format(work_dir, data_horizon, d), dfp)

    return mape_df

def main(log):

	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--season')
	parser.add_argument('-ddir', '--work_dir')
	parser.add_argument('-cumd', '--cum_days')
	parser.add_argument('-t', '--th')

	args = parser.parse_args()
	th=args.th

	##ED_ILIplus is weekly
	datadir = 'data/surveillance'
	county_fips = pd.DataFrame({'FIPS':[36005,36047,36061,36081,36085],
                            'Borough':['Bronx','Brooklyn','Manhattan','Queens','Staten Island'],
                        'color':['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']})
	ground_truth = pd.read_csv(datadir+'nyc_ED_ILIplus_{}.csv'.format(args.season),index_col=0)
	ground_truth.columns = range(len(ground_truth.columns))

	ED_frac = pd.read_csv(datadir+'nyc_ED_frac_{}.csv'.format(args.season),index_col=0)
	adj_ground_truth = ground_truth.divide(ED_frac.ED_frac.values,axis=0)


	mape_dff = pd.DataFrame(columns=range(len(np.arange(10, 27, 2))), index=county_fips.FIPS)
	k = 0
	for data_horizon in np.arange(10, 27, 2):
		ids = get_ids(args.work_dir, data_horizon, args.cum_days, 100000, log=True, th=th)
		mape_df = get_mape(args.work_dir, data_horizon, args.cum_days, ids, ground_truth, ED_frac, log=True, th=th)
		mape_dff.iloc[:,k] = mape_df.mean(1).tolist()
		k = k + 1
	write_mape=True
	if write_mape:
		if log:
		    if th is None:
                        mape_dff.to_csv(args.work_dir+'mape_all_'+ str(args.cum_days)+'_log.csv')
		    else:
                        mape_dff.to_csv(args.work_dir+'mape_all_'+ str(args.cum_days)+'_'+str(th)+'_log.csv')
		else:
		    mape_dff.to_csv(args.work_dir+'mape_all_'+ str(args.cum_days)+'.csv')

main(log=True)
