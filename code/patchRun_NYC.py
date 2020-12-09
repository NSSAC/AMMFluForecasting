import pandas as pd
import itertools
import numpy as np
import multiprocessing
import subprocess
import random
import os
from sklearn.utils.extmath import cartesian
import sys
from joblib import Parallel, delayed
from datetime import datetime
import argparse

sys.path.append('/projects/PatchSim/') ## path to clone of NSSAC/PatchSim
import patchsim as sim

parser = argparse.ArgumentParser()
parser.add_argument('-ddir', '--work_dir')
parser.add_argument('-s', '--season')
args = parser.parse_args()
work_dir = args.work_dir
season=args.season ##example 2016-17

datadir = 'data/surveillance/'
county_fips = pd.DataFrame({'FIPS':[36005,36047,36061,36081,36085],
                            'Borough':['Bronx','Brooklyn','Manhattan','Queens','Staten Island'],
                        'color':['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00']})

##ED_ILIplus is weekly
ground_truth = pd.read_csv(datadir+'nyc_ED_ILIplus_{}.csv'.format(season),index_col=0)
ground_truth.columns = range(len(ground_truth.columns))

ED_frac = pd.read_csv(datadir+'nyc_ED_frac_{}.csv'.format(season),index_col=0)
adj_ground_truth = ground_truth.divide(ED_frac.ED_frac.values,axis=0)

### 125000 particles in full factorial design
Betas = np.linspace(0.5,0.9,10)
Alphas = np.linspace(0.4,0.8,10)
Gammas = np.linspace(0.4,0.8,10)
Scaling = np.linspace(0.0005,0.0025,5)
SeedThreshold = np.linspace(10,30,5)
SeedPatchCount = np.linspace(1,5,5)

cells = pd.DataFrame(cartesian([Betas,Alphas,Gammas,Scaling,SeedThreshold,SeedPatchCount]),
                     columns=['beta','alpha','gamma','scaling','seedT','seedN'])
cells.index.name='id'

if 'seeds' not in os.listdir(work_dir):
    os.mkdir(work_dir+'seeds')

def gen_seed(i, work_dir, cells):
    scaling = cells['scaling'][i]
    seedT = cells['seedT'][i]
    seedN = cells['seedN'][i]

    seed_times = ground_truth.applymap(lambda x: x>seedT).idxmax(axis=1).reset_index()
    seeds = seed_times.sort_values(0).reset_index(drop=True).loc[0:seedN-1]
    seeds['Count'] = seeds.apply(lambda row: ground_truth.loc[row['FIPS'],row[0]],axis=1)
    seeds['Count']/=scaling
    seeds.columns = ['FIPS','Day','Count']
    seeds['Day']*=7
    seeds[['Day','FIPS','Count']].to_csv(work_dir+'seeds/seed_{}.csv'.format(i),
                                         index=None,header=None,sep=' ')
num_cores = 40
results = Parallel(n_jobs=num_cores)(delayed(gen_seed)(i, work_dir, cells) for i in range(cells.shape[0]))

if 'outs' not in os.listdir(work_dir):
    os.mkdir(work_dir+'outs')
def run_patch(cfg):
    sim.run_disease_simulation(sim.read_config(cfg),write_epi=True)
cfgs = [work_dir+'cfgs/{}'.format(x) for x in os.listdir(work_dir+'cfgs/') if 'cfg_' in x]

num_cores = 40
results = Parallel(n_jobs=num_cores)(delayed(run_patch)(cfg) for cfg in cfgs)
