import pandas as pd
import itertools
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
import subprocess
import random
import os
from sklearn.utils.extmath import cartesian
import sys

from datetime import datetime

sys.path.append('/projects/PatchSim/') ## path to clone of NSSAC/PatchSim
import patchsim as sim

datadir = 'data/surveillance'
state_id = pd.DataFrame({'ID':['NSW', 'Qld', 'SA', 'Tas', 'Vic', 'WA', 'ACT', 'NT'],
                            'State':['New South Wales','Queensland','South Australia','Tasmania','Victoria', 'Western Australia', 'Australian Capital Territory', 'Northern Territory'],
                        })
season = 2016
suffix = '_RAD' ## mobility model
config_template = 'cfg_AUS'+suffix
work_dir = str(datetime.now().month)+str(datetime.now().day)+'_AUS'+suffix+'_{}/'.format(season)

if work_dir[:-1] in os.listdir('.'):
    command = "rm -R {}".format(work_dir)
    subprocess.call(command, shell=True)

command = "mkdir {}".format(work_dir)
subprocess.call(command, shell=True)

### uniform sampling of 100000 particles
N = 100000
Betas = np.random.uniform(0.3,0.9,size=N)
Alphas = np.random.uniform(0.3,0.9,size=N)
Gammas = np.random.uniform(0.3,0.9,size=N)
SeedThreshold = np.random.choice(range(10,100), size=N, replace=True)
SeedPatchCount = np.random.choice(range(6,8), size=N, replace=True)

scaling_df = pd.DataFrame(np.random.uniform(0.01, 0.02, size = (N,7)))
scaling_df.columns = ['NSW', 'Qld', 'SA', 'Tas', 'Vic', 'WA', 'NT']

cells = pd.DataFrame(np.array([Betas, Alphas, Gammas, SeedThreshold, SeedPatchCount]))
cells = pd.concat([cells.transpose(), scaling_df], axis=1, ignore_index=True)
cells.columns = ['beta','alpha','gamma','seedT','seedN','scaling1','scaling2','scaling3','scaling4','scaling5','scaling6','scaling7']

cells.index.name='id'
cells.to_csv(work_dir+'cells.csv')
scaling_df.index.name='id'
scaling_df.to_csv(work_dir+'scaling.csv')

if 'cfgs' not in os.listdir(work_dir):
    os.mkdir(work_dir+'cfgs')

if 'logs' not in os.listdir(work_dir):
    os.mkdir(work_dir+'logs')

with open(config_template,'r') as f:
    cfg_template=f.read()
cfg_workdir = cfg_template.replace('$work_dir',str(work_dir))

for index,row in cells.iterrows():
    cfg_copy = cfg_workdir
    cfg_copy = cfg_copy.replace('$id',str(index))
    for param in cells.columns:
        cfg_copy = cfg_copy.replace('${}'.format(param),str(row['{}'.format(param)]))
    with open(work_dir+'cfgs/cfg_{}'.format(index),'w') as f:
        f.write(cfg_copy)

## ground truth
ground_truth = pd.read_csv(datadir+'aus_flupositive_2016.csv')
ground_truth = ground_truth.pivot(index='State', columns='Date', values='Count')
ground_truth.columns = range(len(ground_truth.columns))

if 'seeds' not in os.listdir(work_dir):
    os.mkdir(work_dir+'seeds')

def gen_seed(i, ground_truth, cells, scaling_df):
    seedT=int(cells['seedT'][i])
    seedN=int(cells['seedN'][i])
    scaling=scaling_df.loc[i,:]
    seed_times = ground_truth.applymap(lambda x: x>seedT).idxmax(axis=1).reset_index()
    seeds = seed_times.sort_values(0).reset_index(drop=True).loc[0:seedN-1]
    seeds['Count'] = seeds.apply(lambda row: ground_truth.loc[row['State'],row[0]],axis=1)
    seeds['Count'] = np.divide(np.array(seeds['Count']), np.array(scaling[seeds['State']]))

    seeds.columns = ['State','Day','Count']
    seeds['Day']*=7
    seeds[['Day','State','Count']].to_csv(work_dir+'seeds/seed_{}.csv'.format(i),
                                         index=None,header=None,sep=' ')
num_cores = 40
results = Parallel(n_jobs=num_cores)(delayed(gen_seed)(i,ground_truth,cells,scaling_df) for i in range(N))

if 'outs' not in os.listdir(work_dir):
    os.mkdir(work_dir+'outs')
def run_patch(cfg):
    sim.run_disease_simulation(sim.read_config(cfg),write_epi=True)
cfgs = [work_dir+'cfgs/{}'.format(x) for x in os.listdir(work_dir+'cfgs/') if 'cfg_' in x]
num_cores = 40
results = Parallel(n_jobs=num_cores)(delayed(run_patch)(cfg) for cfg in cfgs)
