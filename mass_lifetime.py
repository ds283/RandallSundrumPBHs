import numpy as np
import pandas as pd

import ray
ray.init()

import LifetimeKit as lkit

import itertools


def compute_lifetime(data):
    serial = data['serial']
    M5 = data['M5']
    Tinit = data['Tinit']
    F = data['F']
    f = data['f']

    params = lkit.ModelParameters(M5)
    engine = lkit.CosmologyEngine(params)

    solution = lkit.PBHInstance(engine, Tinit, accretion_efficiency_F=F, collapse_fraction_f=f)

    standard = solution.lifetimes['standard']

    return {'serial': serial, 'Minit': solution.M_init, 'Tinit': Tinit, 'F': F, 'f': f, 'M5': M5,
            'SB_lifetime': standard.T_lifetime, 'SB_shift': standard.T_shift, 'SB_compute': standard.compute_time}

@ray.remote
def map(f, obj):
    return f(obj)

# build soln_grid of M5/Tinit sample points
M5_grid = np.geomspace(2E8, 1E17, 200)
Tinit_grid = np.geomspace(4E8, 5E16, 200)

# combine into a soln_grid
data_all = itertools.product(M5_grid, Tinit_grid)

# the soln_grid includes all combinations, even where Tinit is larger than M5 (which should not be allowed)
# so, we need to filter these out
data = [p for p in data_all if p[1] < p[0]]

# assign a serial number to each configuration
data_serials = enumerate(data)
data_grid = [{'serial': serial, 'M5': M5, 'Tinit': Tinit, 'F': 0.1, 'f': 0.5} for serial, (M5, Tinit) in data_serials]

# build a data frame indexed by serial numbers
# we'll use this to store the computed lifetimes and associated metadata
work_size = len(data_grid)
df_index = range(0, work_size)

# set up numpy arrays to hold outputs
M5 = np.zeros(work_size)
F = np.zeros(work_size)
f = np.zeros(work_size)
M_init_GeV = np.zeros(work_size)
M_init_gram = np.zeros(work_size)
T_init_GeV = np.zeros(work_size)
T_init_Kelvin = np.zeros(work_size)
SB_lifetime_GeV = np.zeros(work_size)
SB_lifetime_Kelvin = np.zeros(work_size)
SB_shift = np.zeros(work_size)
SB_compute = np.zeros(work_size)

soln_grid = ray.get([map.remote(compute_lifetime, line) for line in data_grid])
for line in soln_grid:
    serial = line['serial']
    M5[serial] = line['M5']
    F[serial] = line['F']
    f[serial] = line['f']
    M_init_GeV[serial] = line['Minit']
    M_init_gram[serial] = line['Minit'] / lkit.Gram
    T_init_GeV[serial] = line['Tinit']
    T_init_Kelvin[serial] = line['Tinit'] / lkit.Kelvin
    SB_lifetime_GeV[serial] = line['SB_lifetime']
    SB_lifetime_Kelvin[serial] = line['SB_lifetime'] / lkit.Kelvin if line['SB_lifetime'] is not None else None
    SB_shift[serial] = line['SB_shift'] / lkit.Kelvin if line['SB_shift'] is not None else None
    SB_compute[serial] = line['SB_compute']


df = pd.DataFrame(data={'M5_GeV': M5,
                        'accretion_F': F,
                        'collapse_f': f,
                        'M_init_GeV': M_init_GeV,
                        'M_init_gram': M_init_gram,
                        'T_init_GeV': T_init_GeV,
                        'T_init_Kelvin': T_init_Kelvin,
                        'SB_lifetime_GeV': SB_lifetime_GeV,
                        'SB_lifetime_Kelvin': SB_lifetime_Kelvin,
                        'SB_shift_Kelvin': SB_shift,
                        'SB_compute': SB_compute}, index=df_index)
df.to_csv('mass_lifetime.csv')
