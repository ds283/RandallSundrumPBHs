import numpy as np
import pandas as pd

import ray
ray.init()

import LifetimeKit as lkit

import itertools


def compute_lifetime(data):
    serial = data['serial']
    M5_serial = data['M5_serial']
    T_serial = data['T_serial']

    M5 = data['M5']
    Tinit = data['Tinit']
    F = data['F']
    f = data['f']

    params = lkit.ModelParameters(M5)
    engine = lkit.CosmologyEngine(params)

    solution = lkit.PBHInstance(engine, Tinit, accretion_efficiency_F=F, collapse_fraction_f=f)

    SB5D = solution.lifetimes['StefanBoltzmann5D']
    SB4D = solution.lifetimes['StefanBoltzmann4D']

    return {'serial': serial, 'M5_serial': M5_serial, 'T_serial': T_serial,
            'Minit': solution.M_init, 'Tinit': Tinit, 'F': F, 'f': f, 'M5': M5,
            'SB_5D_lifetime': SB5D.T_lifetime, 'SB_5D_shift': SB5D.T_shift, 'SB_5D_compute': SB5D.compute_time,
            'SB_4D_lifetime': SB4D.T_lifetime, 'SB_4D_shift': SB4D.T_shift, 'SB_4D_compute': SB4D.compute_time}

@ray.remote
def map(f, obj):
    return f(obj)

# build soln_grid of M5/Tinit sample points
M5_grid = np.geomspace(2E8, 5E17, 250)
Tinit_grid = np.geomspace(4E8, 5E17, 250)

# generate serial numbers for M5 & Tinit sample grids and write these out
M5_grid_size = len(M5_grid)
grid_M5_serials = np.zeros(M5_grid_size)
grid_M5_values = np.zeros(M5_grid_size)
for serial, M5 in enumerate(M5_grid):
    grid_M5_serials[serial] = serial
    grid_M5_values[serial] = M5

m5_df = pd.DataFrame(data={'serial': grid_M5_serials, 'M5_GeV': grid_M5_values})
m5_df.set_index('serial', inplace=True)
m5_df.to_csv('M5_grid.csv')

Tinit_grid_size = len(Tinit_grid)
grid_Tinit_serials = np.zeros(Tinit_grid_size)
grid_Tinit_GeV = np.zeros(Tinit_grid_size)
grid_Tinit_Kelvin = np.zeros(Tinit_grid_size)
for serial, Tinit in enumerate(Tinit_grid):
    grid_Tinit_serials[serial] = serial
    grid_Tinit_GeV[serial] = Tinit
    grid_Tinit_Kelvin[serial] = Tinit/lkit.Kelvin

Tinit_df = pd.DataFrame(data={'serial': grid_Tinit_serials, 'Tinit_GeV': grid_Tinit_GeV, 'Tinit_Kelvin': grid_Tinit_Kelvin})
Tinit_df.set_index('serial', inplace=True)
Tinit_df.to_csv('Tinit_grid.csv')

# now combine M5 & Tinit grids into a single large grid
data_all = itertools.product(enumerate(M5_grid), enumerate(Tinit_grid))

# data_grid now includes all combinations, even where Tinit is larger than M5 (which should not be allowed),
# or the PBH mass that forms would already be a relic
# so, we need to filter these out
def is_valid(M5: float, Tinit: float, f: float):
    if Tinit > M5:
        return False

    params = lkit.ModelParameters(M5)
    engine = lkit.CosmologyEngine(params)

    try:
        # get mass of Hubble volume expressed in GeV
        M_Hubble = engine.M_Hubble(T=Tinit)

        # compute initial mass in GeV
        M_init = f * M_Hubble

        M_PBH = lkit.PBHModel(params, M_init, units='GeV')
    except RuntimeError as e:
        return False

    return True

data = [(M5_serial, T_serial, M5, Tinit) for ((M5_serial, M5), (T_serial, Tinit)) in data_all if is_valid(M5, Tinit, 0.5)]

# assign a serial number to each configuration
data_grid = [{'serial': serial, 'M5_serial': M5_serial, 'T_serial': T_serial, 'M5': M5, 'Tinit': Tinit,
              'F': 0.1, 'f': 0.5} for serial, (M5_serial, T_serial, M5, Tinit) in enumerate(data)]

# use ray to perform a distributed map of compute_lifetime onto data_grid
soln_grid = ray.get([map.remote(compute_lifetime, line) for line in data_grid])

# build a data frame indexed by serial numbers
# we'll use this to store the computed lifetimes and associated metadata
work_size = len(data_grid)
df_index = range(0, work_size)

# set up numpy arrays to hold outputs
M5_serial = np.zeros(work_size)
T_serial = np.zeros(work_size)

M5 = np.zeros(work_size)
F = np.zeros(work_size)
f = np.zeros(work_size)

M_init_GeV = np.zeros(work_size)
M_init_gram = np.zeros(work_size)
T_init_GeV = np.zeros(work_size)
T_init_Kelvin = np.zeros(work_size)

SB_5D_lifetime_GeV = np.zeros(work_size)
SB_5D_lifetime_Kelvin = np.zeros(work_size)
SB_5D_shift = np.zeros(work_size)
SB_5D_compute = np.zeros(work_size)

SB_4D_lifetime_GeV = np.zeros(work_size)
SB_4D_lifetime_Kelvin = np.zeros(work_size)
SB_4D_shift = np.zeros(work_size)
SB_4D_compute = np.zeros(work_size)

for line in soln_grid:
    serial = line['serial']
    M5_serial[serial] = line['M5_serial']
    T_serial[serial] = line['T_serial']

    M5[serial] = line['M5']
    F[serial] = line['F']
    f[serial] = line['f']

    M_init_GeV[serial] = line['Minit']
    M_init_gram[serial] = line['Minit'] / lkit.Gram
    T_init_GeV[serial] = line['Tinit']
    T_init_Kelvin[serial] = line['Tinit'] / lkit.Kelvin

    SB_5D_lifetime_GeV[serial] = line['SB_5D_lifetime']
    SB_5D_lifetime_Kelvin[serial] = line['SB_5D_lifetime'] / lkit.Kelvin if line['SB_5D_lifetime'] is not None else None
    SB_5D_shift[serial] = line['SB_5D_shift'] / lkit.Kelvin if line['SB_5D_shift'] is not None else None
    SB_5D_compute[serial] = line['SB_5D_compute']

    SB_4D_lifetime_GeV[serial] = line['SB_4D_lifetime']
    SB_4D_lifetime_Kelvin[serial] = line['SB_4D_lifetime'] / lkit.Kelvin if line['SB_4D_lifetime'] is not None else None
    SB_4D_shift[serial] = line['SB_4D_shift'] / lkit.Kelvin if line['SB_4D_shift'] is not None else None
    SB_4D_compute[serial] = line['SB_4D_compute']


df = pd.DataFrame(data={'M5_serial': M5_serial,
                        'T_serial': T_serial,
                        'M5_GeV': M5,
                        'accretion_F': F,
                        'collapse_f': f,
                        'M_init_GeV': M_init_GeV,
                        'M_init_gram': M_init_gram,
                        'T_init_GeV': T_init_GeV,
                        'T_init_Kelvin': T_init_Kelvin,
                        'SB_5D_lifetime_GeV': SB_5D_lifetime_GeV,
                        'SB_5D_lifetime_Kelvin': SB_5D_lifetime_Kelvin,
                        'SB_5D_shift_Kelvin': SB_5D_shift,
                        'SB_5D_compute': SB_5D_compute,
                        'SB_4D_lifetime_GeV': SB_4D_lifetime_GeV,
                        'SB_4D_lifetime_Kelvin': SB_4D_lifetime_Kelvin,
                        'SB_4D_shift_Kelvin': SB_4D_shift,
                        'SB_4D_compute': SB_4D_compute
                        }, index=df_index)
df.index.name = 'index'
df.to_csv('mass_lifetime.csv')
