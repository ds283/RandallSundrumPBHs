import numpy as np
import pandas as pd

import ray
from ray.actor import ActorHandle
from progressbar import ProgressBar
ray.init()

import LifetimeKit as lkit

import itertools


# adapted from https://docs.ray.io/en/latest/ray-core/examples/progress_bar.html


def compute_lifetime(data, pba: ActorHandle):
    serial = data['serial']
    M5_serial = data['M5_serial']
    T_serial = data['T_serial']

    M5 = data['M5']
    Tinit = data['Tinit']
    F = data['F']
    f = data['f']

    params = lkit.RS5D.Parameters(M5)

    model_list = ['GreybodyRS5D', 'GreybodyStandard4D',
                  'StefanBoltzmannRS5D', 'StefanBoltzmannStandard4D',
                  'StefanBoltzmannRS5D-noreff', 'StefanBoltzmannStandard4D-noreff',
                  'StefanBoltzmannRS5D-fixedg', 'StefanBoltzmannStandard4D-fixedg',
                  'StefanBoltzmannRS5D-fixedN', 'StefanBoltzmannStandard4D-fixedN',
                  'StefanBoltzmannRS5D-noPage', 'StefanBoltzmannStandard4D-noPage']
    solution = lkit.PBHInstance(params, Tinit, models=model_list, accretion_efficiency_F=F, collapse_fraction_f=f)

    GB5D = solution.lifetimes['GreybodyRS5D']
    GB4D = solution.lifetimes['GreybodyStandard4D']

    SB5D = solution.lifetimes['StefanBoltzmannRS5D']
    SB4D = solution.lifetimes['StefanBoltzmannStandard4D']

    SB5D_noreff = solution.lifetimes['StefanBoltzmannRS5D-noreff']
    SB4D_noreff = solution.lifetimes['StefanBoltzmannStandard4D-noreff']

    SB5D_fixedg = solution.lifetimes['StefanBoltzmannRS5D-fixedg']
    SB4D_fixedg = solution.lifetimes['StefanBoltzmannStandard4D-fixedg']

    SB5D_fixedN = solution.lifetimes['StefanBoltzmannRS5D-fixedN']
    SB4D_fixedN = solution.lifetimes['StefanBoltzmannStandard4D-fixedN']

    SB5D_noPage = solution.lifetimes['StefanBoltzmannRS5D-noPage']
    SB4D_noPage = solution.lifetimes['StefanBoltzmannStandard4D-noPage']

    pba.update.remote(1)

    return {'serial': serial, 'M5_serial': M5_serial, 'T_serial': T_serial,
            'Minit_5D': solution.M_init_5D, 'Minit_4D': solution.M_init_4D, 'Tinit': Tinit, 'F': F, 'f': f, 'M5': M5,
            'GB_5D_lifetime': GB5D.T_lifetime, 'GB_5D_shift': GB5D.T_shift, 'GB_5D_compute': GB5D.compute_time,
            'GB_4D_lifetime': GB4D.T_lifetime, 'GB_4D_shift': GB4D.T_shift, 'GB_4D_compute': GB4D.compute_time,
            'SB_5D_lifetime': SB5D.T_lifetime, 'SB_5D_shift': SB5D.T_shift, 'SB_5D_compute': SB5D.compute_time,
            'SB_4D_lifetime': SB4D.T_lifetime, 'SB_4D_shift': SB4D.T_shift, 'SB_4D_compute': SB4D.compute_time,
            'SB_5D_noreff_lifetime': SB5D_noreff.T_lifetime, 'SB_5D_noreff_shift': SB5D_noreff.T_shift, 'SB_5D_noreff_compute': SB5D_noreff.compute_time,
            'SB_4D_noreff_lifetime': SB4D_noreff.T_lifetime, 'SB_4D_noreff_shift': SB4D_noreff.T_shift, 'SB_4D_noreff_compute': SB4D_noreff.compute_time,
            'SB_5D_fixedg_lifetime': SB5D_fixedg.T_lifetime, 'SB_5D_fixedg_shift': SB5D_fixedg.T_shift, 'SB_5D_fixedg_compute': SB5D_fixedg.compute_time,
            'SB_4D_fixedg_lifetime': SB4D_fixedg.T_lifetime, 'SB_4D_fixedg_shift': SB4D_fixedg.T_shift, 'SB_4D_fixedg_compute': SB4D_fixedg.compute_time,
            'SB_5D_fixedN_lifetime': SB5D_fixedN.T_lifetime, 'SB_5D_fixedN_shift': SB5D_fixedN.T_shift, 'SB_5D_fixedN_compute': SB5D_fixedN.compute_time,
            'SB_4D_fixedN_lifetime': SB4D_fixedN.T_lifetime, 'SB_4D_fixedN_shift': SB4D_fixedN.T_shift, 'SB_4D_fixedN_compute': SB4D_fixedN.compute_time,
            'SB_5D_noPage_lifetime': SB5D_noPage.T_lifetime, 'SB_5D_noPage_shift': SB5D_noPage.T_shift, 'SB_5D_noPage_compute': SB5D_noPage.compute_time,
            'SB_4D_noPage_lifetime': SB4D_noPage.T_lifetime, 'SB_4D_noPage_shift': SB4D_noPage.T_shift, 'SB_4D_noPage_compute': SB4D_noPage.compute_time}

@ray.remote
def map(f, obj, actor):
    return f(obj, actor)

# build soln_grid of M5/Tinit sample points
M5_grid = np.geomspace(2E8, 5E17, 400)
Tinit_grid = np.geomspace(1E5, 5E17, 400)

# generate serial numbers for M5 & Tinit sample grids and write these out
M5_grid_size = len(M5_grid)
grid_M5_serials = np.zeros(M5_grid_size)
grid_M5_values = np.zeros(M5_grid_size)
grid_M5_Tcrossover_GeV = np.zeros(M5_grid_size)
grid_M5_Tcrossover_Kelvin = np.zeros(M5_grid_size)
for serial, M5 in enumerate(M5_grid):
    grid_M5_serials[serial] = serial
    grid_M5_values[serial] = M5

    params = lkit.RS5D.Parameters(M5)
    grid_M5_Tcrossover_GeV[serial] = params.T_crossover
    grid_M5_Tcrossover_Kelvin[serial] = params.T_crossover_Kelvin

m5_df = pd.DataFrame(data={'serial': grid_M5_serials, 'M5_GeV': grid_M5_values,
                           'Tcrossover_GeV': grid_M5_Tcrossover_GeV, 'Tcrossover_Kelvin': grid_M5_Tcrossover_Kelvin})
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

    params = lkit.RS5D.Parameters(M5)
    engine = lkit.RS5D.Model(params)

    try:
        # get mass of Hubble volume expressed in GeV
        M_Hubble = engine.M_Hubble(T=Tinit)

        # compute initial mass in GeV
        M_init = f * M_Hubble

        # constructing a PBHModel with this mass will raise an exception if the mass is out of bounds
        # could possibly just write in the test here, but this way we abstract it into the PBHModel class
        PBH = lkit.RS5D.BlackHole(params, M_init, units='GeV')
    except RuntimeError as e:
        return False

    return True

data = [(M5_serial, T_serial, M5, Tinit) for ((M5_serial, M5), (T_serial, Tinit)) in data_all if is_valid(M5, Tinit, 0.5)]
num_tasks = len(data)
print('-- Sample grid contains {N} valid entries'.format(N=num_tasks))

# assign a serial number to each configuration
data_grid = [{'serial': serial, 'M5_serial': M5_serial, 'T_serial': T_serial, 'M5': M5, 'Tinit': Tinit,
              'F': 0.1, 'f': 0.5} for serial, (M5_serial, T_serial, M5, Tinit) in enumerate(data)]

with lkit.Timer() as compute_timer:
    pb = ProgressBar(num_tasks)
    actor = pb.actor

    # use ray to perform a distributed map of compute_lifetime onto data_grid
    tasks = [map.remote(compute_lifetime, line, actor) for line in data_grid]
    pb.print_until_done()

    soln_grid = ray.get(tasks)

print('-- Task finished, wallclock time = {time:.3} s'.format(time=compute_timer.interval))

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

M_5D_init_GeV = np.zeros(work_size)
M_5D_init_gram = np.zeros(work_size)

M_4D_init_GeV = np.zeros(work_size)
M_4D_init_gram = np.zeros(work_size)

T_init_GeV = np.zeros(work_size)
T_init_Kelvin = np.zeros(work_size)

GB_5D_lifetime_GeV = np.zeros(work_size)
GB_5D_lifetime_Kelvin = np.zeros(work_size)
GB_5D_shift = np.zeros(work_size)
GB_5D_compute = np.zeros(work_size)

GB_4D_lifetime_GeV = np.zeros(work_size)
GB_4D_lifetime_Kelvin = np.zeros(work_size)
GB_4D_shift = np.zeros(work_size)
GB_4D_compute = np.zeros(work_size)

SB_5D_lifetime_GeV = np.zeros(work_size)
SB_5D_lifetime_Kelvin = np.zeros(work_size)
SB_5D_shift = np.zeros(work_size)
SB_5D_compute = np.zeros(work_size)

SB_4D_lifetime_GeV = np.zeros(work_size)
SB_4D_lifetime_Kelvin = np.zeros(work_size)
SB_4D_shift = np.zeros(work_size)
SB_4D_compute = np.zeros(work_size)

SB_5D_noreff_lifetime_GeV = np.zeros(work_size)
SB_5D_noreff_lifetime_Kelvin = np.zeros(work_size)
SB_5D_noreff_shift = np.zeros(work_size)
SB_5D_noreff_compute = np.zeros(work_size)

SB_4D_noreff_lifetime_GeV = np.zeros(work_size)
SB_4D_noreff_lifetime_Kelvin = np.zeros(work_size)
SB_4D_noreff_shift = np.zeros(work_size)
SB_4D_noreff_compute = np.zeros(work_size)

SB_5D_fixedg_lifetime_GeV = np.zeros(work_size)
SB_5D_fixedg_lifetime_Kelvin = np.zeros(work_size)
SB_5D_fixedg_shift = np.zeros(work_size)
SB_5D_fixedg_compute = np.zeros(work_size)

SB_4D_fixedg_lifetime_GeV = np.zeros(work_size)
SB_4D_fixedg_lifetime_Kelvin = np.zeros(work_size)
SB_4D_fixedg_shift = np.zeros(work_size)
SB_4D_fixedg_compute = np.zeros(work_size)

SB_5D_fixedN_lifetime_GeV = np.zeros(work_size)
SB_5D_fixedN_lifetime_Kelvin = np.zeros(work_size)
SB_5D_fixedN_shift = np.zeros(work_size)
SB_5D_fixedN_compute = np.zeros(work_size)

SB_4D_fixedN_lifetime_GeV = np.zeros(work_size)
SB_4D_fixedN_lifetime_Kelvin = np.zeros(work_size)
SB_4D_fixedN_shift = np.zeros(work_size)
SB_4D_fixedN_compute = np.zeros(work_size)

SB_5D_noPage_lifetime_GeV = np.zeros(work_size)
SB_5D_noPage_lifetime_Kelvin = np.zeros(work_size)
SB_5D_noPage_shift = np.zeros(work_size)
SB_5D_noPage_compute = np.zeros(work_size)

SB_4D_noPage_lifetime_GeV = np.zeros(work_size)
SB_4D_noPage_lifetime_Kelvin = np.zeros(work_size)
SB_4D_noPage_shift = np.zeros(work_size)
SB_4D_noPage_compute = np.zeros(work_size)


for line in soln_grid:
    serial = line['serial']
    M5_serial[serial] = line['M5_serial']
    T_serial[serial] = line['T_serial']

    M5[serial] = line['M5']
    F[serial] = line['F']
    f[serial] = line['f']

    M_5D_init_GeV[serial] = line['Minit_5D']
    M_5D_init_gram[serial] = line['Minit_5D'] / lkit.Gram if line['Minit_5D'] is not None else None

    M_4D_init_GeV[serial] = line['Minit_4D']
    M_4D_init_gram[serial] = line['Minit_4D'] / lkit.Gram if line['Minit_4D'] is not None else None

    T_init_GeV[serial] = line['Tinit']
    T_init_Kelvin[serial] = line['Tinit'] / lkit.Kelvin

    GB_5D_lifetime_GeV[serial] = line['GB_5D_lifetime']
    GB_5D_lifetime_Kelvin[serial] = line['GB_5D_lifetime'] / lkit.Kelvin if line['GB_5D_lifetime'] is not None else None
    GB_5D_shift[serial] = line['GB_5D_shift'] / lkit.Kelvin if line['GB_5D_shift'] is not None else None
    GB_5D_compute[serial] = line['GB_5D_compute']

    GB_4D_lifetime_GeV[serial] = line['GB_4D_lifetime']
    GB_4D_lifetime_Kelvin[serial] = line['GB_4D_lifetime'] / lkit.Kelvin if line['GB_4D_lifetime'] is not None else None
    GB_4D_shift[serial] = line['GB_4D_shift'] / lkit.Kelvin if line['GB_4D_shift'] is not None else None
    GB_4D_compute[serial] = line['GB_4D_compute']

    SB_5D_lifetime_GeV[serial] = line['SB_5D_lifetime']
    SB_5D_lifetime_Kelvin[serial] = line['SB_5D_lifetime'] / lkit.Kelvin if line['SB_5D_lifetime'] is not None else None
    SB_5D_shift[serial] = line['SB_5D_shift'] / lkit.Kelvin if line['SB_5D_shift'] is not None else None
    SB_5D_compute[serial] = line['SB_5D_compute']

    SB_4D_lifetime_GeV[serial] = line['SB_4D_lifetime']
    SB_4D_lifetime_Kelvin[serial] = line['SB_4D_lifetime'] / lkit.Kelvin if line['SB_4D_lifetime'] is not None else None
    SB_4D_shift[serial] = line['SB_4D_shift'] / lkit.Kelvin if line['SB_4D_shift'] is not None else None
    SB_4D_compute[serial] = line['SB_4D_compute']

    SB_5D_noreff_lifetime_GeV[serial] = line['SB_5D_noreff_lifetime']
    SB_5D_noreff_lifetime_Kelvin[serial] = line['SB_5D_noreff_lifetime'] / lkit.Kelvin if line['SB_5D_noreff_lifetime'] is not None else None
    SB_5D_noreff_shift[serial] = line['SB_5D_noreff_shift'] / lkit.Kelvin if line['SB_5D_noreff_shift'] is not None else None
    SB_5D_noreff_compute[serial] = line['SB_5D_noreff_compute']

    SB_4D_noreff_lifetime_GeV[serial] = line['SB_4D_noreff_lifetime']
    SB_4D_noreff_lifetime_Kelvin[serial] = line['SB_4D_noreff_lifetime'] / lkit.Kelvin if line['SB_4D_lifetime'] is not None else None
    SB_4D_noreff_shift[serial] = line['SB_4D_noreff_shift'] / lkit.Kelvin if line['SB_4D_noreff_shift'] is not None else None
    SB_4D_noreff_compute[serial] = line['SB_4D_noreff_compute']

    SB_5D_fixedg_lifetime_GeV[serial] = line['SB_5D_fixedg_lifetime']
    SB_5D_fixedg_lifetime_Kelvin[serial] = line['SB_5D_fixedg_lifetime'] / lkit.Kelvin if line['SB_5D_fixedg_lifetime'] is not None else None
    SB_5D_fixedg_shift[serial] = line['SB_5D_fixedg_shift'] / lkit.Kelvin if line['SB_5D_fixedg_shift'] is not None else None
    SB_5D_fixedg_compute[serial] = line['SB_5D_fixedg_compute']

    SB_4D_fixedg_lifetime_GeV[serial] = line['SB_4D_fixedg_lifetime']
    SB_4D_fixedg_lifetime_Kelvin[serial] = line['SB_4D_fixedg_lifetime'] / lkit.Kelvin if line['SB_4D_fixedg_lifetime'] is not None else None
    SB_4D_fixedg_shift[serial] = line['SB_4D_fixedg_shift'] / lkit.Kelvin if line['SB_4D_fixedg_shift'] is not None else None
    SB_4D_fixedg_compute[serial] = line['SB_4D_fixedg_compute']

    SB_5D_fixedN_lifetime_GeV[serial] = line['SB_5D_fixedN_lifetime']
    SB_5D_fixedN_lifetime_Kelvin[serial] = line['SB_5D_fixedN_lifetime'] / lkit.Kelvin if line['SB_5D_fixedN_lifetime'] is not None else None
    SB_5D_fixedN_shift[serial] = line['SB_5D_fixedN_shift'] / lkit.Kelvin if line['SB_5D_fixedN_shift'] is not None else None
    SB_5D_fixedN_compute[serial] = line['SB_5D_fixedN_compute']

    SB_4D_fixedN_lifetime_GeV[serial] = line['SB_4D_fixedN_lifetime']
    SB_4D_fixedN_lifetime_Kelvin[serial] = line['SB_4D_fixedN_lifetime'] / lkit.Kelvin if line['SB_4D_fixedN_lifetime'] is not None else None
    SB_4D_fixedN_shift[serial] = line['SB_4D_fixedN_shift'] / lkit.Kelvin if line['SB_4D_fixedN_shift'] is not None else None
    SB_4D_fixedN_compute[serial] = line['SB_4D_fixedN_compute']

    SB_5D_noPage_lifetime_GeV[serial] = line['SB_5D_noPage_lifetime']
    SB_5D_noPage_lifetime_Kelvin[serial] = line['SB_5D_noPage_lifetime'] / lkit.Kelvin if line['SB_5D_noPage_lifetime'] is not None else None
    SB_5D_noPage_shift[serial] = line['SB_5D_noPage_shift'] / lkit.Kelvin if line['SB_5D_noPage_shift'] is not None else None
    SB_5D_noPage_compute[serial] = line['SB_5D_noPage_compute']

    SB_4D_noPage_lifetime_GeV[serial] = line['SB_4D_noPage_lifetime']
    SB_4D_noPage_lifetime_Kelvin[serial] = line['SB_4D_noPage_lifetime'] / lkit.Kelvin if line['SB_4D_noPage_lifetime'] is not None else None
    SB_4D_noPage_shift[serial] = line['SB_4D_noPage_shift'] / lkit.Kelvin if line['SB_4D_noPage_shift'] is not None else None
    SB_4D_noPage_compute[serial] = line['SB_4D_noPage_compute']


df = pd.DataFrame(data={'M5_serial': M5_serial,
                        'T_serial': T_serial,
                        'M5_GeV': M5,
                        'accretion_F': F,
                        'collapse_f': f,
                        'M_5D_init_GeV': M_5D_init_GeV,
                        'M_5D_init_gram': M_5D_init_gram,
                        'M_4D_init_GeV': M_4D_init_GeV,
                        'M_4D_init_gram': M_4D_init_gram,
                        'T_init_GeV': T_init_GeV,
                        'T_init_Kelvin': T_init_Kelvin,
                        'GB_5D_lifetime_GeV': GB_5D_lifetime_GeV,
                        'GB_5D_lifetime_Kelvin': GB_5D_lifetime_Kelvin,
                        'GB_5D_shift_Kelvin': GB_5D_shift,
                        'GB_5D_compute': GB_5D_compute,
                        'GB_4D_lifetime_GeV': GB_4D_lifetime_GeV,
                        'GB_4D_lifetime_Kelvin': GB_4D_lifetime_Kelvin,
                        'GB_4D_shift_Kelvin': GB_4D_shift,
                        'GB_4D_compute': GB_4D_compute,
                        'SB_5D_lifetime_GeV': SB_5D_lifetime_GeV,
                        'SB_5D_lifetime_Kelvin': SB_5D_lifetime_Kelvin,
                        'SB_5D_shift_Kelvin': SB_5D_shift,
                        'SB_5D_compute': SB_5D_compute,
                        'SB_4D_lifetime_GeV': SB_4D_lifetime_GeV,
                        'SB_4D_lifetime_Kelvin': SB_4D_lifetime_Kelvin,
                        'SB_4D_shift_Kelvin': SB_4D_shift,
                        'SB_4D_compute': SB_4D_compute,
                        'SB_5D_noreff_lifetime_GeV': SB_5D_noreff_lifetime_GeV,
                        'SB_5D_noreff_lifetime_Kelvin': SB_5D_noreff_lifetime_Kelvin,
                        'SB_5D_noreff_shift_Kelvin': SB_5D_noreff_shift,
                        'SB_5D_noreff_compute': SB_5D_noreff_compute,
                        'SB_4D_noreff_lifetime_GeV': SB_4D_noreff_lifetime_GeV,
                        'SB_4D_noreff_lifetime_Kelvin': SB_4D_noreff_lifetime_Kelvin,
                        'SB_4D_noreff_shift_Kelvin': SB_4D_noreff_shift,
                        'SB_4D_noreff_compute': SB_4D_noreff_compute,
                        'SB_5D_fixedg_lifetime_GeV': SB_5D_fixedg_lifetime_GeV,
                        'SB_5D_fixedg_lifetime_Kelvin': SB_5D_fixedg_lifetime_Kelvin,
                        'SB_5D_fixedg_shift_Kelvin': SB_5D_fixedg_shift,
                        'SB_5D_fixedg_compute': SB_5D_fixedg_compute,
                        'SB_4D_fixedg_lifetime_GeV': SB_4D_fixedg_lifetime_GeV,
                        'SB_4D_fixedg_lifetime_Kelvin': SB_4D_fixedg_lifetime_Kelvin,
                        'SB_4D_fixedg_shift_Kelvin': SB_4D_fixedg_shift,
                        'SB_4D_fixedg_compute': SB_4D_fixedg_compute,
                        'SB_5D_fixedN_lifetime_GeV': SB_5D_fixedN_lifetime_GeV,
                        'SB_5D_fixedN_lifetime_Kelvin': SB_5D_fixedN_lifetime_Kelvin,
                        'SB_5D_fixedN_shift_Kelvin': SB_5D_fixedN_shift,
                        'SB_5D_fixedN_compute': SB_5D_fixedN_compute,
                        'SB_4D_fixedN_lifetime_GeV': SB_4D_fixedN_lifetime_GeV,
                        'SB_4D_fixedN_lifetime_Kelvin': SB_4D_fixedN_lifetime_Kelvin,
                        'SB_4D_fixedN_shift_Kelvin': SB_4D_fixedN_shift,
                        'SB_4D_fixedN_compute': SB_4D_fixedN_compute,
                        'SB_5D_noPage_lifetime_GeV': SB_5D_noPage_lifetime_GeV,
                        'SB_5D_noPage_lifetime_Kelvin': SB_5D_noPage_lifetime_Kelvin,
                        'SB_5D_noPage_shift_Kelvin': SB_5D_noPage_shift,
                        'SB_5D_noPage_compute': SB_5D_noPage_compute,
                        'SB_4D_noPage_lifetime_GeV': SB_4D_noPage_lifetime_GeV,
                        'SB_4D_noPage_lifetime_Kelvin': SB_4D_noPage_lifetime_Kelvin,
                        'SB_4D_noPage_shift_Kelvin': SB_4D_noPage_shift,
                        'SB_4D_noPage_compute': SB_4D_noPage_compute}, index=df_index)
df.index.name = 'index'
df.to_csv('mass_lifetime.csv')
