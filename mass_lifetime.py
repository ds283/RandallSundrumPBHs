import numpy as np
import pandas as pd
import argparse

import ray
from ray.actor import ActorHandle
from progressbar import ProgressBar
ray.init(address='auto')

import LifetimeKit as lkit

import itertools


parser = argparse.ArgumentParser()
parser.add_argument("--progress-bar", default=True, action=argparse.BooleanOptionalAction)
args = parser.parse_args()


# mapping between histories to write into output database (keys)
# and internal LifetimeKit names for these models (values)
histories = {'GB_5D': 'GreybodyRS5D',
             'GB_4D': 'GreybodyStandard4D',
             'SB_5D': 'StefanBoltzmannRS5D',
             'SB_4D': 'StefanBoltzmannStandard4D',
             'SB_5D_noreff': 'StefanBoltzmannRS5D-noreff',
             'SB_4D_noreff': 'StefanBoltzmannStandard4D-noreff',
             'SB_5D_fixedg': 'StefanBoltzmannRS5D-fixedg',
             'SB_4D_fixedg': 'StefanBoltzmannStandard4D-fixedg',
             'SB_5D_fixedN': 'StefanBoltzmannRS5D-fixedN',
             'SB_4D_fixedN': 'StefanBoltzmannStandard4D-fixedN',
             'SB_5D_noPage': 'StefanBoltzmannRS5D-noPage',
             'SB_4D_noPage': 'StefanBoltzmannStandard4D-noPage'}


def _build_history_output_labels(h: str):
    label_lifetime_GeV = '{h}_lifetime_GeV'.format(h=h)
    label_lifetime_Kelvin = '{h}_lifetime_Kelvin'.format(h=h)
    label_shift = '{h}_shift'.format(h=h)
    label_Mfinal_GeV = '{h}_Mfinal_GeV'.format(h=h)
    label_Mfinal_Gram = '{h}_Mfinal_Gram'.format(h=h)
    label_compute = '{h}_compute'.format(h=h)

    return label_lifetime_GeV, label_lifetime_Kelvin, label_shift, label_Mfinal_GeV, label_Mfinal_Gram, label_compute


def _build_history_internal_labels(h: str):
    line_lifetime = '{h}_lifetime'.format(h=h)
    line_shift = '{h}_shift'.format(h=h)
    line_Mfinal = '{h}_Mfinal'.format(h=h)
    line_compute = '{h}_compute'.format(h=h)

    return line_lifetime, line_shift, line_Mfinal, line_compute


def compute_lifetime(data, pba: ActorHandle):
    serial = data['serial']
    M5_serial = data['M5_serial']
    T_serial = data['T_serial']
    F_serial = data['F_serial']

    M5 = data['M5']
    Tinit = data['Tinit']
    F = data['F']
    f = data['f']

    params = lkit.RS5D.Parameters(M5)

    solution = lkit.PBHInstance(params, Tinit, models=histories.values(),
                                accretion_efficiency_F=F, collapse_fraction_f=f)

    data = {'serial': serial, 'M5_serial': M5_serial, 'T_serial': T_serial, 'F_serial': F_serial,
            'Minit_5D': solution.M_init_5D, 'Minit_4D': solution.M_init_4D, 'Tinit': Tinit, 'F': F, 'f': f, 'M5': M5}

    for history_label in histories.keys():
        model_label = histories[history_label]

        line_lifetime, line_shift, line_Mfinal, line_compute = _build_history_internal_labels(history_label)
        history = solution.lifetimes[model_label]
        history_data = {line_lifetime: history.T_lifetime,
                        line_shift: history.T_shift,
                        line_Mfinal: history.M_final,
                        line_compute: history.compute_time}

        data = data | history_data

    if pba is not None:
        pba.update.remote(1)

    return data

@ray.remote
def map(f, obj, actor):
    return f(obj, actor)

# build soln_grid of M5/Tinit/F sample points

# lower limit 2E8 GeV roughly corresponds to experimental constraint T_crossover = 1E3 GeV suggested by Guedens et al.
# TODO: Itzi suggests this has since been improved, so that may need changing
# upper limit 5E17 GeV is close to 4D Planck scale, with just a bit of headroom
M5_grid = np.geomspace(2E8, 5E17, 400)

# lower limit 1E5 GeV is arbitrary; black holes that form at these low temperatures are always in the 4D regime
# with the linear Hubble equation, so there is not much need to compute their properties in detail.
# upper limit matches upper limit on M5 grid
Tinit_grid = np.geomspace(1E5, 5E17, 400)
F_grid = np.geomspace(0.001, 1.0, 40)

# generate serial numbers for M5  sample grids and write these out
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

# generate serial numbers for T_init sample grid
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

# generate serial numbers for F sample grid
F_grid_size = len(F_grid)
grid_F_serials = np.zeros(F_grid_size)
grid_F_values = np.zeros(F_grid_size)
for serial, F in enumerate(F_grid):
    grid_F_serials[serial] = serial
    grid_F_values[serial] = F

F_df = pd.DataFrame(data={'serial': grid_F_serials, 'F': grid_F_values})
F_df.set_index('serial', inplace=True)
F_df.to_csv('F_grid.csv')

# now combine M5, Tinit and F grids into a single large grid
data_all = itertools.product(enumerate(M5_grid), enumerate(Tinit_grid), enumerate(F_grid))

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


data = [(M5_serial, T_serial, F_serial, M5, Tinit, F) for ((M5_serial, M5), (T_serial, Tinit), (F_serial, F)) in
        data_all if is_valid(M5, Tinit, 0.5)]
num_tasks = len(data)
print('-- Sample grid contains {N} valid entries'.format(N=num_tasks))

# assign a serial number to each configuration
data_grid = [{'serial': serial, 'M5_serial': M5_serial, 'T_serial': T_serial, 'M5': M5, 'Tinit': Tinit,
              'F_serial': F_serial, 'F': F, 'f': 0.5} for serial, (M5_serial, T_serial, F_serial, M5, Tinit, F) in enumerate(data)]

with lkit.Timer() as compute_timer:
    if args.progress_bar:
        pb = ProgressBar(num_tasks)
        actor = pb.actor
    else:
        pb = None
        actor = None

    # use ray to perform a distributed map of compute_lifetime onto data_grid
    tasks = [map.remote(compute_lifetime, line, actor) for line in data_grid]

    if pb is not None:
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
F_serial = np.zeros(work_size)

M5 = np.zeros(work_size)
F = np.zeros(work_size)
f = np.zeros(work_size)

M_5D_init_GeV = np.zeros(work_size)
M_5D_init_gram = np.zeros(work_size)

M_4D_init_GeV = np.zeros(work_size)
M_4D_init_gram = np.zeros(work_size)

T_init_GeV = np.zeros(work_size)
T_init_Kelvin = np.zeros(work_size)

arrays = {}

for h in histories.keys():
    label_lifetime_GeV, label_lifetime_Kelvin, label_shift, label_Mfinal_GeV, label_Mfinal_Gram, label_compute = \
        _build_history_output_labels(h)

    arrays[label_lifetime_GeV] = np.zeros(work_size)
    arrays[label_lifetime_Kelvin] = np.zeros(work_size)
    arrays[label_shift] = np.zeros(work_size)
    arrays[label_Mfinal_GeV] = np.zeros(work_size)
    arrays[label_Mfinal_Gram] = np.zeros(work_size)
    arrays[label_compute] = np.zeros(work_size)

for line in soln_grid:
    serial = line['serial']

    M5_serial[serial] = line['M5_serial']
    T_serial[serial] = line['T_serial']
    F_serial[serial] = line['F_serial']

    M5[serial] = line['M5']
    F[serial] = line['F']
    f[serial] = line['f']

    M_5D_init_GeV[serial] = line['Minit_5D']
    M_5D_init_gram[serial] = line['Minit_5D'] / lkit.Gram if line['Minit_5D'] is not None else None

    M_4D_init_GeV[serial] = line['Minit_4D']
    M_4D_init_gram[serial] = line['Minit_4D'] / lkit.Gram if line['Minit_4D'] is not None else None

    T_init_GeV[serial] = line['Tinit']
    T_init_Kelvin[serial] = line['Tinit'] / lkit.Kelvin

    for h in histories.keys():
        label_lifetime_GeV, label_lifetime_Kelvin, label_shift, label_Mfinal_GeV, label_Mfinal_Gram, label_compute = \
            _build_history_output_labels(h)

        line_lifetime, line_shift, line_Mfinal, line_compute = _build_history_internal_labels(h)

        arrays[label_lifetime_GeV][serial] = line[line_lifetime]
        arrays[label_lifetime_Kelvin][serial] = line[line_lifetime] / lkit.Kelvin if line[line_lifetime] is not None else None
        arrays[label_shift][serial] = line[line_shift] / lkit.Kelvin if line[line_shift] is not None else None
        arrays[label_Mfinal_GeV][serial] = line[line_Mfinal]
        arrays[label_Mfinal_Gram][serial] = line[line_Mfinal] / lkit.Gram if line[line_Mfinal] is not None else None
        arrays[label_compute][serial] = line[line_compute]

data = {'M5_serial': M5_serial,
        'T_serial': T_serial,
        'F_serial': F_serial,
        'M5_GeV': M5,
        'accretion_F': F,
        'collapse_f': f,
        'M_5D_init_GeV': M_5D_init_GeV,
        'M_5D_init_gram': M_5D_init_gram,
        'M_4D_init_GeV': M_4D_init_GeV,
        'M_4D_init_gram': M_4D_init_gram,
        'T_init_GeV': T_init_GeV,
        'T_init_Kelvin': T_init_Kelvin} | arrays

df = pd.DataFrame(data, index=df_index)
df.index.name = 'index'
df.to_csv('mass_lifetime.csv')
