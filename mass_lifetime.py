import argparse
import sys
from functools import partial
from typing import List

import numpy as np
import ray
from ray.actor import ActorHandle
from ray.data import Dataset, set_progress_bars

import LifetimeKit as lkit
import CacheKit as ckit

# parse arguments supplied on the command line
parser = argparse.ArgumentParser()
parser.add_argument("--progress-bar", default=True, action=argparse.BooleanOptionalAction,
                    help='display real time progress bar (interactive tty only)')
parser.add_argument('--create-database', default=None,
                    help='create a database cache in the specified file')
parser.add_argument('--database', default=None,
                    help='read/write work items using the specified database cache')
parser.add_argument('--compute', default=True, action=argparse.BooleanOptionalAction,
                    help='enable/disable computation of work items (use in conjunction with --create-database')
parser.add_argument('--ray-address', default='auto', type=str,
                    help='specify address of Ray cluster')
args = parser.parse_args()

if args.create_database is None and args.database is None:
    parser.print_help()
    sys.exit()

# connect to ray cluster on supplied address; defaults to 'auto' meaning a locally running cluster
ray.init(address=args.ray_address)

if args.progress_bar:
    set_progress_bars(True)
else:
    set_progress_bars(False)


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


def _build_labels(h: str):
    return {'lifetime_GeV': '{h}_lifetime_GeV'.format(h=h),
            'lifetime_Kelvin': '{h}_lifetime_Kelvin'.format(h=h),
            'shift': '{h}_shift'.format(h=h),
            'Mfinal_GeV': '{h}_Mfinal_GeV'.format(h=h),
            'Mfinal_Gram': '{h}_Mfinal_Gram'.format(h=h),
            '4D_to_5D_GeV': '{h}_4D_to_5D_GeV'.format(h=h),
            '4D_to_5D_Kelvin': '{h}_4D_to_5D_Kelvin'.format(h=h),
            '5D_to_4D_GeV': '{h}_5D_to_4D_GeV'.format(h=h),
            '5D_to_4D_Kelvin': '{h}_5D_to_4D_Kelvin'.format(h=h),
            'Mmax_GeV': '{h}_Mmax_GeV'.format(h=h),
            'Mmax_Gram': '{h}_Mmax_Gram'.format(h=h),
            'compute': '{h}_compute'.format(h=h)}


def compute_lifetime(cache: ActorHandle, serial_batch: List[int]) -> List[float]:
    batch = []
    times = []

    for serial in serial_batch:
        data = ray.get(cache.get_work_item.remote(serial))

        # extract parameter set for this work item
        _serial, M5, Tinit, F, f = data

        params = lkit.RS5D.Parameters(M5)

        solution = lkit.PBHInstance(params, Tinit, models=histories.values(),
                                    accretion_efficiency_F=F, collapse_fraction_f=f)

        data = {'serial': serial,
                'Minit_5D_GeV': solution.M_init_5D,
                'Minit_5D_Gram': solution.M_init_5D/lkit.Gram,
                'Minit_4D_GeV': solution.M_init_4D,
                'Minit_4D_Gram': solution.M_init_4D/lkit.Gram}

        with lkit.Timer() as timer:
            for history_label in histories.keys():
                model_label = histories[history_label]

                labels = _build_labels(history_label)

                history = solution.lifetimes[model_label]

                h_data = {labels['lifetime_GeV']: history.T_lifetime,
                          labels['lifetime_Kelvin']: history.T_lifetime / lkit.Kelvin,
                          labels['shift']: history.T_shift,
                          labels['Mfinal_GeV']: history.M_final,
                          labels['Mfinal_Gram']: history.M_final / lkit.Gram,
                          labels['Mmax_GeV']: history.M_max,
                          labels['Mmax_Gram']: history.M_max / lkit.Gram,
                          labels['4D_to_5D_GeV']: history.T_transition_4Dto5D,
                          labels['4D_to_5D_Kelvin']: None if history.T_transition_4Dto5D is None else history.T_transition_4Dto5D / lkit.Kelvin,
                          labels['5D_to_4D_GeV']: history.T_transition_5Dto4D,
                          labels['5D_to_4D_Kelvin']: None if history.T_transition_5Dto4D is None else history.T_transition_5Dto4D / lkit.Kelvin,
                          labels['compute']: history.compute_time}
                data = data | h_data

        batch.append(data)
        times.append(timer.interval)

    cache.write_work_item.remote(batch)

    return times

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
f_grid = [0.5]

cache: ckit.Cache = ckit.Cache.remote(list(histories.keys()), _build_labels)

if args.create_database is not None:
    obj = cache.create_database.remote(args.create_database, M5_grid, Tinit_grid, F_grid, f_grid)
    output = ray.get(obj)
else:
    obj = cache.open_database.remote(args.database)
    output = ray.get(obj)

if args.compute:
    work_list: Dataset = ray.get(cache.get_work_list.remote())
    work_list_size = work_list.count()
    total_work_size: int = ray.get(cache.get_total_work_size.remote())
    print('** Retrieved {n}/{tot} ({percent:.2f}%) work list items that require '
          'processing'.format(n=work_list_size, tot=total_work_size,
                              percent=100.0*float(work_list_size)/float(total_work_size)))
    work_list.map_batches(partial(compute_lifetime, cache))
