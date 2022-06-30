import argparse
from datetime import datetime
import sys
from functools import partial
from typing import List

import numpy as np
import ray
from ray.actor import ActorHandle
from ray.data import Dataset, set_progress_bars

import LifetimeKit as lkit
from LifetimeKit.constants import T_CMB
import CacheKit as ckit
from progressbar import ProgressBar, ProgressBarActor

# parse arguments supplied on the command line
parser = argparse.ArgumentParser()
parser.add_argument("--Trad-final", nargs=2, help='specify final radiation temperature')
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

if args.Trad_final is None:
    Trad_final = T_CMB
    Trad_units = 'kelvin'
else:
    try:
        Trad_final = float(args.Trad_final[0])
    except ValueError:
        raise RuntimeError('Could not interpret specified final radiation temperature "{value} {unit}" as a '
                           'number'.format(value=args.Trad_final[0], unit=args.Trad_final[1]))
    Trad_units = args.Trad_final[1].lower()

temperature_conversion = {'kelvin': lkit.Kelvin, 'gev': 1.0}
if Trad_units not in temperature_conversion:
    raise RuntimeError('Unit "{unit}" not understood'.format(unit=Trad_units))

temperature_to_GeV = temperature_conversion[Trad_units]
Trad_final_GeV = Trad_final * temperature_to_GeV

# mapping between histories to write into output database (keys)
# and internal LifetimeKit names for these models (values)
histories = {'spin0': 'spin0',
             'spin0pt3': 'spin0pt3',
             'spin0pt7': 'spin0pt7'}

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
            'initially_5D': '{h}_initially_5D'.format(h=h),
            'Mmax_GeV': '{h}_Mmax_GeV'.format(h=h),
            'Mmax_Gram': '{h}_Mmax_Gram'.format(h=h),
            'Jinit': '{h}_Jinit'.format(h=h),
            'J_over_Jmax_init': '{h}_J_over_Jmax_init'.format(h=h),
            'Jfinal': '{h}_Jfinal'.format(h=h),
            'J_over_Jmax_final': '{h}_J_over_Jmax_final'.format(h=h),
            'Jmax': '{h}_Jmax'.format(h=h),
            'J_over_Jmax_max': '{h}_J_over_Jmax_max'.format(h=h),
            'evaporated': '{h}_evaporated'.format(h=h),
            'runaway_accretion': '{h}_runaway_accretion'.format(h=h),
            'compute': '{h}_compute'.format(h=h)}

def compute_lifetime(cache: ActorHandle, serial_batch: List[int]) -> List[float]:
    batch = []
    times = []

    for serial in serial_batch:
        with lkit.Timer() as timer:
            data = ray.get(cache.get_work_item.remote(serial))

            # extract parameter set for this work item
            _serial, M5, Tinit, Tfinal, F, f = data

            params = lkit.RS5D.Parameters(M5)

            solution_spin0 = lkit.PBHInstance(params, Tinit, T_rad_final=Tfinal, units='GeV', models=['SpinningRS5D'],
                                              J_over_Jmax=0.0, accretion_efficiency_F=F, collapse_fraction_f=f)
            solution_spin0pt3 = lkit.PBHInstance(params, Tinit, T_rad_final=Tfinal, units='GeV',
                                                 models=['SpinningRS5D'], J_over_Jmax=0.3, accretion_efficiency_F=F,
                                                 collapse_fraction_f=f)
            solution_spin0pt7 = lkit.PBHInstance(params, Tinit, T_rad_final=Tfinal, units='GeV',
                                                 models=['SpinningRS5D'], J_over_Jmax=0.7, accretion_efficiency_F=F,
                                                 collapse_fraction_f=f)

            data = {'serial': serial,
                    'timestamp': datetime.now(),
                    'Trad_final_GeV': Tfinal,
                    'Trad_final_Kelvin': Tfinal/lkit.Kelvin,
                    'Minit_5D_GeV': solution_spin0.M_init_5D,
                    'Minit_5D_Gram': solution_spin0.M_init_5D/lkit.Gram,
                    'Minit_4D_GeV': solution_spin0.M_init_4D,
                    'Minit_4D_Gram': solution_spin0.M_init_4D/lkit.Gram}

            histories = {'spin0': solution_spin0.lifetimes['SpinningRS5D'],
                         'spin0pt3': solution_spin0pt3.lifetimes['SpinningRS5D'],
                         'spin0pt7': solution_spin0pt7.lifetimes['SpinningRS5D']}

            for history_label in histories.keys():
                labels = _build_labels(history_label)
                history = histories[history_label]

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
                          labels['initially_5D']: history.initially_5D,
                          labels['Jinit']: history.J_init,
                          labels['J_over_Jmax_init']: history.J_over_Jmax_init,
                          labels['Jfinal']: history.J_final,
                          labels['J_over_Jmax_final']: history.J_over_Jmax_final,
                          labels['Jmax']: history.J_max,
                          labels['J_over_Jmax_max']: history.J_over_Jmax_max,
                          labels['evaporated']: history.evaporated,
                          labels['runaway_accretion']: history.runaway_accretion,
                          labels['compute']: history.compute_time}
                data = data | h_data

        batch.append(data)
        times.append(timer.interval)

    cache.write_work_item.remote(batch)

    return times


cache: ckit.Cache = \
    ckit.Cache.remote(list(histories.keys()), _build_labels,
                      standard_columns=['timestamp', 'Trad_final_GeV', 'Trad_final_Kelvin',
                                        'Minit_5D_GeV', 'Minit_5D_Gram', 'Minit_4D_GeV', 'Minit_4D_Gram'])

if args.create_database is not None:
    print('spin_evolution_grid.py: target radiation temperature T_rad = {Trad_GeV:.5} GeV = {Trad_K:.5} '
          'Kelvin'.format(Trad_GeV=Trad_final_GeV, Trad_K=Trad_final_GeV / lkit.Kelvin))

    # build soln_grid of M5/Tinit/F sample points

    # lower limit 2E8 GeV roughly corresponds to experimental constraint T_crossover = 1E3 GeV suggested by Guedens et al.
    # TODO: Itzi suggests this has since been improved, so that may need changing
    # upper limit 5E17 GeV is close to 4D Planck scale, with just a bit of headroom
    M5_grid = np.geomspace(2E8, 5E17, 10)

    # lower limit 1E5 GeV is arbitrary; black holes that form at these low temperatures are always in the 4D regime
    # with the linear Hubble equation, so there is not much need to compute their properties in detail.
    # upper limit matches upper limit on M5 grid
    Tinit_grid = np.geomspace(1E5, 5E17, 500)

    F_grid = np.geomspace(0.001, 1.0, 10)

    # f = 0.395 is just under Zel'dovich-Novikov limit for Ff = 32/81
    f_grid = [0.395]

    # don't scan over final temperatures, so there is just a single element in the Tfinal grid
    Tfinal_grid = [Trad_final_GeV]

    obj = cache.create_database.remote(args.create_database, M5_grid, Tinit_grid, Tfinal_grid, F_grid, f_grid)
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
