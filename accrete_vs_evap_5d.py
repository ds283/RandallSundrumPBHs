## Study whether black holes that form in the 5D regime always accrete faster than they evaporate,
## or vice-versa
## https://app.asana.com/0/1201954909529908/1202426213466302/f

import argparse
import math
import subprocess
import sys
from functools import partial
from typing import List

import numpy as np
import ray
from ray.actor import ActorHandle
from ray.data import Dataset, set_progress_bars

import CacheKit as ckit
import LifetimeKit as lkit
from LifetimeKit.constants import T_CMB

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
histories = {'GB_5D': 'GreybodyRS5D'}

def _build_labels(h: str):
    return {'is_5D': '{h}_is_5D'.format(h=h),
            'evap_faster': '{h}_evap_faster'.format(h=h),
            'net_evap_rate_GeV_sq': '{h}_net_evap_rate_GeV_sq'.format(h=h),
            'net_evap_rate_Gram_Yr': '{h}_net_evap_rate_Gram_Yr'.format(h=h)}


def compute_lifetime(cache: ActorHandle, serial_batch: List[int]) -> List[float]:
    batch = []
    times = []

    for serial in serial_batch:
        with lkit.Timer() as timer:
            data = ray.get(cache.get_work_item.remote(serial))

            # extract parameter set for this work item
            _serial, M5, Tinit, Tfinal, F, f = data

            params = lkit.RS5D.Parameters(M5)
            engine = lkit.RS5D.Model(params)

            M_Hubble = engine.M_Hubble(T=Tinit)
            M_init = f * M_Hubble

            model = lkit.RS5D_Friedlander.LifetimeModel(engine, accretion_efficiency_F=F, use_Page_suppression=True,
                                                        use_effective_radius=True)
            PBH = lkit.RS5D.SpinlessBlackHole(params, M_init, units='GeV')

            evap_rate = model._dMdt_evaporation(Tinit, PBH)
            accrete_rate = model._dMdt_accretion(Tinit, PBH)

            net_evap_rate = -(evap_rate + accrete_rate)

            data = {'serial': serial,
                    'Minit_5D_GeV': M_init,
                    'Minit_5D_Gram': M_init/lkit.Gram,
                    'GB_5D_is_5D': PBH.is_5D,
                    'GB_5D_evap_faster': math.fabs(evap_rate) > math.fabs(accrete_rate),
                    'GB_5D_net_evap_rate_GeV_sq': net_evap_rate,
                    'GB_5D_net_evap_rate_Gram_Yr': net_evap_rate / (lkit.Gram / lkit.Year)}

            batch.append(data)

        times.append(timer.interval)

    # send these results to the database
    future = cache.write_work_item.remote(batch)

    # we want to ensure the results actually get written before we move on, so block until this has happened.
    # Otherwise, we don't get the benefits from checkpointing results into the database - if the job crashes,
    # everything will still be lost
    out = ray.get(future)

    return times

def _test_valid(data) -> bool:
    M5_data, Tinit_data, Tfinal_data, F_data, f_data = data

    M5_serial, M5 = M5_data
    Tinit_serial, Tinit = Tinit_data
    Tfinal_serial, Tfinal = Tfinal_data
    F_serial, F = F_data
    f_serial, f = f_data

    # Reject combinations where the initial temperature is larger than the 5D Planck mass.
    # In this case, the quantum gravity corrections are not under control and the scenario
    # probably does not make sense
    if Tinit > M5:
        return False

    # reject combinations where there is runaway 4D accretion
    # The criterion for this is f * F * (1 + delta) > 8/(3*alpha^2) where alpha is the
    # effective radius scaling parameter. Here we are going to solve histories with and without
    # using the effective radius, so with alpha = 3 sqrt(3) / 2 we get the combination
    # 32.0/81.0. This is used in Guedens et al., but seems to have first been reported in the
    # early Zel'dovich & Novikov paper ("The hypothesis of cores retarded during expansion
    # and the hot cosmological model"), see their Eq. (2)
    if F * f > 32.0 / 81.0:
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
        PBH = lkit.RS5D.SpinlessBlackHole(params, M=M_init, units='GeV')
    except RuntimeError:
        return False

    return True

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

cache: ckit.Cache = \
    ckit.Cache.remote(list(histories.keys()), _build_labels, get_git_revision_hash(),
                      standard_columns=['Minit_5D_GeV', 'Minit_5D_Gram'])

if args.create_database is not None:
    print('mass_lifetime_grid.py: target radiation temperature T_rad = {Trad_GeV:.5} GeV = {Trad_K:.5} '
          'Kelvin'.format(Trad_GeV=Trad_final_GeV, Trad_K=Trad_final_GeV / lkit.Kelvin))

    # build soln_grid of M5/Tinit/F sample points

    # lower limit 2E8 GeV roughly corresponds to experimental constraint T_crossover = 1E3 GeV suggested by Guedens et al.
    # TODO: Itzi suggests this has since been improved, so that may need changing
    # upper limit 5E17 GeV is close to 4D Planck scale, with just a bit of headroom
    M5_grid = np.geomspace(2E8, 5E17, 500)

    # lower limit 1E5 GeV is arbitrary; black holes that form at these low temperatures are always in the 4D regime
    # with the linear Hubble equation, so there is not much need to compute their properties in detail.
    # upper limit matches upper limit on M5 grid
    Tinit_grid = np.geomspace(1E5, 5E17, 500)

    F_grid = np.geomspace(0.001, 1.0, 10)

    # f = 0.395 is just under Zel'dovich-Novikov limit for Ff = 32/81
    f_grid = [0.395]

    # don't scan over final temperatures, so there is just a single element in the Tfinal grid
    Tfinal_grid = [Trad_final_GeV]

    obj = cache.create_database.remote(args.create_database, M5_grid, Tinit_grid, Tfinal_grid, F_grid, f_grid,
                                       _test_valid)
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
