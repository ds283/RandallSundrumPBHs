import argparse
import sys
from datetime import datetime
from typing import List

import pandas as pd
import ray
from ray.data import from_items, set_progress_bars

import LifetimeKit as lkit

# parse arguments supplied on the command line
parser = argparse.ArgumentParser()
parser.add_argument("--progress-bar", default=True, action=argparse.BooleanOptionalAction,
                    help='display real time progress bar (interactive tty only)')
parser.add_argument('--ray-address', default='auto', type=str,
                    help='specify address of Ray cluster')
args = parser.parse_args()

# connect to ray cluster on supplied address; defaults to 'auto' meaning a locally running cluster
ray.init(address=args.ray_address)

if args.progress_bar:
    set_progress_bars(True)
else:
    set_progress_bars(False)


def compute_emission_rates(serial_batch) -> List[float]:
    batch = []
    times = []

    for data in serial_batch:
        with lkit.Timer() as timer:
            _serial, M5, Tinit, F, f = data

            params = lkit.RS5D.Parameters(M5)
            engine = lkit.RS5D.Model(params)

            # get mass of Hubble volume expressed in GeV
            M_Hubble = engine.M_Hubble(T=Tinit)

            # compute initial mass in GeV
            M_init = f * M_Hubble

            model = lkit.RS5D_Friedlander.LifetimeModel(engine, accretion_efficiency_F=F,
                                                        use_Page_suppression=True, use_effective_radius=True)
            soln = lkit.PBHLifetimeModel(M_init, Tinit, model, num_samples=500, compute_rates=False)

            PBH = lkit.RS5D.SpinlessBlackHole(params, M=soln.M_final, units='GeV', strict=False)

            data = {'serial': _serial,
                    'timestamp': datetime.now(),
                    'M5_GeV': M5,
                    'Minit_5D_GeV': M_init,
                    'Minit_5D_Gram': M_init/lkit.Gram,
                    'F': F,
                    'f': f,
                    'Tinit_GeV': Tinit,
                    'Tinit_Kelvin': Tinit/lkit.Kelvin,
                    'lifetime_GeV': soln.T_lifetime,
                    'lifetime_Kelvin': soln.T_lifetime/lkit.Kelvin,
                    'Mfinal_5D_GeV': soln.M_final,
                    'Mfinal_5D_Gram': soln.M_final/lkit.Gram,
                    'initially_5D': soln.initially_5D,
                    'finally_5D':PBH.is_5D,
                    'evaporated': soln.evaporated,
                    'T_Hawking_GeV': PBH.T_Hawking,
                    'T_Hawking_Kelvin': PBH.T_Hawking/lkit.Kelvin,
                    'quarks': -model._dMdt_quarks(0.0, PBH) / (lkit.Gram/lkit.Year),
                    'leptons': -model._dMdt_leptons(0.0, PBH) / (lkit.Gram/lkit.Year),
                    'photons': -model._dMdt_photons(0.0, PBH) / (lkit.Gram/lkit.Year),
                    'gluons': -model._dMdt_gluons(0.0, PBH) / (lkit.Gram/lkit.Year),
                    'EW bosons': -model._dMdt_EW_bosons(0.0, PBH) / (lkit.Gram/lkit.Year),
                    'gravitons': -model._dMdt_graviton5D(0.0, PBH) / (lkit.Gram/lkit.Year)}

        batch.append(data)
        times.append(timer.interval)

    return batch

accreteF = 0.372759372
collapsef = 0.395

df = pd.read_csv('evaporation_boundaryJ=0.csv')

work_list = from_items([(serial, row['M5'], row['Tinit'], accreteF, collapsef) for serial, row in df.iterrows()])
output = work_list.map_batches(compute_emission_rates)

out_df = output.to_pandas()
out_df.set_index('serial', inplace=True)
out_df.to_csv('evaporation_productsJ=0.csv')
