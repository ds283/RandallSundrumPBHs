import numpy as np
import pandas as pd

import ray
from ray.actor import ActorHandle
ray.init()

from asyncio import Event
from typing import Tuple

import LifetimeKit as lkit

import itertools
from tqdm import tqdm


# adapted from https://docs.ray.io/en/latest/ray-core/examples/progress_bar.html
@ray.remote
class ProgressBarActor:
    counter: int
    delta: int
    event: Event

    def __init__(self) -> None:
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed: int) -> None:
        '''
        Updates the ProgressBar with the incremental number of items that were just completed
        '''
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self) -> Tuple[int, int]:
        '''
        Blocking call.

        Waits until somebody calls update(), then returns a tuple of the number of updates since the last call
        to wait_for_update(), and the total number of completed items.
        '''
        await self.event.wait()
        self.event.clear()
        saved_delta = self.delta
        self.delta = 0
        return saved_delta, self.counter

    def get_countter(self) -> int:
        '''
        Returns the total number of complete items
        '''
        return self.counter


class ProgressBar:
    progress_actor: ActorHandle
    total: int
    description: str
    pbar: tqdm

    def __init__(self, total: int, description: str=""):
        # Ray actions don't seem to play nice with mypy, generating a spurious warning for the
        # following line, which we need to suppress. The code is fine.
        self.progress_actor = ProgressBarActor.remote()     # type: ignore
        self.total = total
        self.description = description

    @property
    def actor(self) -> ActorHandle:
        """
        Returns a reference to the remote ProgressBarActor.
        When you complete tasks, call update() on the actor
        """
        return self.progress_actor

    def print_until_done(self) -> None:
        """
        Blocking call.

        Do this after starting a series of remote Ray tasks, to which you've passed the actor handle. Each of them
        calls update() on the actor. When the progress meter reaches 100%, this method returns.
        """
        pbar = tqdm(desc=self.description, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return

def compute_lifetime(data, pba: ActorHandle):
    serial = data['serial']
    M5_serial = data['M5_serial']
    T_serial = data['T_serial']

    M5 = data['M5']
    Tinit = data['Tinit']
    F = data['F']
    f = data['f']

    params = lkit.RS5D.Parameters(M5)

    solution = lkit.PBHInstance(params, Tinit, models=['GreybodyRS5D', 'GreybodyStandard4D'],
                                accretion_efficiency_F=F, collapse_fraction_f=f)

    SB5D = solution.lifetimes['GreybodyRS5D']
    SB4D = solution.lifetimes['GreybodyStandard4D']

    pba.update.remote(1)

    return {'serial': serial, 'M5_serial': M5_serial, 'T_serial': T_serial,
            'Minit_5D': solution.M_init_5D, 'Minit_4D':solution.M_init_4D, 'Tinit': Tinit, 'F': F, 'f': f, 'M5': M5,
            'SB_5D_lifetime': SB5D.T_lifetime, 'SB_5D_shift': SB5D.T_shift, 'SB_5D_compute': SB5D.compute_time,
            'SB_4D_lifetime': SB4D.T_lifetime, 'SB_4D_shift': SB4D.T_shift, 'SB_4D_compute': SB4D.compute_time}

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
        M_PBH = lkit.RS5D.BlackHole(params, M_init, units='GeV')
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

    M_5D_init_GeV[serial] = line['Minit_5D']
    M_5D_init_gram[serial] = line['Minit_5D'] / lkit.Gram if line['Minit_5D'] is not None else None

    M_4D_init_GeV[serial] = line['Minit_4D']
    M_4D_init_gram[serial] = line['Minit_4D'] / lkit.Gram if line['Minit_4D'] is not None else None

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
                        'M_5D_init_GeV': M_5D_init_GeV,
                        'M_5D_init_gram': M_5D_init_gram,
                        'M_4D_init_GeV': M_4D_init_GeV,
                        'M_4D_init_gram': M_4D_init_gram,
                        'T_init_GeV': T_init_GeV,
                        'T_init_Kelvin': T_init_Kelvin,
                        'SB_5D_lifetime_GeV': SB_5D_lifetime_GeV,
                        'SB_5D_lifetime_Kelvin': SB_5D_lifetime_Kelvin,
                        'SB_5D_shift_Kelvin': SB_5D_shift,
                        'SB_5D_compute': SB_5D_compute,
                        'SB_4D_lifetime_GeV': SB_4D_lifetime_GeV,
                        'SB_4D_lifetime_Kelvin': SB_4D_lifetime_Kelvin,
                        'SB_4D_shift_Kelvin': SB_4D_shift,
                        'SB_4D_compute': SB_4D_compute}, index=df_index)
df.index.name = 'index'
df.to_csv('mass_lifetime.csv')
