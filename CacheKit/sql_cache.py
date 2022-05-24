from pathlib import Path
from itertools import product
from typing import List, Callable, Dict, Tuple

import ray
from ray.data import Dataset
import sqlalchemy as sqla
from sqlalchemy.dialects.sqlite import insert

import LifetimeKit as lkit


def is_valid(data):
    M5_data, Tinit_data, F_data, f_data = data

    M5_serial, M5 = M5_data
    Tinit_serial, Tinit = Tinit_data
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
    if F*f > 32.0/81.0:
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


@ray.remote
class Cache:

    def __init__(self, histories: List[str], label_maker: Callable[[str], Dict[str, str]]):
        self._engine = None
        self._metadata: sqla.MetaData = None

        self._histories: List[str] = histories
        self._label_maker: Callable[[str], Dict[str, str]] = label_maker


    def _get_engine(self, db_name: str, expect_exists: bool=False):
        db_file = Path(db_name).resolve()

        if not expect_exists and db_file.exists():
            raise RuntimeError('Specified database cache {path} already exists'.format(path=db_name))

        if expect_exists and not db_file.exists():
            raise RuntimeError('Specified database cache {path} does not exist; please create using '
                               '--create-database'.format(path=db_name))

        if db_file.is_dir():
            raise RuntimeError('Specified database cache {path} is a directory'.format(path=db_name))

        # if file does not exist, ensure its parent directories exist
        if not db_file.exists():
            db_file.parents[0].mkdir(exist_ok=True, parents=True)

        self._engine = sqla.create_engine('sqlite:///{name}'.format(name=str(db_name)), future=True)
        self._metadata = sqla.MetaData()

        self._M5_table = sqla.Table('M5_grid', self._metadata,
                                    sqla.Column('serial', sqla.Integer, primary_key=True),
                                    sqla.Column('value_GeV', sqla.Float(precision=64)),
                                    sqla.Column('Tcrossover_GeV', sqla.Float(precision=64)),
                                    sqla.Column('Tcrossover_Kelvin', sqla.Float(precision=64)))

        self._Tinit_table = sqla.Table('Tinit_grid', self._metadata,
                                       sqla.Column('serial', sqla.Integer, primary_key=True),
                                       sqla.Column('value_GeV', sqla.Float(precision=64)),
                                       sqla.Column('value_Kelvin', sqla.Float(precision=64)))

        self._F_table = sqla.Table('accrete_F_grid', self._metadata,
                                   sqla.Column('serial', sqla.Integer, primary_key=True),
                                   sqla.Column('value', sqla.Float(precision=64)))

        self._f_table = sqla.Table('collapse_f_grid', self._metadata,
                                   sqla.Column('serial', sqla.Integer, primary_key=True),
                                   sqla.Column('value', sqla.Float(precision=64)))

        self._work_table = sqla.Table('work_grid', self._metadata,
                                      sqla.Column('serial', sqla.Integer, primary_key=True),
                                      sqla.Column('M5_serial', sqla.Integer, sqla.ForeignKey('M5_grid.serial')),
                                      sqla.Column('Tinit_serial', sqla.Integer, sqla.ForeignKey('Tinit_grid.serial')),
                                      sqla.Column('accrete_F_serial', sqla.Integer, sqla.ForeignKey('accrete_F_grid.serial')),
                                      sqla.Column('collapse_f_serial', sqla.Integer, sqla.ForeignKey('collapse_f_grid.serial')))

        self._data_table = sqla.Table('data', self._metadata,
                                      sqla.Column('serial', sqla.Integer, sqla.ForeignKey('work_grid.serial'), primary_key=True),
                                      sqla.Column('timestamp', sqla.DateTime()),
                                      sqla.Column('Minit_5D_GeV', sqla.Float(precision=64)),
                                      sqla.Column('Minit_5D_Gram', sqla.Float(precision=64)),
                                      sqla.Column('Minit_4D_GeV', sqla.Float(precision=64)),
                                      sqla.Column('Minit_4D_Gram', sqla.Float(precision=64))
        )

        # append column labels to data table for each data item stored, for each history type
        for label in self._histories:
            column_labels = self._label_maker(label).values()
            for clabel in column_labels:
                self._data_table.append_column(sqla.Column(clabel, sqla.Float(precision=64)))


    def create_database(self, db_name: str, M5_grid: List[float], Tinit_grid: List[float],
                        F_grid: List[float], f_grid: List[float]) -> None:
        """
        Create a cache database with the specified grids in M5, Tinit, and F
        :param db_name:
        :param M5_grid:
        :param Tinit_grid:
        :param F_grid:
        :param f_grid:
        :return:
        """
        if self._engine is not None:
            raise RuntimeError('create_database() called when a database engine already exists')

        self._get_engine(db_name, expect_exists=False)

        M5s = self._create_M5_table(M5_grid)
        Tinits = self._create_Tinit_table(Tinit_grid)
        Fs = self._create_F_table(F_grid)
        fs = self._create_f_table(f_grid)

        # tensor together all grids to produce a total work list
        work = product(M5s, Tinits, Fs, fs)

        # filter out values that are not value (for whatever reason)
        work_filtered = [x for x in work if is_valid(x)]

        self._create_work_list(work_filtered)
        self._data_table.create(self._engine)


    def _create_M5_table(self, M5_grid: List[float]) -> List[Tuple[int, float]]:
        self._M5_table.create(self._engine)

        # generate serial numbers for M5 sample grid and write these out
        serial_map = []

        with self._engine.begin() as conn:
            for serial, value in enumerate(M5_grid):
                serial_map.append((serial, value))

                params = lkit.RS5D.Parameters(value)
                conn.execute(
                    self._M5_table.insert().values(
                        serial=serial,
                        value_GeV=value,
                        Tcrossover_GeV=params.T_crossover,
                        Tcrossover_Kelvin=params.T_crossover_Kelvin
                    )
                )

            conn.commit()

        return serial_map


    def _create_Tinit_table(self, Tinit_grid: List[float]) -> List[Tuple[int, float]]:
        self._Tinit_table.create(self._engine)

        # generate serial numbers for Tinit sample grid and write these out
        serial_map = []

        with self._engine.begin() as conn:
            for serial, value in enumerate(Tinit_grid):
                serial_map.append((serial, value))

                conn.execute(
                    self._Tinit_table.insert().values(
                        serial=serial,
                        value_GeV=value,
                        value_Kelvin=value / lkit.Kelvin
                    )
                )

            conn.commit()

        return serial_map


    def _create_F_table(self, F_grid: List[float]) -> List[Tuple[int, float]]:
        self._F_table.create(self._engine)

        # generate serial number for F sample grid and write these out
        serial_map = []

        with self._engine.begin() as conn:
            for serial, value in enumerate(F_grid):
                serial_map.append((serial, value))

                conn.execute(
                    self._F_table.insert().values(
                        serial=serial,
                        value=value
                    )
                )

            conn.commit()

        return serial_map


    def _create_f_table(self, F_grid: List[float]) -> List[Tuple[int, float]]:
        self._f_table.create(self._engine)

        # generate serial number for F sample grid and write these out
        serial_map = []

        with self._engine.begin() as conn:
            for serial, value in enumerate(F_grid):
                serial_map.append((serial, value))

                conn.execute(
                    self._f_table.insert().values(
                        serial=serial,
                        value=value
                    )
                )

            conn.commit()

        return serial_map


    WorkElementType = Tuple[int, float]
    WorkGridItemType = Tuple[WorkElementType, WorkElementType, WorkElementType, WorkElementType]
    WorkGridType = List[WorkGridItemType]
    def _create_work_list(self, work_grid: WorkGridType) -> None:
        self._work_table.create(self._engine)

        # write work items into work table
        with self._engine.begin() as conn:
            for serial, (M5_data, Tinit_data, F_data, f_data) in enumerate(work_grid):
                M5_serial, M5 = M5_data
                Tinit_serial, Tinit = Tinit_data
                F_serial, F = F_data
                f_serial, f = f_data

                conn.execute(
                    self._work_table.insert().values(
                        serial=serial,
                        M5_serial=M5_serial,
                        Tinit_serial=Tinit_serial,
                        accrete_F_serial=F_serial,
                        collapse_f_serial=f_serial
                    )
                )

            conn.commit()


    def open_database(self, db_name: str) -> None:
        if self._engine is not None:
            raise RuntimeError('open_database() called when a database engine already exists')

        self._get_engine(db_name, expect_exists=True)


    def get_total_work_size(self) -> int:
        if self._engine is None:
            raise RuntimeError('No database connection exists in get_total_work_size()')

        with self._engine.begin() as conn:
            result = conn.execute(
                sqla.select(sqla.func.count()).select_from(self._work_table)
            )

            return result.scalar()


    def get_work_list(self) -> Dataset:
        if self._engine is None:
            raise RuntimeError('No database connection exists in get_work_list()')

        with self._engine.begin() as conn:
            result = conn.execute(
                sqla.select(self._work_table.c.serial) \
                    .join(self._data_table, self._work_table.c.serial == self._data_table.c.serial, isouter=True) \
                    .filter(self._data_table.c.serial == None)
            )

            return ray.data.from_items([x[0] for x in result.fetchall()])


    def get_work_item(self, serial: int) -> Tuple[float, float, float, float]:
        if self._engine is None:
            raise RuntimeError("No database connection in get_work_item()")

        with self._engine.begin() as conn:
            result = conn.execute(
                sqla.select(self._work_table.c.serial,
                            self._M5_table.c.value_GeV,
                            self._Tinit_table.c.value_GeV,
                            self._F_table.c.value,
                            self._f_table.c.value) \
                    .select_from(self._work_table) \
                    .filter(self._work_table.c.serial == serial) \
                    .join(self._M5_table, self._M5_table.c.serial == self._work_table.c.M5_serial) \
                    .join(self._Tinit_table, self._Tinit_table.c.serial == self._work_table.c.Tinit_serial) \
                    .join(self._F_table, self._F_table.c.serial == self._work_table.c.accrete_F_serial) \
                    .join(self._f_table, self._f_table.c.serial == self._work_table.c.collapse_f_serial)
            )

            return result.first()


    def write_work_item(self, data) -> None:
        if self._engine is None:
            raise RuntimeError('No database connection in write_work_item()')

        # insert this batch of outputs, using .on_conflict_do_nothing() so that multiple entries are not
        # written into the database
        # (e.g. if a ray processes a batch item multiple times, perhaps due to a worker failure)
        with self._engine.begin() as conn:
            conn.execute(
                insert(self._data_table).on_conflict_do_nothing(), data
            )

            conn.commit()
