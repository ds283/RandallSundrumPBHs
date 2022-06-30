from functools import partial
from pathlib import Path
from itertools import product
from typing import List, Callable, Dict, Tuple
from datetime import datetime


import ray
from ray.data import Dataset
import sqlalchemy as sqla
from sqlalchemy.dialects.sqlite import insert

import LifetimeKit as lkit

WorkElementType = Tuple[int, float]
WorkGridItemType = Tuple[WorkElementType, WorkElementType, WorkElementType, WorkElementType, WorkElementType]

BatchItemType = Tuple[int, WorkGridItemType]
BatchListType = List[BatchItemType]

GIT_HASH_LENGTH = 40

def _is_valid_batch(validator, batch: List[WorkGridItemType]) -> List[Tuple[bool, WorkGridItemType]]:
    batch_out = []

    for data in batch:
        result = validator(data)
        batch_out.append((result, data))

    return batch_out


@ray.remote
class Cache:
    def __init__(self, histories: List[str], label_maker: Callable[[str], Dict[str, str]],
                 git_hash: str, standard_columns: List[str]=None):
        self._engine = None
        self._metadata: sqla.MetaData = None

        self._histories: List[str] = histories
        self._label_maker: Callable[[str], Dict[str, str]] = label_maker

        self._standard_columns = standard_columns

        self._git_hash = git_hash
        self._version_serial = None

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

        self._version_table = sqla.Table('versions', self._metadata,
                                         sqla.Column('serial', sqla.Integer, primary_key=True),
                                         sqla.Column('git_hash', sqla.String(GIT_HASH_LENGTH)))

        self._M5_table = sqla.Table('M5_grid', self._metadata,
                                    sqla.Column('serial', sqla.Integer, primary_key=True),
                                    sqla.Column('value_GeV', sqla.Float(precision=64)),
                                    sqla.Column('Tcrossover_GeV', sqla.Float(precision=64)),
                                    sqla.Column('Tcrossover_Kelvin', sqla.Float(precision=64)))

        self._Tinit_table = sqla.Table('Tinit_grid', self._metadata,
                                       sqla.Column('serial', sqla.Integer, primary_key=True),
                                       sqla.Column('value_GeV', sqla.Float(precision=64)),
                                       sqla.Column('value_Kelvin', sqla.Float(precision=64)))

        self._Tfinal_table = sqla.Table('Tfinal_grid', self._metadata,
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
                                      sqla.Column('Tfinal_serial', sqla.Integer, sqla.ForeignKey('Tfinal_grid.serial')),
                                      sqla.Column('accrete_F_serial', sqla.Integer, sqla.ForeignKey('accrete_F_grid.serial')),
                                      sqla.Column('collapse_f_serial', sqla.Integer, sqla.ForeignKey('collapse_f_grid.serial')))

        self._data_table = sqla.Table('data', self._metadata,
                                      sqla.Column('serial', sqla.Integer, sqla.ForeignKey('work_grid.serial'), primary_key=True),
                                      sqla.Column('version', sqla.Integer, sqla.ForeignKey('versions.serial')),
                                      sqla.Column('timestamp', sqla.DateTime()))

        if self._standard_columns is not None:
            for clabel in self._standard_columns:
                self._data_table.append_column(sqla.Column(clabel, sqla.Float(precision=64)))

        # append column labels to data table for each data item stored, for each history type
        for label in self._histories:
            column_labels = self._label_maker(label).values()
            for clabel in column_labels:
                self._data_table.append_column(sqla.Column(clabel, sqla.Float(precision=64)))


    def create_database(self, db_name: str, M5_grid: List[float], Tinit_grid: List[float], Tfinal_grid: List[float],
                        F_grid: List[float], f_grid: List[float], validator) -> None:
        """
        Create a cache database with the specified grids in M5, Tinit, and F
        :param db_name:
        :param M5_grid:
        :param Tinit_grid:
        :param Tfinal_grid:
        :param F_grid:
        :param f_grid:
        :return:
        """
        if self._engine is not None:
            raise RuntimeError('create_database() called when a database engine already exists')

        self._get_engine(db_name, expect_exists=False)

        print('-- creating database tables')

        self._version_table.create(self._engine)
        self._version_serial = self._get_version_serial()

        M5s = self._create_M5_table(M5_grid)
        Tinits = self._create_Tinit_table(Tinit_grid)
        Tfinals = self._create_Tfinal_table(Tfinal_grid)
        Fs = self._create_F_table(F_grid)
        fs = self._create_f_table(f_grid)

        M5s_size = len(M5s)
        Tinits_size = len(Tinits)
        Tfinals_size = len(Tfinals)
        Fs_size = len(Fs)
        fs_size = len(fs)

        # tensor together all grids to produce a total work list
        print('-- building work list')
        print('   * M5 grid size = {sz}'.format(sz=M5s_size))
        print('   * Tinit grid size = {sz}'.format(sz=Tinits_size))
        print('   * Tfinal grid size = {sz}'.format(sz=Tfinals_size))
        print('   * F grid size = {sz}'.format(sz=Fs_size))
        print('   * f grid size = {sz}'.format(sz=fs_size))
        print('   # TOTAL RAW WORK TABLE SIZE = {sz}'.format(sz=M5s_size*Tinits_size*Tfinals_size*Fs_size*fs_size))
        work = product(M5s, Tinits, Tfinals, Fs, fs)

        print('-- filtering work list')
        work_filtered = ray.data.from_items(list(work)).map_batches(partial(_is_valid_batch, validator))

        print('-- writing work list to database')
        count = self._write_work_table(work_filtered)

        print('## wrote {sz} work items'.format(sz=count))

    def _get_version_serial(self):
        if self._engine is None:
            raise RuntimeError('No database connection exists in _get_version_serial()')

        with self._engine.begin() as conn:
            result = conn.execute(
                sqla.select(self._version_table.c.serial).filter(self._version_table.c.git_hash == self._git_hash)
            )

            x = result.first()

            if x is not None:
                return x

            serial = conn.execute(sqla.select(sqla.func.count()).select_from(self._version_table)).scalar()
            conn.execute(sqla.insert(self._version_table), {'serial': serial, 'git_hash': self._git_hash})

            conn.commit()

        return serial

    def _write_work_table(self, work_filtered):
        self._work_table.create(self._engine)
        self._data_table.create(self._engine)

        count = 0

        with self._engine.begin() as conn:
            for flag, data in work_filtered.iter_rows():
                if not flag:
                    continue

                M5_data, Tinit_data, Tfinal_data, F_data, f_data = data

                M5_serial, M5 = M5_data
                Tinit_serial, Tinit = Tinit_data
                Tfinal_serial, Tfinal = Tfinal_data
                F_serial, F = F_data
                f_serial, f = f_data

                conn.execute(
                    self._work_table.insert().values(
                        serial=count,
                        M5_serial=M5_serial,
                        Tinit_serial=Tinit_serial,
                        Tfinal_serial=Tfinal_serial,
                        accrete_F_serial=F_serial,
                        collapse_f_serial=f_serial
                    )
                )

                count += 1

            conn.commit()

        return count

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

    def _create_Tfinal_table(self, Tfinal_grid: List[float]) -> List[Tuple[int, float]]:
        self._Tfinal_table.create(self._engine)

        # generate serial numbers for Tfinal sample grid and write these out
        serial_map = []

        with self._engine.begin() as conn:
            for serial, value in enumerate(Tfinal_grid):
                serial_map.append((serial, value))

                conn.execute(
                    self._Tfinal_table.insert().values(
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

    def open_database(self, db_name: str) -> None:
        if self._engine is not None:
            raise RuntimeError('open_database() called when a database engine already exists')

        self._get_engine(db_name, expect_exists=True)
        self._version_serial = self._get_version_serial()

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

    def get_work_item(self, serial: int) -> Tuple[float, float, float, float, float]:
        if self._engine is None:
            raise RuntimeError("No database connection in get_work_item()")

        with self._engine.begin() as conn:
            result = conn.execute(
                sqla.select(self._work_table.c.serial,
                            self._M5_table.c.value_GeV,
                            self._Tinit_table.c.value_GeV,
                            self._Tfinal_table.c.value_GeV,
                            self._F_table.c.value,
                            self._f_table.c.value) \
                    .select_from(self._work_table) \
                    .filter(self._work_table.c.serial == serial) \
                    .join(self._M5_table, self._M5_table.c.serial == self._work_table.c.M5_serial) \
                    .join(self._Tinit_table, self._Tinit_table.c.serial == self._work_table.c.Tinit_serial) \
                    .join(self._Tfinal_table, self._Tfinal_table.c.serial == self._work_table.c.Tfinal_serial) \
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

        now = datetime.now()
        versioned_data = [x | {'version': self._version_serial, 'timestamp': now} for x in data]
        with self._engine.begin() as conn:
            conn.execute(
                insert(self._data_table).on_conflict_do_nothing(), versioned_data
            )

            conn.commit()
