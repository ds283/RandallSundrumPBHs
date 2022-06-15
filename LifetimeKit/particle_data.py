import copy

# spin weights for contribution to energy density; fermions contribute like 7/8 of a particle
boson_weight = 1.0
fermion_weight = 7.0/8.0

def _table_merge(A, B):
    """
    Merge two particle data tables together to produce a third table
    :param A: table 1
    :param B: table 2
    :return: merged table
    """
    newA = copy.deepcopy(A)

    for key in B:
        if key in newA:
            newA[key] |= B[key]
        else:
            newA[key] = B[key]

    return newA

# common data elements should be shared to prevent copying/propagation errors
# adapted from BlackHawk manual (Appendix C, Table 3)
# https://blackhawk.hepforge.org/manuals/blackhawk1.1.pdf
SM_particle_base_table = \
 {'Higgs': {'mass': 1.2503E2, 'dof': 1},
  'photon': {'mass': 0.0, 'dof': 2},
  'gluon': {'mass': 0.2, 'dof': 16},
  'W boson': {'mass': 8.0403E1, 'dof': 6},
  'Z boson': {'mass': 9.11876E1, 'dof': 3},
  'neutrino': {'mass': 0.0, 'dof': 6},
  'electron': {'mass': 5.109989461E-4, 'dof': 4},
  'muon': {'mass': 1.056583745E-1, 'dof': 4},
  'tau': {'mass': 1.77686, 'dof': 4},
  'up quark': {'mass': 2.2E-3, 'dof': 12},
  'down quark': {'mass': 4.7E-3, 'dof': 12},
  'strange quark': {'mass': 9.6E-2, 'dof': 12},
  'charm quark': {'mass': 1.27, 'dof': 12},
  'bottom quark': {'mass': 4.18, 'dof': 12},
  'top quark': {'mass': 1.7321E2, 'dof': 12}}

# 4D statistical weights for thermal history
SM_particle_table = _table_merge(SM_particle_base_table,
 {'Higgs': {'spin-weight': boson_weight},
  'photon': {'spin-weight': boson_weight},
  'gluon': {'spin-weight': boson_weight},
  'W boson': {'spin-weight': boson_weight},
  'Z boson': {'spin-weight': boson_weight},
  'neutrino': {'spin-weight': fermion_weight},
  'electron': {'spin-weight': fermion_weight},
  'muon': {'spin-weight': fermion_weight},
  'tau': {'spin-weight': fermion_weight},
  'up quark': {'spin-weight': fermion_weight},
  'down quark': {'spin-weight': fermion_weight},
  'strange quark': {'spin-weight': fermion_weight},
  'charm quark': {'spin-weight': fermion_weight},
  'bottom quark': {'spin-weight': fermion_weight},
  'top quark': {'spin-weight': fermion_weight}}
)

# degrees of freedom for 4D spin-2 states
Standard4D_graviton_particle_table = \
 {'4D graviton': {'mass': 0.0, 'dof': 2.0}}

# degrees of freedom for *on brane* spin-2 states in Randall-Sundrum
RS_graviton_particle_table = \
 {'5D graviton': {'mass': 0.0, 'dof': 5.0}}
