# spin weights for contribution to energy density; fermions contribute like 7/8 of a particle
boson_weight = 1.0
fermion_weight = 7.0/8.0

# grey-body fitting coefficients, taken from Table II of Friedlander et al., http://arxiv.org/abs/2201.11761v1

# SPIN-0 FITTING FUNCTIONS
xi0_spin0_4D = 0.00187
b_spin0_4D = 0.395
c_spin0_4D = 1.186

xi0_spin0_5D = 0.0167
b_spin0_5D = 0.333
c_spin0_5D = 1.236

# SPIN-1/2 FITTING FUNCTIONS
xi0_spin0pt5_4D = 0.00103
b_spin0pt5_4D = 0.337
c_spin0pt5_4D = 1.221

xi0_spin0pt5_5D = 0.0146
b_spin0pt5_5D = 0.276
c_spin0pt5_5D = 1.297

# SPIN-1 FITTING FUNCTIONS
xi0_spin1_4D = 0.000423
b_spin1_4D = 0.276
c_spin1_4D = 1.264

xi0_spin1_5D = 0.0115
b_spin1_5D = 0.220
c_spin1_5D = 1.361

# SPIN-2 FITTING FUNCTIONS
xi0_spin2_4D = 0.0000966

xi0_spin2_5D = 0.00972

# table of primary Hawking quanta and masses, adapted from BlackHawk manual (Appendix C, Table 3)
# https://blackhawk.hepforge.org/manuals/blackhawk1.1.pdf

def _table_merge(A, B):
    newA = A.copy()

    for key in B:
        if key in newA:
            newA[key] |= B[key]
        else:
            newA[key] = B[key]

    return newA

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

# 4D emission rates
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

Friedlander_greybody_table_4D = _table_merge(SM_particle_base_table,
 {'Higgs': {'xi0': xi0_spin0_4D, 'b': b_spin0_4D, 'c': c_spin0_4D, 'xi-per-dof': True},
  'photon': {'xi0': xi0_spin1_4D, 'xi-per-dof': True},
  'gluon': {'xi0': xi0_spin1_4D, 'b': b_spin1_4D, 'c': c_spin1_4D, 'xi-per-dof': True},
  'W boson': {'xi0': xi0_spin1_4D, 'b': b_spin1_4D, 'c': c_spin1_4D, 'xi-per-dof': True},
  'Z boson': {'xi0': xi0_spin1_4D, 'b': b_spin1_4D, 'c': c_spin1_4D, 'xi-per-dof': True},
  'neutrino': {'xi0': xi0_spin0pt5_4D, 'xi-per-dof': True},
  'electron': {'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'muon': {'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'tau': {'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'up quark': {'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'down quark': {'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'strange quark': {'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'charm quark': {'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'bottom quark': {'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'top quark': {'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True}}
)

# 5D emission rates
Friedlander_greybody_table_5D = _table_merge(SM_particle_base_table,
 {'Higgs': {'xi0': xi0_spin0_5D, 'b': b_spin0_5D, 'c': c_spin0_5D, 'xi-per-dof': True},
  'photon': {'xi0': xi0_spin1_5D, 'xi-per-dof': True},
  'gluon': {'xi0': xi0_spin1_5D, 'b': b_spin1_5D, 'c': c_spin1_5D, 'xi-per-dof': True},
  'W boson': {'xi0': xi0_spin1_5D, 'b': b_spin1_5D, 'c': c_spin1_5D, 'xi-per-dof': True},
  'Z boson': {'xi0': xi0_spin1_5D, 'b': b_spin1_5D, 'c': c_spin1_5D, 'xi-per-dof': True},
  'neutrino': {'xi0': xi0_spin0pt5_5D, 'xi-per-dof': True},
  'electron': {'xi0': xi0_spin0pt5_5D, 'b': b_spin0pt5_5D, 'c': c_spin0pt5_5D, 'xi-per-dof': True},
  'muon': {'xi0': xi0_spin0pt5_5D, 'b': b_spin0pt5_5D, 'c': c_spin0pt5_5D, 'xi-per-dof': True},
  'tau': {'xi0': xi0_spin0pt5_5D, 'b': b_spin0pt5_5D, 'c': c_spin0pt5_5D, 'xi-per-dof': True},
  'up quark': {'xi0': xi0_spin0pt5_5D, 'b': b_spin0pt5_5D, 'c': c_spin0pt5_5D, 'xi-per-dof': True},
  'down quark': {'xi0': xi0_spin0pt5_5D, 'b': b_spin0pt5_5D, 'c': c_spin0pt5_5D, 'xi-per-dof': True},
  'strange quark': {'xi0': xi0_spin0pt5_5D, 'b': b_spin0pt5_5D, 'c': c_spin0pt5_5D, 'xi-per-dof': True},
  'charm quark': {'xi0': xi0_spin0pt5_5D, 'b': b_spin0pt5_5D, 'c': c_spin0pt5_5D, 'xi-per-dof': True},
  'bottom quark': {'xi0': xi0_spin0pt5_5D, 'b': b_spin0pt5_5D, 'c': c_spin0pt5_5D, 'xi-per-dof': True},
  'top quark': {'xi0': xi0_spin0pt5_5D, 'b': b_spin0pt5_5D, 'c': c_spin0pt5_5D, 'xi-per-dof': True}}
)

# NOTE Friedlander et al. report xi values *per dof* except for gravitons, for which all dofs are
# already included in the xi0 value

Standard4D_graviton_particle_table = \
 {'4D graviton': {'mass': 0.0, 'dof': 2.0}}

Standard4D_graviton_greybody_table = \
 {'4D graviton': {'xi0': xi0_spin2_4D, 'xi-per-dof': False}}

RS_graviton_particle_table = \
 {'5D graviton': {'mass': 0.0, 'dof': 5.0}}

RS_graviton_greybody_table = \
 {'5D graviton': {'xi0': xi0_spin2_5D, 'xi-per-dof': False}}
