# spin weights for contribution to energy density; fermions contribute like 7/8 of a particle
boson_weight = 1.0
fermion_weight = 7.0/8.0

# grey-body fitting coefficients, taken from Table II of Friedlander et al., http://arxiv.org/abs/2201.11761v1
xi0_spin0_4D = 0.00187
b_spin0_4D = 0.395
c_spin0_4D = 1.186

xi0_spin0pt5_4D = 0.00103
b_spin0pt5_4D = 0.337
c_spin0pt5_4D = 1.221

xi0_spin1_4D = 0.000423
b_spin1_4D = 0.276
c_spin1_4D = 1.264

xi0_spin2_4D = 0.0000966

xi0_spin2_5D = 0.00972

# table of primary Hawking quanta and masses, adapted from BlackHawk manual (Appendix C, Table 3)
# https://blackhawk.hepforge.org/manuals/blackhawk1.1.pdf
SM_particle_table = \
 {'Higgs': {'mass': 1.2503E2, 'dof': 1, 'spin-weight': boson_weight, 'xi0': xi0_spin0_4D, 'b': b_spin0_4D, 'c': c_spin0_4D, 'xi-per-dof': True},
  'photon': {'mass': 0.0, 'dof': 2, 'spin-weight': boson_weight, 'xi0': xi0_spin1_4D, 'xi-per-dof': True},
  'gluon': {'mass': 0.2, 'dof': 16, 'spin-weight': boson_weight, 'xi0': xi0_spin1_4D, 'b': b_spin1_4D, 'c': c_spin1_4D, 'xi-per-dof': True},
  'W boson': {'mass': 8.0403E1, 'dof': 6, 'spin-weight': boson_weight, 'xi0': xi0_spin1_4D, 'b': b_spin1_4D, 'c': c_spin1_4D, 'xi-per-dof': True},
  'Z boson': {'mass': 9.11876E1, 'dof': 3, 'spin-weight': boson_weight, 'xi0': xi0_spin1_4D, 'b': b_spin1_4D, 'c': c_spin1_4D, 'xi-per-dof': True},
  'neutrino': {'mass': 0.0, 'dof': 6, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'xi-per-dof': True},
  'electron': {'mass': 5.109989461E-4, 'dof': 4, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'muon': {'mass': 1.056583745E-1, 'dof': 4, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'tau': {'mass': 1.77686, 'dof': 4, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'up quark': {'mass': 2.2E-3, 'dof': 12, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'down quark': {'mass': 4.7E-3, 'dof': 12, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'strange quark': {'mass': 9.6E-2, 'dof': 12, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'charm quark': {'mass': 1.27, 'dof': 12, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'bottom quark': {'mass': 4.18, 'dof': 12, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True},
  'top quark': {'mass': 1.7321E2, 'dof': 12, 'spin-weight': fermion_weight, 'xi0': xi0_spin0pt5_4D, 'b': b_spin0pt5_4D, 'c': c_spin0pt5_4D, 'xi-per-dof': True}}

# NOTE Friedlander et al. report xi values *per dof* except for gravitons, for which all dofs are
# already included in the xi0 value

Standard4D_graviton_table = \
 {'4D graviton': {'mass': 0.0, 'dof': 2.0, 'xi0': xi0_spin2_4D, 'xi-per-dof': False}}

RS_bulk_particle_table = \
 {'5D graviton': {'mass': 0.0, 'dof': 5.0, 'xi0': xi0_spin2_5D, 'xi-per-dof': False}}
