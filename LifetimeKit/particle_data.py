# table of primary Hawking quanta and masses, taken from BlackHawk manual (Appendix C, Table 3)
# https://blackhawk.hepforge.org/manuals/blackhawk1.1.pdf

# spin weights for contribution to energy density; fermions contribute like 7/8 of a particle
boson_weight = 1.0
fermion_weight = 7.0/8.0

SM_particle_table = \
    {'Higgs': {'mass': 1.2503E2, 'dof': 1, 'spin-weight': boson_weight},
     'photon': {'mass': 0.0, 'dof': 2, 'spin-weight': boson_weight},
     'gluon': {'mass': 0.2, 'dof': 16, 'spin-weight': boson_weight},
     'W boson': {'mass': 8.0403E1, 'dof': 6, 'spin-weight': boson_weight},
     'Z boson': {'mass': 9.11876E1, 'dof': 3, 'spin-weight': boson_weight},
     'neutrino': {'mass': 0.0, 'dof': 6, 'spin-weight': fermion_weight},
     'electron': {'mass': 5.109989461E-4, 'dof': 4, 'spin-weight': fermion_weight},
     'muon': {'mass': 1.056583745E-1, 'dof': 4, 'spin-weight': fermion_weight},
     'tau': {'mass': 1.77686, 'dof': 4, 'spin-weight': fermion_weight},
     'up quark': {'mass': 2.2E-3, 'dof': 12, 'spin-weight': fermion_weight},
     'down quark': {'mass': 4.7E-3, 'dof': 12, 'spin-weight': fermion_weight},
     'charm quark': {'mass': 1.27, 'dof': 12, 'spin-weight': fermion_weight},
     'strange quark': {'mass': 9.6E-2, 'dof': 12, 'spin-weight': fermion_weight},
     'top quark': {'mass': 1.7321E2, 'dof': 12, 'spin-weight': fermion_weight},
     'bottom quark': {'mass': 4.18, 'dof': 12, 'spin-weight': fermion_weight}}

RS_bulk_particle_table = \
    {'5D graviton': {'mass': 0.0, 'dof': 5}}
