from __future__ import print_function

import argparse
import os
import sys
from ase.io import read
import numpy as np

from sgdml import __version__
from sgdml.utils import io, ui

if sys.version[0] == '3':
    raw_input = input


def from_traj(dataset, overwrite=True, custom_name='', theory='unknown', r_unit='Ang', e_unit='eV'):
    # 'Please provide a description of the length unit used in your input file, e.g. \'Ang\' or \'au\': ')
    # 'Note: This string will be stored in the dataset file and passed on to models files for later reference.')
    # 'Please provide a description of the energy unit used in your input file, e.g. \'kcal/mol\' or \'eV\': ')
    # 'Note: This string will be stored in the dataset file and passed on to models files for later reference.')
    name = os.path.splitext(os.path.basename(dataset))[0]
    dataset_file_name = name + '.npz'

    dataset_exists = os.path.isfile(dataset_file_name)
    if dataset_exists and overwrite:
        print(ui.color_str('[INFO]', bold=True) + ' Overwriting existing dataset file.')
    if not dataset_exists or overwrite:
        print('Writing dataset to \'{}\'...'.format(dataset_file_name))
    else:
        sys.exit(
            ui.color_str('[FAIL]', fore_color=ui.RED, bold=True)
            + ' Dataset \'{}\' already exists.'.format(dataset_file_name)
        )

    mols = read(dataset, index=':')

    lattice, R, z, E, F = None, None, None, None, None

    calc = mols[0].get_calculator()

    print("\rNumber geometries found: {:,}\n".format(len(mols)))

    if 'forces' not in calc.results:
        sys.exit(
            ui.color_str('[FAIL]', fore_color=ui.RED, bold=True)
            + ' Forces are missing in the input file!'
        )

    lattice = np.array(mols[0].get_cell())
    if not np.any(lattice):
        print(
            ui.color_str('[INFO]', bold=True)
            + ' No lattice vectors specified in extended XYZ file.'
        )

    Z = np.array([mol.get_atomic_numbers() for mol in mols])
    all_z_the_same = (Z == Z[0]).all()
    if not all_z_the_same:
        sys.exit(
            ui.color_str('[FAIL]', fore_color=ui.RED, bold=True)
            + ' Order of atoms changes accross dataset.'
        )

    lattice = np.array(mols[0].get_cell())
    if not np.any(lattice):  # all zeros
        lattice = None

    R = np.array([mol.get_positions() for mol in mols])
    z = Z[0]

    E = np.array([mol.get_potential_energy() for mol in mols])
    F = np.array([mol.get_forces() for mol in mols])

    if custom_name != '':
        name = custom_name

    # Base variables contained in every model file.
    base_vars = {'type': 'd', 'code_version': __version__, 'name': name, 'theory': theory, 'R': R, 'z': z, 'F': F,
                 'F_min': np.min(F.ravel()), 'F_max': np.max(F.ravel()), 'F_mean': np.mean(F.ravel()),
                 'F_var': np.var(F.ravel())}

    if r_unit != '':
        base_vars['r_unit'] = r_unit

    if e_unit != '':
        base_vars['e_unit'] = e_unit

    if E is not None:
        base_vars['E'] = E
        base_vars['E_min'], base_vars['E_max'] = np.min(E), np.max(E)
        base_vars['E_mean'], base_vars['E_var'] = np.mean(E), np.var(E)
    else:
        print(ui.color_str('[INFO]', bold=True) + ' No energy labels found in dataset.')

    if lattice is not None:
        base_vars['lattice'] = lattice

    base_vars['md5'] = io.dataset_md5(base_vars)
    np.savez_compressed(dataset_file_name, **base_vars)
    print(ui.color_str('[DONE]', fore_color=ui.GREEN, bold=True))
    return np.load(dataset_file_name)
