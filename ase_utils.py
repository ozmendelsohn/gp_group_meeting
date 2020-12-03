from ase.lattice.cubic import FaceCenteredCubic
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase import io
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.build import molecule
from ase.visualize import ngl
from ase.io.trajectory import Trajectory
from ase import Atom, Atoms
from scipy.spatial.distance import pdist
from matplotlib import use
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize
from numpy.linalg import cholesky, det, lstsq
from numpy.linalg import inv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
import matplotlib
from matplotlib import cm  # Colormaps
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ipywidgets as widgets
from ipywidgets import FloatSlider
from ase.calculators.calculator import Calculator
from ase.units import kcal, mol
from sgdml.cli import _print_dataset_properties, _print_model_properties, _print_task_properties
from data_utils import from_traj
from sgdml.predict import GDMLPredict
from matplotlib.lines import Line2D
import scipy as sp
from sgdml.utils import io
import ase
import time


def movingaverage(interval, theta=0.01):
    y_avg = np.mean(interval)
    y = np.zeros(len(interval))
    for i, yi in enumerate(interval):
        y[i] = yi * theta + y_avg * (1 - theta)
        y_avg = y[i]
    return y

def plot_gpr(gpr, x, y, x_test, y_test):
    y_hat_train = gpr.predict(x)
    y_hat_train_norm = np.linalg.norm(y_hat_train, axis=1)
    y_train_norm = np.linalg.norm(y, axis=1)
    y_hat = gpr.predict(x_test)
    y_hat_norm = np.linalg.norm(y_hat, axis=1)
    y_test_norm = np.linalg.norm(y_test, axis=1)

    col = ['C1', 'C2', 'C3', 'C4']
    plt.close('all')
    fig, axis = plt.subplots(ncols=1, nrows=2, sharex=True)
    x1 = np.arange(len(y_hat_train_norm))
    axis[0].plot(x1, y_hat_train_norm, '.', alpha=0.1, color=col[0])
    axis[0].plot(x1, y_train_norm, '.', alpha=0.1, color=col[1])
    axis[0].plot(x1, movingaverage(y_hat_train_norm), color=col[0])
    axis[0].plot(x1, movingaverage(y_train_norm), color=col[1])
    x2 = np.arange(len(y_hat_norm)) + len(y_hat_train_norm)
    axis[0].plot(x2, y_hat_norm, '.', alpha=0.1, color=col[2])
    axis[0].plot(x2, y_test_norm, '.', alpha=0.1, color=col[3])
    axis[0].plot(x2, movingaverage(y_hat_norm), color=col[2])
    axis[0].plot(x2, movingaverage(y_test_norm), color=col[3])

    axis[0].set_xticks([])
    axis[0].set_title('The force norm')
    custom_lines = [Line2D([0], [0], color=col[1], lw=4),
                    Line2D([0], [0], color=col[0], lw=4),
                    Line2D([0], [0], color=col[2], lw=4),
                    Line2D([0], [0], color=col[3], lw=4)]
    axis[0].legend(custom_lines, ['training', 'predicted on train', 'predicted', 'ground truth'])

    x1 = np.arange(len(y_hat_train_norm))
    axis[1].plot(x1, y_hat_train_norm - y_train_norm, '.', alpha=0.1, color=col[0])
    axis[1].plot(x1, movingaverage(y_hat_train_norm - y_train_norm), color=col[0])
    x2 = np.arange(len(y_hat_norm)) + len(y_hat_train_norm)
    axis[1].plot(x2, y_hat_norm - y_test_norm, '.', alpha=0.1, color=col[1])
    axis[1].plot(x2, movingaverage(y_hat_norm - y_test_norm), color=col[1])
    custom_lines = [Line2D([0], [0], color=col[0], lw=4),
                    Line2D([0], [0], color=col[1], lw=4)]
    axis[1].legend(custom_lines, ['difference on train', 'difference on test'])
    plt.show()


def printenergy(atoms):  # store a reference to atoms in the definition.
    t0 = time.time()

    def ptint_func(a=atoms):
        """Function to print the potential, kinetic and total energy."""
        epot = a.get_potential_energy() / len(a)
        ekin = a.get_kinetic_energy() / len(a)
        print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
              'Etot = %.3feV, time elapsed: %.1f' % (
                  epot, ekin, ekin / (1.5 * units.kB), epot + ekin, time.time() - t0))

    return ptint_func


def md_dataset_split(traj, start=1, every=1):
    y, x = [], []
    y_test, x_test = [], []
    for i, config in enumerate(traj):
        if i % every == 0 and i >= start - 1:
            if i <= (len(traj) + start) / 2:
                y.append(config.get_forces().reshape([1, -1]))
                x.append(config.get_positions().reshape([1, -1]))
            else:
                x_test.append(config.get_positions().reshape([1, -1]))
                y_test.append(config.get_forces().reshape([1, -1]))

    x, y = np.concatenate(x, axis=0), np.concatenate(y, axis=0)
    x_test, y_test = np.concatenate(x_test, axis=0), np.concatenate(y_test, axis=0)
    return (x, y), (x_test, y_test)


from ase.calculators.calculator import Calculator


class GPRCalculator(Calculator):
    implemented_properties = ['forces']

    def __init__(self, gdr, *args, **kwargs):
        super(GPRCalculator, self).__init__(*args, **kwargs)
        self.gpr_model = gdr

    def calculate(self, atoms=None, *args, **kwargs):
        super(GPRCalculator, self).calculate(atoms, *args, **kwargs)
        r = np.array(atoms.get_positions())
        x = r.reshape([1, -1])
        f = self.gpr_model.predict(x)
        self.results = {'forces': f.reshape(-1, 3)}


def pbc_d(diffs, lat):
    """
    Clamp differences of vectors to super cell.

    Parameters
    ----------
        diffs : :obj:`numpy.ndarray`
            N x 3 matrix of N pairwise differences between vectors `u - v`
        lat_and_inv : tuple of :obj:`numpy.ndarray`
            Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.
        use_torch : boolean, optional
            Enable, if the inputs are PyTorch objects.

    Returns
    -------
        :obj:`numpy.ndarray`
            N x 3 matrix clamped differences
    """
    lat_inv = np.linalg.inv(lat)

    c = lat_inv.dot(diffs.T)
    diffs -= lat.dot(np.rint(c)).T

    return diffs


def dist(r, lat=None): # TODO: update return (no squareform anymore)
    """
    Compute pairwise Euclidean distance matrix between all atoms.

    Parameters
    ----------
        r : :obj:`numpy.ndarray`
            Array of size 3N containing the Cartesian coordinates of
            each atom.
        lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
            Tuple of 3x3 matrix containing lattice vectors as columns and its inverse.

    Returns
    -------
        :obj:`numpy.ndarray`
            Array of size N x N containing all pairwise distances between atoms.
    """

    n_atoms = r.shape[0]

    if lat is None:
        pdist = sp.spatial.distance.pdist(r, 'euclidean')
    else:
        pdist = sp.spatial.distance.pdist(
            r, lambda u, v: np.linalg.norm(pbc_d(u - v, lat))
        )

    tril_idxs = np.tril_indices(n_atoms, k=-1)
    return sp.spatial.distance.squareform(pdist, checks=False)[tril_idxs]

def dist_mat(r, lat=None): # TODO: update return (no squareform anymore)
    """
    Compute pairwise Euclidean distance matrix between all atoms.

    Parameters
    ----------
        r : :obj:`numpy.ndarray`
            Array of size 3N containing the Cartesian coordinates of
            each atom.
        lat_and_inv : tuple of :obj:`numpy.ndarray`, optional
            Tuple of 3x3 matrix containing lattice vectors as columns and its inverse.

    Returns
    -------
        :obj:`numpy.ndarray`
            Array of size N x N containing all pairwise distances between atoms.
    """

    if lat is None:
        pdist = sp.spatial.distance.pdist(r, 'euclidean')
    else:
        pdist = sp.spatial.distance.pdist(
            r, lambda u, v: np.linalg.norm(pbc_d(u - v, lat))
        )

    return sp.spatial.distance.squareform(pdist, checks=False)


def x_to_d(x, lat=None):
    d = []
    for xi in x:
        xi = xi.reshape([-1, 3])
        d.append(dist(xi, lat))
    return np.stack(d, axis=0)


class iGPRCalculator(Calculator):
    implemented_properties = ['forces']

    def __init__(self, gdr, *args, **kwargs):
        super(iGPRCalculator, self).__init__(*args, **kwargs)
        self.gpr_model = gdr

    def calculate(self, atoms=None, *args, **kwargs):
        super(iGPRCalculator, self).calculate(atoms, *args, **kwargs)
        r = np.array(atoms.get_positions())
        x = r.reshape([1, -1])
        f = self.gpr_model.predict(x_to_d(x))
        self.results = {'forces': f.reshape(-1, 3)}


def create(dataset_path, n_train, n_valid,
           task_dir=None,
           valid_dataset_path=None,
           sigs=None,
           overwrite=True,
           use_torch=False,
           max_processes=None):
    execute_line = f'sgdml create {dataset_path} {n_train} {n_valid} '
    if task_dir is not None:
        execute_line += f'--task_dir {task_dir}'
    if sigs is not None:
        sig = '--sig ' + ' '.join([str(s) for s in sigs])
        execute_line += f'{sig} '
    if overwrite:
        execute_line += '-o '
    if use_torch:
        execute_line += '--torch '
    if max_processes is not None:
        execute_line += f'-p {max_processes}'
    return execute_line


def train(task_dir, valid_dataset_path,
          overwrite=True,
          use_torch=False,
          max_processes=None):
    execute_line = f'sgdml train {task_dir} {valid_dataset_path} '
    if overwrite:
        execute_line += '-o '
    if use_torch:
        execute_line += '--torch '
    if max_processes is not None:
        execute_line += f'-p {max_processes}'
    return execute_line


def all_script(dataset_path, n_train, n_valid, n_test=None,
               valid_dataset_path=None,
               train_dataset_path=None,
               sigs=None,
               overwrite=True,
               use_torch=False,
               use_cg=True,
               max_processes=None):
    execute_line = f'sgdml all {dataset_path} {n_train} {n_valid} '
    if n_test is not None:
        execute_line += f'{n_test} '
    if valid_dataset_path is not None:
        execute_line += f'-v {valid_dataset_path} '
    if train_dataset_path is not None:
        execute_line += f'-t {train_dataset_path} '
    if sigs is not None:
        sig = '--sig ' + ' '.join([str(s) for s in sigs])
        execute_line += f'{sig} '
    if overwrite:
        execute_line += '-o '
    if use_torch:
        execute_line += '--torch '
    if use_cg:
        execute_line += '--cg '
    if max_processes is not None:
        execute_line += f'-p {max_processes}'
    script_name = 'run_me.sh'
    with open(script_name, 'w') as f:
        f.write('#!/usr/bin/sh\n')
        f.write(execute_line)
    return script_name


def plot_gdml(model_path, traj, start=1, every=1):
    gdml = GDMLPredict(np.load(model_path, allow_pickle=True))
    (x, y), (x_test, y_test) = md_dataset_split(traj, start, every)
    print(x.shape)
    _, y_hat_train = gdml.predict(x)
    y_hat_train_norm = np.linalg.norm(y_hat_train, axis=1)
    y_train_norm = np.linalg.norm(y, axis=1)
    _, y_hat = gdml.predict(x_test)
    y_hat = np.array(y_hat)
    y_hat_norm = np.linalg.norm(y_hat, axis=1)
    y_test_norm = np.linalg.norm(y_test, axis=1)

    col = ['C1', 'C2', 'C3', 'C4']
    plt.close('all')
    fig, axis = plt.subplots(ncols=1, nrows=2, sharex=True)
    x1 = np.arange(len(y_hat_train_norm))
    axis[0].plot(x1, y_hat_train_norm, '.', alpha=0.1, color=col[0])
    axis[0].plot(x1, y_train_norm, '.', alpha=0.1, color=col[1])
    axis[0].plot(x1, movingaverage(y_hat_train_norm), color=col[0])
    axis[0].plot(x1, movingaverage(y_train_norm), color=col[1])
    x2 = np.arange(len(y_hat_norm)) + len(y_hat_train_norm)
    axis[0].plot(x2, y_hat_norm, '.', alpha=0.1, color=col[2])
    axis[0].plot(x2, y_test_norm, '.', alpha=0.1, color=col[3])
    axis[0].plot(x2, movingaverage(y_hat_norm), color=col[2])
    axis[0].plot(x2, movingaverage(y_test_norm), color=col[3])

    axis[0].set_xticks([])
    axis[0].set_title('The force norm')
    custom_lines = [Line2D([0], [0], color=col[1], lw=4),
                    Line2D([0], [0], color=col[0], lw=4),
                    Line2D([0], [0], color=col[2], lw=4),
                    Line2D([0], [0], color=col[3], lw=4)]
    axis[0].legend(custom_lines, ['training', 'predicted on train', 'predicted', 'ground truth'])

    x1 = np.arange(len(y_hat_train_norm))
    axis[1].plot(x1, y_hat_train_norm - y_train_norm, '.', alpha=0.1, color=col[0])
    axis[1].plot(x1, movingaverage(y_hat_train_norm - y_train_norm), color=col[0])
    x2 = np.arange(len(y_hat_norm)) + len(y_hat_train_norm)
    axis[1].plot(x2, y_hat_norm - y_test_norm, '.', alpha=0.1, color=col[1])
    axis[1].plot(x2, movingaverage(y_hat_norm - y_test_norm), color=col[1])
    custom_lines = [Line2D([0], [0], color=col[0], lw=4),
                    Line2D([0], [0], color=col[1], lw=4)]
    axis[1].legend(custom_lines, ['difference on train', 'difference on test'])
    plt.show()
