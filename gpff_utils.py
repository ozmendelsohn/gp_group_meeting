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
from matplotlib import cm # Colormaps
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ipywidgets as widgets
from ipywidgets import FloatSlider


def plot_gpr(gpr, x, y, x_test, y_test):
    y_hat_train = gpr.predict(x)
    y_hat_train_norm = np.linalg.norm(y_hat_train, axis=1)
    y_train_norm = np.linalg.norm(y, axis=1)
    y_hat = gpr.predict(x_test)
    y_hat_norm = np.linalg.norm(y_hat, axis=1)
    y_test_norm = np.linalg.norm(y_test, axis=1)

    plt.close('all')
    fig, axis = plt.subplots(ncols=1, nrows=2, sharex=True)
    x1 = np.arange(len(y_hat_train_norm))
    axis[0].plot(x1, y_hat_train_norm, label='predicted on train')
    axis[0].plot(x1, y_train_norm, label='training')
    x2 = np.arange(len(y_hat_norm))+len(y_hat_train_norm)
    axis[0].plot(x2, y_hat_norm, label='predicted')
    axis[0].plot(x2, y_test_norm, label='ground truth')
    axis[0].set_xticks([])
    axis[0].set_title('The force norm')
    axis[0].legend()

    x1 = np.arange(len(y_hat_train_norm))
    axis[1].plot(x1, y_hat_train_norm-y_train_norm, label='diffrence on train')
    x2 = np.arange(len(y_hat_norm))+len(y_hat_train_norm)
    axis[1].plot(x2, y_hat_norm-y_test_norm, label='diffrence on test')
    axis[1].legend()
    plt.show()