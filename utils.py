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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm  # Colormaps
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ipywidgets as widgets
from ipywidgets import FloatSlider
import matplotlib.pyplot as plt
import numpy as np


def univariate_plot(univariate_normal, normal=True, xlim=[-5, 5]):
    def update(mu=0.0, sigma=1):
        """Remove old lines from plot and plot new one"""
        [l.remove() for l in ax.lines]
        if normal:
            label = '$\mathcal{N}(' + str(mu) + ', ' + str(sigma) + ')$'
        else:
            label = ''
        x = np.linspace(-5, 5, num=150)
        plt.plot(x, univariate_normal(x, mu, sigma),
                 label=label, color='C0')

        plt.xlabel('$y$', fontsize=13)
        plt.ylabel('density: $p(x)$', fontsize=13)
        if normal:
            plt.title('Univariate normal distributions')
        plt.ylim([0, 1])
        plt.xlim(xlim)
        plt.legend(loc=1)
        fig.subplots_adjust(bottom=0.15)

    fig, ax = plt.subplots(figsize=(8, 5))
    widgets.interact(update, mu=(-3, 3, .1), sigma=(0.1, 2, .01))


def multivariate_normal(x, d, mean, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance))) *
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))


def generate_surface(mean, covariance, d, nb_of_x=40):
    """Helper function to generate density surface."""
    nb_of_x = nb_of_x  # grid size
    x1s = np.linspace(-5, 5, num=nb_of_x)
    x2s = np.linspace(-5, 5, num=nb_of_x)
    x1, x2 = np.meshgrid(x1s, x2s)  # Generate grid
    pdf = np.zeros((nb_of_x, nb_of_x))
    # Fill the cost matrix for each combination of weights
    for i in range(nb_of_x):
        for j in range(nb_of_x):
            pdf[i, j] = multivariate_normal(
                np.matrix([[x1[i, j]], [x2[i, j]]]),
                d, mean, covariance)
    return x1, x2, pdf  # x1, x2, pdf(x1,x2)


def multivariate_plot(multivariate_normal, nb_of_x=40):
    def generate_surface(mean, covariance, d, nb_of_x=40):
        """Helper function to generate density surface."""
        nb_of_x = nb_of_x  # grid size
        x1s = np.linspace(-5, 5, num=nb_of_x)
        x2s = np.linspace(-5, 5, num=nb_of_x)
        x1, x2 = np.meshgrid(x1s, x2s)  # Generate grid
        pdf = np.zeros((nb_of_x, nb_of_x))
        # Fill the cost matrix for each combination of weights
        for i in range(nb_of_x):
            for j in range(nb_of_x):
                pdf[i, j] = multivariate_normal(
                    np.matrix([[x1[i, j]], [x2[i, j]]]),
                    d, mean, covariance)
        return x1, x2, pdf  # x1, x2, pdf(x1,x2)

    def update(C=0.8):
        plt.clf()
        ax2 = fig.add_subplot(111)
        bivariate_mean_strong = np.matrix([[0.], [1.]])  # Mean
        bivariate_covariance_strong = np.matrix([
            [1., C],
            [C, 1.]])
        # subplot

        d = 2  # number of dimensions
        # Plot of correlated Normals
        bivariate_mean = bivariate_mean_strong  # Mean
        bivariate_covariance = bivariate_covariance_strong
        x1, x2, p = generate_surface(
            bivariate_mean, bivariate_covariance, d, nb_of_x)
        # Plot bivariate distributiong
        con = ax2.contourf(x1, x2, p, nb_of_x, cmap=cm.YlGnBu)
        ax2.set_xlabel('$y_1$', fontsize=13)
        ax2.set_ylabel('$y_2$', fontsize=13)
        ax2.axis([-2.5, 2.5, -1.5, 3.5])
        ax2.set_aspect('equal')
        ax2.set_aspect('equal')
        ax2.set_title(fr'Correlated variables, C={C}, $\mu_1$={0} $\mu_2$={1}', fontsize=11)

        # Add colorbar and title
        fig.subplots_adjust(right=0.84)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(con, cax=cbar_ax)
        cbar.ax.set_ylabel('$p(x_1, x_2)$', fontsize=13)
        plt.suptitle('Bivariate normal distributions', fontsize=13, y=0.98)
        plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 5))
    c_widget = FloatSlider(min=0.0, max=.99, step=0.01, continuous_update=False)
    widgets.interact(update, C=c_widget)


def condition_plot(nb_of_x=40):
    def univariate_normal(x, mean, variance):
        """pdf of the univariate normal distribution."""
        return ((1. / np.sqrt(2 * np.pi * variance)) *
                np.exp(-(x - mean) ** 2 / (2 * variance)))

    def multivariate_normal(x, d, mean, covariance):
        """pdf of the multivariate normal distribution."""
        x_m = x - mean
        return (1. / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance))) *
                np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

    def generate_surface(mean, covariance, d, nb_of_x):
        """Helper function to generate density surface."""
        nb_of_x = nb_of_x  # grid size
        x1s = np.linspace(-5, 5, num=nb_of_x)
        x2s = np.linspace(-5, 5, num=nb_of_x)
        x1, x2 = np.meshgrid(x1s, x2s)  # Generate grid
        pdf = np.zeros((nb_of_x, nb_of_x))
        # Fill the cost matrix for each combination of weights
        for i in range(nb_of_x):
            for j in range(nb_of_x):
                pdf[i, j] = multivariate_normal(
                    np.matrix([[x1[i, j]], [x2[i, j]]]),
                    d, mean, covariance)
        return x1, x2, pdf  # x1, x2, pdf(x1,x2)

    def update(y1=-1., y2=1., C=0.8):
        plt.clf()
        d = 2  # dimensions
        mean = np.matrix([[0.], [1.]])
        cov = np.matrix([[1, C],
                         [C, 1]])

        # Get the mean values from the vector
        mean_x = mean[0, 0]
        mean_y = mean[1, 0]
        # Get the blocks (single values in this case) from 
        #  the covariance matrix
        A = cov[0, 0]
        B = cov[1, 1]
        C = cov[0, 1]  # = C transpose in this case

        # Calculate x|y
        y_condition = y2  # To condition on y
        mean_xgiveny = mean_x + (C * (1 / B) * (y_condition - mean_y))
        cov_xgiveny = A - C * (1 / B) * C

        # Calculate y|x
        x_condition = y1  # To condition on x
        mean_ygivenx = mean_y + (C * (1 / A) * (x_condition - mean_x))
        cov_ygivenx = B - (C * (1 / A) * C)

        # Plot the conditional distributions

        gs = gridspec.GridSpec(
            2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
        # gs.update(wspace=0., hspace=0.)
        plt.suptitle('Conditional distributions', y=0.93)

        # Plot surface on top left
        ax1 = plt.subplot(gs[0])
        x, y, p = generate_surface(mean, cov, d, nb_of_x)
        # Plot bivariate distribution
        con = ax1.contourf(x, y, p, 100, cmap=cm.YlGnBu)
        # y=1 that is conditioned upon
        ax1.plot([-2.5, 2.5], [y_condition, y_condition], 'r--')
        # x=-1. that is conditioned upon
        ax1.plot([x_condition, x_condition], [-1.5, 3.5], 'b--')
        ax1.set_xlabel('$y_1$', fontsize=13)
        ax1.set_ylabel('$y_2$', fontsize=13)
        ax1.yaxis.set_label_position('right')
        ax1.axis([-2.5, 2.5, -1.5, 3.5])

        # Plot y|x
        ax2 = plt.subplot(gs[1])
        yx = np.linspace(-5, 5, num=100)
        pyx = univariate_normal(yx, mean_ygivenx, cov_ygivenx)
        # Plot univariate distribution
        ax2.plot(pyx, yx, 'b--',
                 label=f'$p(y_2|y_1={x_condition:.1f})$')
        ax2.legend(loc=0)
        ax2.set_xlabel('density', fontsize=13)
        ax2.set_ylim(-1.5, 3.5)

        # Plot x|y
        ax3 = plt.subplot(gs[2])
        xy = np.linspace(-5, 5, num=100)
        pxy = univariate_normal(xy, mean_xgiveny, cov_xgiveny)
        # Plot univariate distribution
        ax3.plot(xy, pxy, 'r--',
                 label=f'$p(y_1|y_2={y_condition:.1f})$')
        ax3.legend(loc=0)
        ax3.set_ylabel('density', fontsize=13)
        ax3.yaxis.set_label_position('right')
        ax3.set_xlim(-2.5, 2.5)

        # Clear axis 4 and plot colarbar in its place
        ax4 = plt.subplot(gs[3])
        ax4.set_visible(False)
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes('left', size='20%', pad=0.05)
        cbar = fig.colorbar(con, cax=cax)
        cbar.ax.set_ylabel('density: $p(x, y)$', fontsize=13)
        plt.show()

    fig = plt.figure(figsize=(8, 8))
    x_widget = FloatSlider(min=-1.0, max=1.0, step=0.1, continuous_update=False)
    y_widget = FloatSlider(min=-1.0, max=1.0, step=0.1, continuous_update=False)
    c_widget = FloatSlider(min=0.0, max=.99, step=0.01, continuous_update=False)
    widgets.interact(update, y1=x_widget, y2=y_widget, C=c_widget)


def plot_noise_sin(n=50, noise=0.1, plot_line=False, label=False):
    x = np.linspace(0, np.pi * 2, n)
    y = np.sin(x) + np.random.normal(loc=0.0, scale=noise, size=n)
    if plot_line:
        xi = np.linspace(0, np.pi * 2, 10 * n)
        plt.plot(xi, np.sin(xi), color='C4', label=r'$sin(x)$')
    plt.scatter(x, y, label=label)
    if label:
        plt.legend()
    plt.show()
    return x


def heapmap_kernal(x, kernel, noise=0.1):
    K = kernel(x, x) + (noise ** 2) * np.eye(len(x))
    plt.imshow(K, cmap='Blues')
    plt.show()


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i + 1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')


def plot_gp_2D(gx, gy, mu, X_train, Y_train, title, i):
    ax = plt.gcf().add_subplot(1, 2, i, projection='3d')
    ax.plot_surface(gx, gy, mu.reshape(gx.shape), cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
    ax.scatter(X_train[:, 0], X_train[:, 1], Y_train, c=Y_train, cmap=cm.coolwarm)
    ax.set_title(title)


def kernel(X1, X2, l=1.0, sigma_f=1.0):
    '''
    Isotropic squared exponential kernel. Computes
    a covariance matrix from points in X1 and X2.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        Covariance matrix (m x n).
    '''
    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * sqdist)


def posterior_predictive(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    '''
    Computes the suffifient statistics of the GP posterior predictive distribution
    from m training data X_train and Y_train and n new inputs X_s.

    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    '''
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # Equation (4)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (5)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


def nll_fn(X_train, Y_train, noise, kernel=kernel, naive=True):
    '''
    Returns a function that computes the negative log marginal
    likelihood for training data X_train and Y_train and given 
    noise level.

    Args:
        X_train: training locations (m x d).
        Y_train: training targets (m x 1).
        noise: known noise level of Y_train.
        naive: if True use a naive implementation of Eq. (7), if 
               False use a numerically more stable implementation. 

    Returns:
        Minimization objective.
    '''

    def nll_naive(theta):
        # Naive implementation of Eq. (7). Works well for the examples 
        # in this article but is numerically less stable compared to 
        # the implementation in nll_stable below.
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise ** 2 * np.eye(len(X_train))
        return (0.5 * np.log(det(K)) +
                0.5 * Y_train.T.dot(inv(K).dot(Y_train)) +
                0.5 * len(X_train) * np.log(2 * np.pi)).reshape([1, ])

    def nll_stable(theta):
        # Numerically more stable implementation of Eq. (7) as described
        # in http://www.gaussianprocess.org/gpml/chapters/RW2.pdf, Section
        # 2.2, Algorithm 2.1.
        K = kernel(X_train, X_train, l=theta[0], sigma_f=theta[1]) + \
            noise ** 2 * np.eye(len(X_train))
        L = cholesky(K)

        return (np.sum(np.log(np.diagonal(L))) +
                0.5 * Y_train.T.dot(lstsq(L.T, lstsq(L, Y_train)[0])[0]) +
                0.5 * len(X_train) * np.log(2 * np.pi)).reshape([1, ])

    if naive:
        return nll_naive
    else:
        return nll_stable


def gaussian_process(x, f, noise, posterior_predictive=posterior_predictive, kernel=kernel):
    X_train = np.array(x).reshape(-1, 1)
    Y_train = f(X_train) + noise * np.random.randn(*X_train.shape)
    X = np.arange(-5, 5, 0.1).reshape(-1, 1)
    # Minimize the negative log-likelihood w.r.t. parameters l and sigma_f.
    # We should actually run the minimization several times with different
    # initializations to avoid local minima but this is skipped here for
    # simplicity.
    res = minimize(nll_fn(X_train, Y_train, noise, kernel=kernel), np.array([1, 1]),
                   bounds=((1e-5, None), (1e-5, None)),
                   method='L-BFGS-B')

    # Store the optimization results in global variables so that we can
    # compare it later with the results from other implementations.
    l_opt, sigma_f_opt = res.x
    # print(l_opt, sigma_f_opt)
    # Compute the prosterior predictive statistics with optimized kernel parameters and plot the results
    plt.close('all')
    mu_s, cov_s = posterior_predictive(X, X_train, Y_train, l=l_opt, sigma_f=sigma_f_opt, sigma_y=noise)

    plt.plot(X, f(X), ':', label='Ground Truth')
    plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train)
    plt.legend()
    plt.show()


def gaussian_process_interactive(f, noise, kernal=ConstantKernel(1.0) * RBF(length_scale=1.0)):
    pass
    # Compute posterior predictive mean and covariance
    x, y = [], []
    click = 0
    X = np.arange(-5, 5, 0.01).reshape(-1, 1)
    # Plot the results
    fig, ax = plt.subplots()

    def onclick(event):
        plt.clf()
        x.append(event.xdata)
        y.append(event.ydata)

        gpr = GaussianProcessRegressor(kernel=kernal, alpha=noise ** 2)
        gpr.fit(np.array(x).reshape([-1, 1]), np.array(y).reshape([-1, 1]))
        mu_s, cov_s = gpr.predict(X, return_cov=True)

        plt.plot(X, f(X), ':', label='Ground Truth')
        plot_gp(mu_s, cov_s, X, X_train=x, Y_train=y)
        plt.xlim([min(X), max(X)])
        plt.ylim([min(f(X)) - 0.5, max(f(X)) + 0.5])
        plt.legend()

    plt.plot(X, f(X), ':', label='Ground Truth')
    plt.legend()
    plt.xlim([min(X), max(X)])
    plt.ylim([min(f(X)) - 0.5, max(f(X)) + 0.5])
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    plt.draw()


from itertools import product


class gprArray:
    def __init__(self, N, kernel, alpha):
        self.gpr_array = [GaussianProcessRegressor(kernel=kernal, alpha=noise ** 2) for _ in range(N)]

    def fit(self, x, y):
        y_split = [[] for _ in range(y.shape[1])]
        for i, j in product(range(y.shape[1]), range(y.shape[0])):
            y_split[i].append(y[j, i])
        y_split = [np.array(y).reshape([-1, 1]) for y in y_split]
        for i, gp in enumerate(self.gpr_array):
            gp.fit(x, y_split[i])

    def predict(self, x):
        y_split = []
        for gp in self.gpr_array:
            y_split.append(gp.predict(x))
        y = np.zeros([x.shape[0], len(self.gpr_array)])
        for i, j in product(range(y.shape[1]), range(y.shape[0])):
            y[j, i] = y_split[i][j]
        return y
