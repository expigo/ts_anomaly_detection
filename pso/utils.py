import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

from pso import PSO


cmap = [(0, '#2f9599'), (0.45, '#eeeeee'), (1, '#8800ff')]
cmap = cm.colors.LinearSegmentedColormap.from_list('Custom', cmap, N=256)


def plot_2d(function, n_space=1000, cmap=cmap, show=True):
    X_domain, Y_domain = function.input_domain

    X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
    X, Y = np.meshgrid(X, Y)
    XY = np.array([X, Y])
    Z = np.apply_along_axis(function, 0, XY)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.contourf(X, Y, Z, levels=30, cmap=cmap, alpha=0.7)

    # add labels and set equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect(aspect='equal')
    if show:
        plt.show()

    return fig,  X, Y, Z


def plot_3d(function, n_space=1000, cmap=cmap, show=True):
    X_domain, Y_domain = function.input_domain

    X, Y = np.linspace(*X_domain, n_space), np.linspace(*Y_domain, n_space)
    X, Y = np.meshgrid(X, Y)
    XY = np.array([X, Y])
    Z = np.apply_along_axis(function, 0, XY)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Plot the surface.
    ax.plot_surface(X, Y, Z, cmap=cmap,
                    linewidth=0, antialiased=True, alpha=0.7)
    ax.contour(X, Y, Z, zdir='z', levels=30, offset=np.min(Z), cmap=cmap)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    if show:
        plt.show()


class AlpineN1:
    continuous = False
    convex = False
    differentiable = True
    mutimodal = True
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (isinstance(d, int) and (not d < 0)), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def __init__(self, d):
        self.d = d
        self.input_domain = np.array([[0, 10] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for i in range(d)])
        return (X, self(X))

    def __call__(self, X):
        res = np.sum(np.abs(X * np.sin(X) + 0.1 * X))
        return res


class AlpineN2:
    continuous = True
    convex = False
    differentiable = True
    mutimodal = True
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (isinstance(d, int) and (not d < 0)), "The dimension d must be None or a positive integer"
        return  (d is None) or (d > 0)

    def __init__(self, d):
        self.d = d
        self.input_domain = np.array([[0, 10] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([7.917 for i in range(d)])
        return (X, self(X))

    def __call__(self, X):
        res = -np.prod(np.sqrt(X) * np.sin(X))
        return res

class Sphere:
    continuous = True
    convex = True
    differentiable = False
    mutimodal = False
    parametric = False

    @classmethod
    def is_dim_compatible(cls, d):
        assert (d is None) or (isinstance(d, int) and (not d < 0)), "The dimension d must be None or a positive integer"
        return (d is None) or (d > 0)

    def __init__(self, d,):
        self.d = d
        # self.input_domain = np.array([[-5, 5], [-5, 5]])
        self.input_domain = np.array([[-5.12, 5.12] for _ in range(d)])

    def get_global_minimum(self, d):
        X = np.array([0 for _ in range(d)])
        return (X, self(X))

    def __call__(self, X):
        d = X.shape[0]
        res = np.sum(X**2)
        return res

#
def plot_pso(function, pso:PSO):
    # time_steps = N
    time_steps = pso.n_iters

    positions = [i.position_history for i in pso.particles]
    positions = [list(map(lambda v: np.array(v.raw), l)) for l in positions]
    best_postitions = [p.raw for p in pso.best_position_history]

    fig, X, Y, Z = plot_2d(function, show=False)

    def animate(i):
        """ Perform animation step. """

        # important - the figure is cleared and new axes are added
        fig.clear()

        xlim, ylim = function.input_domain
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=tuple(xlim), ylim=tuple(ylim))
        # the new axes must be re-formatted
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(b=None)

        # add contours and contours lines
        # ax.contour(X, Y, Z, levels=30, linewidths=0.5, colors='#999')
        ax.contourf(X, Y, Z, levels=30, cmap=cmap, alpha=0.7)

        # add labels and set equal aspect ratio
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect(aspect='equal')

        marker_size = 25

        # add the elements for this frame are added
        ax.text(0.02, 0.95, f'[constricted] Time step  = {i}', transform=ax.transAxes)
        xs = [p[i][0] for p in positions]
        ys = [p[i][1] for p in positions]
        s = ax.scatter(xs, ys,
                       s=marker_size,
                       # c=0,
                       cmap="RdBu_r", marker="o", edgecolor=None)
        # fig.colorbar(s)
        best_x = best_postitions[i][0]
        best_y = best_postitions[i][1]
        s = ax.scatter(best_x, best_y,
                       s=marker_size,
                       c='firebrick',
                       marker="o")

    ani = animation.FuncAnimation(fig, animate, interval=100, frames=range(time_steps))

    print('done')
    from datetime import datetime
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    ani.save(f'animation-{date}.gif', writer='pillow')