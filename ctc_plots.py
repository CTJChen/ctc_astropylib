import numpy as np


def plot_stern(ax, ab=True, lo=False, color='red', linestyle='--', label=None):
    if ab:
        aboffs0 = -0.7
        aboffs1 = -0.48
    else:
        aboffs0 = 0.
        aboffs1 = 0.
    ymin, ymax = ax.get_ylim()
    xc = np.array([0.6,0.6,1.6,2.]) + aboffs0
    yc = np.array([ymax+1.,0.3,0.5,1.5]) + aboffs1
    sl = lambda y: (y - yc[2]) / (yc[3] - yc[2]) * (xc[3] - xc[2]) + xc[2]
    ax.plot(xc[0:2],yc[0:2],color=color, linestyle = linestyle)

    ax.plot(xc[1:3],yc[1:3],
    color=color, linestyle = linestyle)
    yarr = np.linspace(yc[2], ymax + 1., 5)
    xarr = sl(yarr)
    ax.plot(xarr, yarr, color=color, linestyle = linestyle,label=label)
    ax.set_ylim(ymin, ymax)