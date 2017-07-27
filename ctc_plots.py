import numpy as np
from matplotlib import pyplot as plt


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



def init_plotting(fsize=None,lw_axis=None):
    #plt.rcParams['figure.figsize'] = (8, 6)
    if fsize is None: fsize=14
    if lw_axis is None: lw_axis=1.0
    plt.rcParams['font.size'] = fsize
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.weight'] = 'medium'#'bold','medium'
    plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 0.6*plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 0.9*plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 0.9*plt.rcParams['font.size']
    #plt.rcParams['savefig.dpi'] = 2*plt.rcParams['savefig.dpi']
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['xtick.minor.size'] = 4
    plt.rcParams['xtick.major.width'] = 1
    plt.rcParams['xtick.minor.width'] = 1
    plt.rcParams['ytick.major.size'] = 8
    plt.rcParams['ytick.minor.size'] = 4
    plt.rcParams['ytick.major.width'] = 1
    plt.rcParams['ytick.minor.width'] = 1
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['axes.linewidth'] = lw_axis
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.interactive(False)
