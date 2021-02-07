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
    xc = np.array([0.6, 0.6, 1.6, 2.]) + aboffs0
    yc = np.array([ymax + 1., 0.3, 0.5, 1.5]) + aboffs1
    sl = lambda y: (y - yc[2]) / (yc[3] - yc[2]) * (xc[3] - xc[2]) + xc[2]
    ax.plot(xc[0:2],yc[0:2],color=color, linestyle = linestyle)

    ax.plot(xc[1:3],yc[1:3],
    color=color, linestyle = linestyle)
    yarr = np.linspace(yc[2], ymax + 1., 5)
    xarr = sl(yarr)
    ax.plot(xarr, yarr, color=color, linestyle = linestyle,label=label)
    ax.set_ylim(ymin, ymax)

def is_stern(x, y, ab=True):
    '''
    For VEGA : 
    x = IRAC 3-4
    y = IRAC 1-2 
    1. x > 0.6 
    2. y > 0.2 * x + 0.18 
    3. y > 2.5 * x - 3.5
    m(AB) = m(Vega) +   2.79    3.26    3.73    4.40
    so ch1 - ch2 =
    mab1 + 2.79 - mab2 - 3.26 
    = mab1 - mab2 - 0.47
    ch3 - ch4 = 
    '''
    if ab:
        y = y + 0.48
        x = x + 0.7
    xmin = 0.6
    #1 
    ix1 = x > xmin    
    #2
    ix2 = y > 0.2 * x + 0.18
    #3
    ix3 = y > 2.5 * x - 3.5
    return ix1 & ix2 & ix3


def plot_lacy(ax, d12=True, c12='black',l12=':',
    color='red', linestyle='--', label=None, labeld12=None):
    '''
    log(S5.8/S3.6) > -0.1 -- x > -0.1
    log(S8.0/S4.5) > -0.2 -- y > -0.2
    log(S8.0/S4.5) ≤ 0.8 * log(S5.8/S3.6) + 0.5
    -- y <= 0.8 * x + 0.5
    '''
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    ax.plot([-0.1, xmax], [-0.2, -0.2],
        color=color, linestyle = linestyle, label=label)
    ax.plot([-0.1, -0.1], [-0.2, 0.42],
    color=color, linestyle = linestyle)
    ax.plot([-0.1, xmax], [0.42, 0.8 * xmax + 0.5],
    color=color, linestyle = linestyle)
    if d12:
        '''
        Plot Doneley 2012 revised criterion as well -- 
        x > 0.08  &&  y>0.15  &&  y >(1.12 * x)-0.27  
        &&  y<(1.12 * x)+0.27  &&  f4.5 > f3.6  &&  f5.8 > f4.5  &&  f8.0 > f5.8
        '''
        x0 = [0.08, 0.08]
        y0 = [0.15, 0.3596]
        ax.plot(x0, y0,
            color=c12, linestyle=l12,label=labeld12)
        yh = (1.12 * xmax)+0.27 
        if yh > ymax:
            yh = ymax 
            xh = (ymax - 0.27) / 1.12
            x1 = [0.08, xh]
            y1 = [0.3596, yh]
        else:
            x1 = [0.08, xmax]
            y1 = [0.3596, yh]
        ax.plot(x1, y1, 
        color=c12, linestyle = l12)
        x2 = [0.08, 0.375]
        y2 = [0.15, 0.15]
        ax.plot(x2, y2, 
        color=c12, linestyle = l12)
        x3 = [0.375, xmax]
        y3 = [0.15, (1.12 * xmax) - 0.27]
        ax.plot(x3, y3, 
        color=c12, linestyle = l12)

def is_lacy(x, y):
    '''
    log(S5.8/S3.6) > -0.1 -- x > -0.1
    log(S8.0/S4.5) > -0.2 -- y > -0.2
    log(S8.0/S4.5) ≤ 0.8 * log(S5.8/S3.6) + 0.5
    -- y <= 0.8 * x + 0.5    
    '''
    c1 = x > -0.1
    c2 = y > -0.2 
    c3 = y <= 0.8 * x + 0.5
    return c1 & c2 & c3

def is_d12(x, y, f1, f2, f3, f4):
    '''
    x > 0.08  &&  y>0.15  &&  y >(1.12 * x)-0.27  
    &&  y<(1.12 * x)+0.27  &&  f4.5 > f3.6  &&  f5.8 > f4.5  &&  f8.0 > f5.8
    '''
    c1 = x > 0.08  
    c2 = y > 0.15
    c3 = y > (1.12 * x) - 0.27  
    c4 = y < (1.12 * x) + 0.27
    c5 = (f2 > f1) & (f3 > f2) & (f4 > f3)
    return c1 & c2 & c3 & c4 & c5

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
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.loc'] = 'center left'
    plt.rcParams['axes.linewidth'] = lw_axis
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.interactive(False)


def abline(slope=1, intercept=0):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')
