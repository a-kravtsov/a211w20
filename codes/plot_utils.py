import matplotlib.pylab as plt
import numpy as np

def plot_pretty(dpi=175,fontsize=9):
    # import pyplot and set some parameters to make plots prettier
    plt.rc("savefig", dpi=dpi)
    plt.rc("figure", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in') 
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5) 
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5) 
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [2., 2.])
    #if you don't have LaTeX installed on your laptop and this statement 
    # generates error, comment it out
    plt.rc('text', usetex=True)

    return

def plot_line_points(x, y, figsize=6, xlabel=' ', ylabel=' ', col= 'darkslateblue', 
                     xp = None, yp = None, points = False, pmarker='.', pcol='slateblue',
                     legend=None, plegend = None, legendloc='lower right', 
                     plot_title = None, grid=None, figsave = None):
    """
    A simple helper routine to make plots that involve a line and (optionally)
    a set of points, which was introduced and used during the first two weeks 
    of class.
    """
    plt.figure(figsize=(figsize,figsize))
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    # Initialize minor ticks
    plt.minorticks_on()

    if legend:
        plt.plot(x, y, lw = 1., c=col, label = legend)
        if points: 
            if plegend:
                plt.scatter(xp, yp, marker=pmarker, lw = 2., c=pcol, label=plegend)
            else:
                plt.scatter(xp, yp, marker=pmarker, lw = 2., c=pcol)
        plt.legend(frameon=False, loc=legendloc, fontsize=3.*figsize)
    else:
        plt.plot(x, y, lw = 1., c=col)
        if points:
            plt.scatter(xp, yp, marker=pmarker, lw = 2., c=pcol)

    if plot_title:
        plt.title(plot_title, fontsize=3.*figsize)
        
    if grid: 
        plt.grid(linestyle='dotted', lw=0.5, color='lightgray')
        
    if figsave:
        plt.savefig(figsave, bbox_inches='tight')

    plt.show()
    
from matplotlib import cm

def plot_color_map(x, y, data, xlim=[0.,1], ylim=[0.,1.], 
                   xlabel = ' ', ylabel = ' ', cmap='winter', colorbar=None, 
                   contours = False, levels = [], contcmap = 'winter',
                   plot_title=None, figsize=3.0, figsave=None):
    fig, ax = plt.subplots(figsize=(figsize,figsize))
    ax.axis([xlim[0], xlim[1], ylim[0], ylim[1]])

    plt.xlabel(xlabel); plt.ylabel(ylabel)
    cmap = cm.get_cmap(cmap)
    im = ax.pcolormesh(x, y, data, cmap=cmap, rasterized=False)
    if contours:
        ax.contour(x, y, data, levels=levels, cmap=contcmap)
    if colorbar: 
        fig.colorbar(im, ax=ax)
    if plot_title:
        plt.title(plot_title, fontsize=3.*figsize)

    if figsave:
        plt.savefig(figsave, bbox_inches='tight')
    plt.show()


def plot_histogram(data, bins=None, xlabel=' ', ylabel=' ', tickmarks = False, 
                   plot_title=' ', figsize=3.):
    "helper routine to histogram values in vector data"
    fig = plt.figure(figsize=(figsize, figsize)) # define figure environment
    plt.xlabel(xlabel); plt.ylabel(ylabel) # define axis labels
    
    # plot histogram of values in data
    plt.hist(data, bins=bins, histtype='stepfilled', 
             facecolor='slateblue', alpha=0.5)
    
    # this line is not stictly needed for plotting histogram
    # it plots individual values in data as little ticks along x-axis
    if tickmarks: 
        plt.plot(data, np.full_like(data, data.max()*0.1), '|k', 
                markeredgewidth=1)
    if plot_title:
        plt.title(plot_title, fontsize=3.*figsize)

    plt.show()