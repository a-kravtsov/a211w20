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
    #plt.rc('text', usetex=True)

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

import scipy.optimize as opt
from matplotlib.colors import LogNorm

def conf_interval(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level

def plot_2d_dist(x,y, xlim, ylim, nxbins, nybins, figsize=(5,5), 
                cmin=1.e-4, cmax=1.0, smooth=None, 
                log=False, weights=None, xlabel='x',ylabel='y', 
                clevs=None, fig_setup=None, savefig=None):
    """
    construct and plot a binned, 2d distribution in the x-y plane 
    using nxbins and nybins in x- and y- direction, respectively
    
    log = specifies whether logged quantities are passed to be plotted on log-scale outside this routine
    """
    if fig_setup is None:
        fig, ax = plt.subplots(figsize=figsize)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
    else:
        ax = fig_setup
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim); ax.set_ylim(ylim)

    if xlim[1] < 0.: ax.invert_xaxis()

    if weights is None: weights = np.ones_like(x)
    H, xbins, ybins = np.histogram2d(x, y, weights=weights, bins=(np.linspace(xlim[0], xlim[1], nxbins),np.linspace(ylim[0], ylim[1], nybins)))
    
    H = np.rot90(H); H = np.flipud(H); 
             
    X,Y = np.meshgrid(xbins[:-1],ybins[:-1]) 

    if smooth != None:
        from scipy.signal import wiener
        H = wiener(H, mysize=smooth)
        
    H = H/np.sum(H)        
    Hmask = np.ma.masked_where(H==0,H)
    
    if log:
        X = np.power(10.,X); Y = np.power(10.,Y)

    pcol = ax.pcolormesh(X, Y,(Hmask), vmin=cmin*np.max(Hmask), vmax=cmax*np.max(Hmask), cmap=plt.cm.BuPu, norm = LogNorm(), linewidth=0., rasterized=True)
    pcol.set_edgecolor('face')
    
    # plot contours if contour levels are specified in clevs 
    if clevs is not None:
        lvls = []
        for cld in clevs:  
            sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )   
            lvls.append(sig)
        
        ax.contour(X, Y, H, linewidths=(1.0,0.75, 0.5, 0.25), colors='black', levels = sorted(lvls), 
                norm = LogNorm(), extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]])
    if savefig:
        plt.savefig(savefig,bbox_inches='tight')
    if fig_setup is None:
        plt.show()
    return

import itertools as it

def triage(par, weights, parnames, figsize=[5,5], nbins = 30, figname=None, fontsize=8):
    
    npar = np.size(par[1,:])
    
    f, ax = plt.subplots(npar, npar, figsize=(figsize), sharex='col')
    # f, ax = plt.subplots(figsize=(10,10), sharex='col', sharey='row')
    plt.rc('font',size=fontsize)
    for h,v in it.product(range(npar), range(npar)) :
        if v < h :
            hvals, xedges, yedges = np.histogram2d(par[:,v], par[:,h], weights=weights[:,0], bins = nbins)
            hvals = np.rot90(hvals)
            hvals = np.flipud(hvals)
             
            Hmasked = np.ma.masked_where(hvals==0, hvals)
            hvals = hvals / np.sum(hvals)        
             
            X,Y = np.meshgrid(xedges,yedges) 
             
            sig1 = opt.brentq( conf_interval, 0., 1., args=(hvals,0.683) )
            sig2 = opt.brentq( conf_interval, 0., 1., args=(hvals,0.953) )
            sig3 = opt.brentq( conf_interval, 0., 1., args=(hvals,0.997) )
            lvls = [sig3, sig2, sig1]   
                     
            ax[h,v].pcolor(X, Y, (Hmasked), cmap=plt.cm.BuPu, norm = LogNorm())
            ax[h,v].contour(hvals, linewidths=(1.0, 0.5, 0.25), colors='lavender', levels = lvls, norm = LogNorm(), extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]])
            if v > 0:
                ax[h,v].get_yaxis().set_ticklabels([])
        elif v == h :
            ax[h,v].hist(par[:,h],bins = nbins,color='mediumslateblue',histtype='step',lw=1.5)
            ax[h,v].yaxis.tick_right()
            ax[h,v].get_yticklabels()
            
            conf_lims = np.percentile(par[:,h], [2.5, 16, 50, 84, 97.5])
            hmean = np.mean(par[:,h])
            hstd = np.std(par[:,h])
            
            print(parnames[h] + '\t%.3f +- %.3f'%(hmean, hstd)+ '; [2.5, 16, 50, 84, 97.5] %-tiles: ' + ' '.join(['%.3f'%(p) for p in conf_lims]))
            
            #textable.write( parnames[h] + ' & %.4f & %.4f'%(hmean, hstd)+ ' & ' + ' & '.join(['%.4f'%(p) for p in conf_lims]) + '\\\\\n')
            
            #for i in range(len(conf_lims)) :   ax[h,v].axvline(conf_lims[i], color='lavender', lw = (3. - np.abs(2-i))/2. )
            
        else :
            ax[h,v].axis('off')
             
        if v == 0:
            ax[h,v].set_ylabel(parnames[h])
            ax[h,v].get_yaxis().set_label_coords(-0.35,0.5)
        if h == npar-1:
            ax[h,v].set_xlabel(parnames[v])
            ax[h,v].get_xaxis().set_label_coords(0.5,-0.35)
            labels = ax[h,v].get_xticklabels()
            for label in labels: 
                label.set_rotation(90) 
         
         
    plt.tight_layout(pad=1.5, w_pad=-4, h_pad=-0.6)
    if figname:
        plt.savefig(figname, bbox_inches='tight')

def distance_matrix(A, B):
    '''
    Given two sets of data points, computes the Euclidean distances
    between each pair of points.

    *A*: (N, D) array of data points
    *B*: (M, D) array of data points

    Returns: (N, M) array of Euclidean distances between points.
    '''
    Na, D = A.shape
    Nb, Db = B.shape
    assert(Db == D)
    dists = np.zeros((Na,Nb))
    for a in range(Na):
        dists[a,:] = np.sqrt(np.sum((A[a] - B)**2, axis=1))
    return dists

# Copied and very slightly modified from scipy
def voronoi_plot_2d(vor, ax=None):
    #ptp_bound = vor.points.ptp(axis=0)
    ptp_bound = np.array([1000,1000])
    
    center = vor.points.mean(axis=0)
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            ax.plot([vor.vertices[i,0], far_point[0]],
                    [vor.vertices[i,1], far_point[1]], 'k--')

colors = 'brgmck'

def plot_kmeans(i, X, K, centroids, newcentroids, nearest, show=True):
    import pylab as plt
    plt.clf()
    plotsymbol = 'o'
    if nearest is None:
        distances = distance_matrix(X, centroids)
        nearest = np.argmin(distances, axis=1)
        
    for i,c in enumerate(centroids):
        I = np.flatnonzero(nearest == i)
        plt.plot(X[I,0], X[I,1], plotsymbol, mfc=colors[i], mec='k')
        
    ax = plt.axis()
    for i,(oc,nc) in enumerate(zip(centroids, newcentroids)):
        plt.plot(oc[0], oc[1], 'kx', mew=2, ms=10)
        plt.plot([oc[0], nc[0]], [oc[1], nc[1]], '-', color=colors[i])
        plt.plot(nc[0], nc[1], 'x', mew=2, ms=15, color=colors[i])
        
    vor = None
    if K > 2:
        from scipy.spatial import Voronoi #, voronoi_plot_2d
        vor = Voronoi(centroids)
        voronoi_plot_2d(vor, plt.gca())
    else:
        mid = np.mean(centroids, axis=0)
        x0,y0 = centroids[0]
        x1,y1 = centroids[1]
        slope = (y1-y0)/(x1-x0)
        slope = -1./slope
        run = 1000.
        plt.plot([mid[0] - run, mid[0] + run],
                 [mid[1] - run*slope, mid[1] + run*slope], 'k--')
    plt.axis(ax)
    if show:
        plt.show()
