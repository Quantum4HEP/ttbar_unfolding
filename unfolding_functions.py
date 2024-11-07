import ROOT
import numpy as np
import pylab as plt
from qunfold import QUnfolder
from qunfold.root2numpy import  TH1_to_numpy, TMatrix_to_numpy
#from qunfold.utils import normalize_response, compute_chi2

def roounfolder(method, roo_response, th1_measured, th1_fake):
    if method == "MI":
        unfolder = ROOT.RooUnfoldInvert(roo_response, th1_measured)
    elif method == "IBU":
        unfolder = ROOT.RooUnfoldBayes(roo_response, th1_measured-th1_fake, 4)
    
    roo_unfolded=unfolder.Hreco()
    cov_matrix = unfolder.Eunfold()
    sol = TH1_to_numpy(roo_unfolded, overflow=True)
    cov_matrix = unfolder.Eunfold()
    cov=TMatrix_to_numpy(cov_matrix)
    return sol, cov

def qunfolder(method, response, measured, fake, binning, num_reads=None, lam=0.0):
    qunfolder = QUnfolder(response=response, measured=measured-fake, binning=binning, lam=lam)
    qunfolder.initialize_qubo_model()
    if method == "GRB":
        sol, cov = qunfolder.solve_gurobi_integer()
    elif method == "SA":
        sol, cov = qunfolder.solve_simulated_annealing(num_reads=num_reads, num_toys=None)
    elif method == "HYB":
        sol, cov = qunfolder.solve_hybrid_sampler()
    elif method == "QA":
        qunfolder.set_quantum_device()
        qunfolder.set_graph_embedding()
        sol, cov[1:-1], _ = qunfolder.solve_quantum_annealing(num_reads=num_reads, num_toys=None)

    return sol, cov


"""
#da aggiustare
def plot_unfolding(fig, true, measured, method, sol, cov, binning, norm=True):
    
    #conviene fare tre liste, una con la soluzione, una con la covarianza e una con il metodo
    
    if fig == False:
        fig = plt.figure(figsize=(9, 7))
        gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0)
        ax1 = fig.add_subplot(gs[0])
        #ax2 = fig.add_subplot(gs[1], sharex=ax1)

        QPlotter.histogram_plot(ax=ax1, xedges=binning, hist=true, label="Truth", norm=True)
        QPlotter.histogram_plot(ax=ax1, xedges=binning, hist=measured, label="Measured", norm=True)

        xlims = (binning[1], binning[-2])
        xpt = xpt = 0.5 * (binning[:-1] + binning[1:])
        ax1.set_ybound(0, 0.08)
    
    err = np.sqrt(np.diag(cov))
    chi2 = compute_chi2(observed=sol, expected=true)
    QPlotter.errorbar_plot(ax=ax1, xmid=xpt[1:-1], hist=sol[1:-1], err=err, xlims=xlims, label=method, chi2=chi2, norm=norm)
"""

