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
        sol, cov = qunfolder.solve_simulate_annealing(num_reads=num_reads, num_toys=None)
    elif method == "HYB":
        sol, cov = qunfolder.solve_hybrud_sampler()
    elif method == "QA":
        qunfolder.set_quantum_device()
        qunfolder.set_graph_embedding()
        sol, cov[1:-1], _ = qunfolder.solve_quantum_annealing(num_reads=num_reads, num_toys=None)

    return sol, cov



