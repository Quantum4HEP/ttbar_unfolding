import ROOT
from qunfold import QUnfolder, QPlotter
import numpy as np
from qunfold.utils import  normalize_response, lambda_optimizer, compute_chi2
from qunfold.root2numpy import TH2_to_numpy, TH1_to_numpy
import matplotlib.pyplot as plt
from analysis_function import get_purity, get_efficiency, gurobi, chi2_fixed_reads, chi2_fixed_toys

myFile_1 = ROOT.TFile.Open("hadpol_particle.root")
myFile_2 = ROOT.TFile.Open("hadpol_parton.root")

res_root = myFile_1.Get("costhetap_range800_matrix_hadpol") #800
reco_root = myFile_1.Get("costhetap_range800_particle_hadpol")
truth_root = myFile_2.Get("costhetap_range800_parton_hadpol") 

reco_mc_root=res_root.ProjectionX("reco_mc")
truth_mc_root=res_root.ProjectionY("truth_mc")
#purity and efficiency

#purity = get_purity(reco_mc_root, reco_root)
#efficiency = get_efficiency(truth_mc_root, truth_root)

#binning

bins = 10
xrange = np.linspace(start=-1, stop=1, num=bins + 1).tolist()
binning = np.array([-np.inf] + xrange + [np.inf])  # under/over-flow bins


#TH1, TH2 to array
truth = TH1_to_numpy(truth_root, overflow =True)
truth_mc = TH1_to_numpy(truth_mc_root, overflow=True)
reco = TH1_to_numpy(reco_root, overflow =True)
reco_mc=TH1_to_numpy(reco_mc_root, overflow=True)
response = TH2_to_numpy(res_root, overflow=True)
response = normalize_response(TH2_to_numpy(res_root, overflow=True), truth_mc)

#find the best lambda and print
#lam = gurobi(response = response, reco = reco_mc, truth = truth, efficiency = efficiency)#ci dovrei mettere reco*purity, non reco_mc, ma qua è uguale

#calculate chi2
#chi2_fixed_reads(truth = truth, reco = reco_mc, response = response, efficiency = efficiency, n_reads=100, lam=lam)#ci dovrei mettere reco*purity, non reco_mc, ma qua è uguale

#chi2_fixed_toys(truth = truth, reco = reco_mc, response = response, efficiency = efficiency, n_toys=70, lam=lam)


"""

lam = lambda_optimizer(
    response=response,
    measured=reco,
    truth=truth_mc,
    binning=binning,
)
"""

lam = 0

unfolder = QUnfolder(response=response, measured=reco, binning=binning, lam=lam)
unfolder.initialize_qubo_model()
sol, cov = unfolder.solve_gurobi_integer()



factor = truth/truth_mc
reco_factor = reco_mc/reco

sol = sol*factor*reco_factor

reco = reco * factor*reco_factor


"""
sol = sol/np.sum(sol)
truth = truth/np.sum(truth)
reco = reco/np.sum(reco)
truth_mc = truth_mc/np.sum(truth_mc)
"""
#print(sol)
#print(efficiency)

#sol = sol/efficiency

#print("reco = ", reco)
#print("truth = ",truth)
#print("sol = ",sol)
#cov = cov/np.sum(sol)

plotter = QPlotter(
    response=response,
    measured=reco,
    truth=truth,
    unfolded=sol,
    covariance=cov,
    binning=binning,
    normed=True,
    method="SA"
)
plotter.save_response("response.png")
plotter.save_histograms("histo.png")






