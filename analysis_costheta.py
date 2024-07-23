import ROOT
from QUnfold import QUnfoldQUBO, QUnfoldPlotter
import numpy as np
from QUnfold.utility import compute_chi2, normalize_response, TH1_to_numpy, TH2_to_numpy
import matplotlib.pyplot as plt
from analysis_function import get_purity, get_efficiency, gurobi, chi2_fixed_reads, chi2_fixed_toys

myFile_1 = ROOT.TFile.Open("hadpol_particle.root")
myFile_2 = ROOT.TFile.Open("hadpol_parton.root")

res_root = myFile_1.Get("costhetap_range800_matrix_hadpol")
reco_root = myFile_1.Get("costhetap_range800_particle_hadpol")
truth_root = myFile_2.Get("costhetap_range800_parton_hadpol") 

reco_mc_root=res_root.ProjectionX("reco_mc")
truth_mc_root=res_root.ProjectionY("truth_mc")
#purity and efficiency
purity = get_purity(reco_mc_root, reco_root)
efficiency = get_efficiency(truth_mc_root, truth_root)

#binning
min_bin = -1
max_bin = 1
num_bins = 10
bins = np.linspace(min_bin, max_bin, num_bins + 1)
binning = np.array(bins.tolist())

#TH1, TH2 to array
truth = TH1_to_numpy(truth_root, overflow =False)
truth_mc = TH1_to_numpy(truth_mc_root, overflow=False)
reco = TH1_to_numpy(reco_root, overflow =False)
reco_mc=TH1_to_numpy(reco_mc_root, overflow=False)
response = TH2_to_numpy(res_root, overflow=False)
response = normalize_response(TH2_to_numpy(res_root, overflow=False), truth_mc)

#find the best lambda and print
lam = gurobi(response = response, reco = reco_mc, truth = truth, efficiency = efficiency)#ci dovrei mettere reco*purity, non reco_mc, ma qua è uguale

#calculate chi2
chi2_fixed_reads(truth = truth, reco = reco_mc, response = response, efficiency = efficiency, n_reads=100, lam=lam)#ci dovrei mettere reco*purity, non reco_mc, ma qua è uguale

#chi2_fixed_toys(truth = truth, reco = reco_mc, response = response, efficiency = efficiency, n_toys=70, lam=lam)



