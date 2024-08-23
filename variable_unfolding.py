import ROOT
import numpy as np
from qunfold.root2numpy import TH2_to_numpy, TH1_to_numpy, TVector_to_numpy
from qunfold.utils import normalize_response
from qunfold import QUnfolder, QPlotter
from utils import *

#######################################
# Pick a variable
variable = "c_thetap"
#######################################


# Load RooUnfold library
if ROOT.gSystem.Load("libRooUnfold"):
    raise ImportError("RooUnfold was not loaded successfully!")

# Open input ROOT file
root_file = ROOT.TFile.Open("ttbar_qunfold.root")

# Get binning for histograms
overflow = True
binning = get_binning(root_file=root_file, variable=variable, overflow=overflow)

# Read particle-level(=measured) histograms
dir = "particle"
th1_measured = root_file.Get(f"{dir}/{variable}")
th1_fake = root_file.Get(f"{dir}/{variable}_fake")
d = TH1_to_numpy(th1_measured, overflow=overflow)
fake = TH1_to_numpy(th1_fake, overflow=overflow)

# Read parton-level(=truth) histograms
dir = "parton"
th1_truth = root_file.Get(f"{dir}/{variable}_parton")
th1_miss = root_file.Get(f"{dir}/{variable}_miss")
t = TH1_to_numpy(th1_truth, overflow=overflow)
miss = TH1_to_numpy(th1_miss, overflow=overflow)

# Read Monte Carlo migration matrix (measured X truth)
dir = "migration"
th2_migration = root_file.Get(f"{dir}/{variable}_migration")
M = TH2_to_numpy(th2_migration, overflow=overflow)

# Get Monte Carlo reco and truth histograms
th1_proj_reco_mc = th2_migration.ProjectionX()
th1_reco_mc = th1_proj_reco_mc.Clone("th1_reco_mc")
th1_reco_mc.Add(th1_fake)
th1_proj_truth_mc = th2_migration.ProjectionY()
th1_truth_mc = th1_proj_truth_mc.Clone("th1_truth_mc")
th1_truth_mc.Add(th1_miss)
mu = TH1_to_numpy(th1_reco_mc, overflow=overflow)
x = TH1_to_numpy(th1_truth_mc, overflow=overflow)

# Normalize migration matrix in numpy
proj_truth_mc = TH1_to_numpy(th1_proj_truth_mc)
M = normalize_response(response=M, truth_mc=proj_truth_mc)

# Create response matrix object in RooUnfold
roo_response = ROOT.RooUnfoldResponse(th1_reco_mc, th1_truth_mc, th2_migration)

# Estimate purity from Monte Carlo data
purity = estimate_purity(reco=mu, fake=fake)
roo_purity = TVector_to_numpy(roo_response.Vpurity())

# Estimate efficiency from Monte Carlo data
efficiency = estimate_efficiency(truth=x, miss=miss)
roo_efficiency = TVector_to_numpy(roo_response.Vefficiency())

# Sanity checks
assert np.allclose(np.sum(M, axis=0), 1)
# assert np.allclose(purity[1:-1], roo_purity)
# assert np.allclose(efficiency[1:-1], roo_efficiency)
assert np.allclose(mu, M @ (x - miss) + fake)
assert np.allclose(mu, M @ (x * efficiency) / purity)

# Run Matrix Inversion unfolding in RooUnfold
unfolder = ROOT.RooUnfoldInvert("MI", "Matrix Inversion")
unfolder.SetVerbose(0)
unfolder.SetResponse(roo_response)
unfolder.SetMeasured(th1_measured)
roo_unfolded = TH1_to_numpy(unfolder.Hunfold(), overflow=overflow)

# Run Matrix Inversion unfolding in numpy
np_unfolded = (1 / efficiency) * (np.linalg.inv(M) @ (d * purity))

# Check MI_RooUnfold and MI_numpy solutions are the same
# assert np.allclose(roo_unfolded, np_unfolded)

# Run QUnfold algorithm
resp = M * efficiency
meas = d - fake
qunfolder = QUnfolder(response=resp, measured=meas, binning=binning, lam=0.0)
qunfolder.initialize_qubo_model()
sol, cov = qunfolder.solve_gurobi_integer()

# Plot QUnfold results
qplotter = QPlotter(
    response=M,
    measured=d,
    truth=t,
    unfolded=sol,
    covariance=cov,
    binning=binning,
    method="QUBO",
    norm=True,
)
qplotter.save_response(f"{variable}_response.pdf")
qplotter.save_histograms(f"{variable}_histograms.pdf")
