import ROOT
import numpy as np
from qunfold.root2numpy import TH2_to_numpy, TH1_to_numpy
from qunfold.utils import normalize_response
from qunfold import QPlotter
from utils import *


# Load RooUnfold library
roounfold_lib_path = "../../QUnfold/HEP_deps/RooUnfold/libRooUnfold.so"
roounfold_error = ROOT.gSystem.Load(roounfold_lib_path)
if roounfold_error:
    raise ImportError("RooUnfold was not loaded successfully!")

# Open input ROOT file
root_file = ROOT.TFile.Open("ttbar_qunfold.root")

# Read particle-level(=measured) histograms
dir = "particle"
th1_measured = root_file.Get(f"{dir}/c_thetap")
th1_fake = root_file.Get(f"{dir}/c_thetap_fake")

# Read parton-level(=truth) histograms
dir = "parton"
th1_truth = root_file.Get(f"{dir}/c_thetap_parton")
th1_miss = root_file.Get(f"{dir}/c_thetap_miss")

# Read Monte Carlo migration matrix (measured X truth)
dir = "migration"
th2_migration = root_file.Get(f"{dir}/c_thetap_migration")

# Get Monte Carlo reco and truth histograms
th1_reco_mc = th2_migration.ProjectionX()
th1_truth_mc = th2_migration.ProjectionY()

# Get binning for histograms
overflow = True
binning = get_binning(th1_measured, overflow=overflow)

# Convert ROOT objects into numpy arrays
d = TH1_to_numpy(th1_measured, overflow=overflow)
fake = TH1_to_numpy(th1_fake, overflow=overflow)
t = TH1_to_numpy(th1_truth, overflow=overflow)
miss = TH1_to_numpy(th1_miss, overflow=overflow)
mu = TH1_to_numpy(th1_reco_mc, overflow=overflow)
x = TH1_to_numpy(th1_truth_mc, overflow=overflow)
M = TH2_to_numpy(th2_migration, overflow=overflow)

# Normalize migration matrix by Monte Carlo truth histogram
M = normalize_response(response=M, truth_mc=x)
if overflow:
    M[0, 0] = M[-1, -1] = 1

# Check Σ(column)=1 for each column
assert np.allclose(np.sum(M, axis=0), 1)

# Chech μ=Mx for Monte Carlo
assert np.allclose(mu, M @ x)

# Estimate efficiency from Monte Carlo data
e = estimate_efficiency(reco=mu, miss=miss)
print("\nEfficiency =", e)

# Estimate purity from Monte Carlo data
p = estimate_purity(reco=mu, fake=fake)
print("\nPurity =", p)

# Run unfolding using matrix inversion in RooUnfold
roo_response = ROOT.RooUnfoldResponse(th1_reco_mc, th1_truth_mc, th2_migration)
unfolder = ROOT.RooUnfoldInvert("MI", "Matrix Inversion")
unfolder.SetVerbose(0)
unfolder.SetResponse(roo_response)
unfolder.SetMeasured(th1_measured)
roo_sol = TH1_to_numpy(unfolder.Hunfold(), overflow=overflow)

# Run unfolding computing the inverse matrix in numpy
np_sol = np.linalg.inv(M) @ d

# Check RooUnfold and numpy MI solutions are the same
assert np.allclose(roo_sol, np_sol)

# Show results using QUnfold plotter
dim = len(d)
qplotter = QPlotter(
    response=M,
    measured=d / np.sum(d),
    truth=t / np.sum(t),
    unfolded=np_sol / np.sum(np_sol),
    covariance=np.zeros(shape=(dim, dim)),  # temp
    binning=binning,
    method="TEMP",
    ybottom=0.045,
    normed=False,
)
qplotter.show_response()
qplotter.show_histograms()
